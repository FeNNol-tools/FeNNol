"""Training dataset readers: HTTP streams and Parquet files."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("torch", reason="training requires PyTorch")

from fennol.training.config import TrainingConfig
from fennol.training.io import load_dataset


@pytest.mark.parametrize("split_inputs", [False, True])
def test_http_dataset_stream_preserves_case_sensitive_endpoint_path(split_inputs):
    expected_path = "/DataSet/MiXeD/?val=false&batch=1"
    row = {"species": [1], "coordinates": [[0.0, 0.0, 0.0]], "forces": [[1.0, 2.0, 3.0]]}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            status = 200 if self.path == expected_path else 404
            body = json.dumps([row] if status == 200 else {"error": "not found"})
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body.encode())

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        address = f"http://127.0.0.1:{server.server_port}/DataSet/MiXeD"
        loader = load_dataset(
            dspath=address,
            batch_size=1,
            training_config=TrainingConfig(),
            train_val_split=False,
            split_data_inputs=split_inputs,
            ref_keys=["forces"] if split_inputs else None,
        )
        loader.multiprocessing_context = "spawn"
        batch = next(iter(loader))
        inputs, refs = batch if split_inputs else (batch, batch)
        np.testing.assert_array_equal(inputs["species"], [1])
        np.testing.assert_array_equal(refs["forces"], [[1.0, 2.0, 3.0]])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("load_mode", ["inmemory", "on_demand"])
def test_parquet_batches_mix_periodic_and_nonperiodic_systems(tmp_path, load_mode):
    parquet_path = tmp_path / "mixed-pbc.parquet"
    periodic_cell = [[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]]
    rows = [
        {"species": [1], "coordinates": [[0.1, 0.0, 0.0]], "total_charge": 0, "cell": periodic_cell, "split": "training"},
        {"species": [1], "coordinates": [[0.2, 0.0, 0.0]], "total_charge": 0, "cell": None, "split": "training"},
        {"species": [1], "coordinates": [[0.3, 0.0, 0.0]], "total_charge": 0, "cell": periodic_cell, "split": "validation"},
    ]
    pq.write_table(pa.Table.from_pylist(rows), parquet_path)
    config = TrainingConfig(
        pbc_training=True,
        load_mode=load_mode,
        shuffle_dataset=False,
    )

    training, validation = load_dataset(
        dspath=str(parquet_path),
        batch_size=2,
        training_config=config,
        train_val_split=True,
        infinite_iterator=False,
    )

    training_batch = next(iter(training))
    validation_batch = next(iter(validation))
    np.testing.assert_array_equal(training_batch["cells_mask"], [True, False])
    np.testing.assert_allclose(training_batch["cells"][0], periodic_cell)
    np.testing.assert_allclose(training_batch["cells"][1], np.eye(3) * config.length_nopbc)
    np.testing.assert_array_equal(validation_batch["cells_mask"], [True])
