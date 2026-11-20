import io
import logging
import os
import time
from collections.abc import Callable, Iterable, Sequence
from typing import Literal

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from torch.utils.data import Dataset, IterableDataset
from urllib.request import urlopen
import json

logger = logging.getLogger(__name__)

try:
    import sqlite3

    def convert_array(text):
        out = io.BytesIO(text)
        out.seek(0)
        return np.load(out)
    def adapt_array(arr):
        """
        http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
        """
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)
        return sqlite3.Binary(out.read())

    sqlite3.register_adapter(np.ndarray, adapt_array)
    sqlite3.register_converter("array", convert_array)
except ImportError:
    sqlite3 = None




class DBDataset:
    def __init__(self,dbfile,table="training",select_keys=None):
        if sqlite3 is None:
            raise ImportError("sqlite3 is not available")
        self.con = sqlite3.connect(dbfile, detect_types=sqlite3.PARSE_DECLTYPES)
        self.cur = self.con.cursor()
        self.table = table
        self.keys = [k[1] for k in self.cur.execute(f"PRAGMA table_info({self.table})")]
        if select_keys is not None:
            self.select_keys = [k for k in self.keys if k in select_keys]
        else:
            self.select_keys = self.keys
        self.N = self.cur.execute(f"SELECT MAX(rowid) FROM {self.table}").fetchone()[0]

    def __len__(self):
        return self.N

    def __getitem__(self,idx):
        if isinstance(idx,Iterable):
            indices = [i for i in idx]
        elif isinstance(idx,slice):
            indices = range(*idx.indices(self.N))
        else:
            indices = [idx]

        indices = [str(i+1) if i>=0 else str(i+self.N+1) for i in indices]
        query = f"SELECT {', '.join(self.select_keys)} FROM {self.table} WHERE rowid IN ({', '.join(indices)})"
        if len(indices)==1:
            data = self.cur.execute(query).fetchone()
            return {k:d for k,d in zip(self.select_keys,data)}
        else:
            return [{k:d for k,d in zip(self.select_keys,row)} for row in self.cur.execute(query)]

    def __iter__(self):
        for i in range(self.N):
            yield self[i]

    def __del__(self):
        self.con.close()

    def __repr__(self):
        return f"DBDataset({self.table})"

    def __str__(self):
        return f"DBDataset({self.table}) with {self.N} entries"

class H5Dataset:
    def __init__(self,h5file,table="training"):
        self.h5file = h5file
        self.table = table
        self.f = h5py.File(self.h5file,"r")
        self.dataset = self.f[self.table]
        self.N = len(self.dataset)
        self.keys = list(self.dataset["0"].keys())

    def __len__(self):
        return self.N

    def __getitem__(self,idx):
        if isinstance(idx,Iterable):
            indices = [i for i in idx]
        elif isinstance(idx,slice):
            indices = range(*idx.indices(self.N))
        else:
            indices = [idx]

        indices = [str(i) for i in indices]
        if len(indices)==1:
            data = self.dataset[indices[0]]
            return {k:data[k][()] for k in self.keys}
        else:
            return [{k:data[k][()] for k in self.keys} for data in self.dataset[indices]]

    def __iter__(self):
        for i in range(self.N):
            yield self[i]

    def __del__(self):
        self.f.close()


class HTTPDataset(IterableDataset):
    """Stream batches from a training dataset HTTP endpoint."""

    def __init__(self, address, val=False, batch_size=1):
        super().__init__()
        self.address = address
        self.val = val
        self.batch_size = batch_size
        self.request = (
            self.address + f"/?val={str(self.val).lower()}&batch={self.batch_size}"
        )

    def __iter__(self):
        while True:
            data = json.loads(urlopen(self.request).read())
            for row in data:
                yield {k: np.asarray(v) if isinstance(v, list) else v for k, v in row.items()}


def build_split_indices(parquet_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Build global row indices for the "training" and "validation" splits from a Parquet file or dataset directory.

    This scans the Parquet data lazily (batch-by-batch) using `pyarrow.dataset`,
    looks up the `split` column, and returns two NumPy index arrays containing
    the 0-based row positions where `split == "training"` and `split == "validation"`
    respectively. Indices are computed relative to the concatenation of all
    scanned batches (i.e., global row numbering).

    Parameters
    ----------
    parquet_path : str
        Path to a Parquet file or a directory of Parquet files (partitioned or flat).

    Returns
    -------
    (np.ndarray, np.ndarray)
        A tuple `(train_idx, val_idx)` where:
        - `train_idx` is a 1-D `int64` array of global row indices with `split == "training"`.
        - `val_idx`   is a 1-D `int64` array of global row indices with `split == "validation"`.
        Arrays may be empty if no rows match a given split.

    Notes
    -----
    - This function reads only the `split` column (projection) and processes data
      in record batches, keeping memory usage low even for large datasets.
    - The `split` values are compared as strings; they must match exactly
      `"training"` or `"validation"`.
    - If your dataset is partitioned by `split/=` on disk, consider using directory
      pruning (e.g., separate scans per partition) for even faster index building.

    Examples
    --------
    >>> train_idx, val_idx = build_split_indices("data.parquet")
    >>> train_idx.dtype, val_idx.dtype
    (dtype('int64'), dtype('int64'))

    """
    logger.info("building split indices start")
    dset = ds.dataset(parquet_path, format="parquet")
    scan = dset.scanner(columns=["split"], use_threads=True)

    out_train, out_val = [], []
    base = 0
    for batch in scan.to_batches():
        col = batch.column(0)
        if not pa.types.is_string(col.type):
            col = pc.cast(col, pa.string())
        mt = pc.equal(col, "training")
        nz_t = np.flatnonzero(mt.to_numpy(zero_copy_only=False))
        if nz_t.size:
            out_train.append(nz_t + base)
        mv = pc.equal(col, "validation")
        nz_v = np.flatnonzero(mv.to_numpy(zero_copy_only=False))
        if nz_v.size:
            out_val.append(nz_v + base)

        base += len(col)

    logger.info("building split indices done")
    train_idx = np.concatenate(out_train) if out_train else np.empty(0, dtype=np.int64)
    val_idx   = np.concatenate(out_val)   if out_val   else np.empty(0, dtype=np.int64)
    return train_idx, val_idx


class ParquetDataset(Dataset):
    """
    Parquet-backed dataset with selectable columns and two access modes.

    The dataset returns Python dicts per row with NumPy arrays / scalars.

    Mask convention
    ---------------
    If a requested column ``c`` has a sibling column named ``c + "_mask"``
    present in the Parquet schema, that mask column is automatically added to
    the projection (immediately after ``c``). Missing mask columns are ignored.

    Parameters
    ----------
    parquet_path : str
        Path to a local Parquet file.
    columns : Sequence[str], optional
        Subset of columns to project and return. If ``None``, uses the optimized
        set listed above. Columns must exist in the file.
    energy_dtype : {"float32", "float64"}, default "float32"
        Target dtype for energy scalar fields.
    mode : {"inmemory", "on_demand"}, default "inmemory"
        Loading strategy. "inmemory" reads and converts all rows at construction.
        "on_demand" keeps only one row-group cached and converts rows on access.
    chunk_size : int, default 4096
        Temporary batch size for the in-memory materialization loop.

    Raises
    ------
    ValueError
        If the Parquet file is missing, requested columns are absent,
        or no rows exist.
    """

    _OPTIMIZED_COLS: list[str] = [
        "species", "coordinates", "forces", "total_charge",
        "formation_energy", "internal_energy", "interaction_energy",
    ]

    def __init__(
        self,
        parquet_path: str,
        *,
        columns: Sequence[str] | None = None,
        energy_dtype: str = "float32",
        mode: Literal["inmemory", "on_demand"] = "inmemory",
        chunk_size: int = 16384,
    ):
        """
        Initialize the dataset and either inmemory or prepare on-demand access.

        Notes
        -----
        - Column projection is computed from the **logical Arrow schema**.
          For each requested column `c`, `c + "_mask"` is auto-included if present.
        - In "inmemory" mode, all projected columns are read and converted into
          a Python list of dicts.
        - In "on_demand" mode, nothing is materialized until the first
          ``__getitem__``; only one row-group is cached at a time.
        """
        if not os.path.exists(parquet_path):
            raise ValueError(f"Parquet file not found: {parquet_path}")
        if energy_dtype not in ("float32", "float64"):
            raise ValueError("energy_dtype must be 'float32' or 'float64'")

        self._parquet_path = parquet_path
        self._mode = mode
        self._energy_dtype = np.float32 if energy_dtype == "float32" else np.float64

        self._pf = pq.ParquetFile(self._parquet_path)
        logical_schema = self._pf.schema_arrow
        self._dtype_map = self._infer_dtype_map(logical_schema)
        schema_names = set(logical_schema.names)
        requested = list(columns) if columns is not None else self._OPTIMIZED_COLS[:]
        self._cols_order = self._compute_projection_columns(requested, schema_names)
        logger.info(f"Requested columns parquet: {self._cols_order}")
        print("Requested columns parquet:", self._cols_order)

        # Converters are built once and reused by both modes
        self._converters: dict[str, Callable[[object], object]] = self._make_converters()

        # State shared across modes
        self._rows: list[dict] | None = None
        self._rg_offsets: np.ndarray | None = None
        self._cached_rg: int = -1
        self._cached_rb: pa.RecordBatch | None = None
        self._cached_cols: dict[str, Sequence[object]] | None = None

        if mode == "inmemory":
            t0 = time.perf_counter()
            try:
                table = pq.read_table(
                    self._parquet_path,
                    columns=self._cols_order,
                    memory_map=True,
                    use_threads=True,
                )
            except KeyError as e:
                raise ValueError(f"Requested columns not found: {e}") from e

            self._nrows = int(table.num_rows)
            if self._nrows == 0:
                raise ValueError(f"No rows found for '{self._parquet_path}'")
            self._cols = set(table.column_names)

            rows: list[dict] = []
            max_chunksize = int(chunk_size) if chunk_size and chunk_size > 0 else None
            for rb in table.to_batches(max_chunksize=max_chunksize):
                cols_data, nrows = self._materialize_record_batch(rb)
                rows.extend(self._rows_from_cols(cols_data, nrows))
            self._rows = rows

            dt = time.perf_counter() - t0
            logger.info(
                "inmemory load done: rows=%d cols=%s time=%.2fs rate=%.1fk/s",
                self._nrows, self._cols_order, dt, (self._nrows / dt) / 1e3 if dt else 0.0
            )
            print(
                f"Parquet inmemory load: rows={self._nrows} cols={len(self._cols_order)} "
                f"time={dt:.2f}s rate={(self._nrows / dt) / 1e3 if dt else 0.0:.1f}k/s"
            )

            self._rg_offsets = np.array([0, self._nrows], dtype=np.int64)

        else:
            # On-demand: open file, validate columns, precompute row-group offsets
            self._cols = set(logical_schema.names)

            missing = [c for c in self._cols_order if c not in self._cols]
            if missing:
                raise ValueError(f"Columns not found in parquet: {missing}")

            md = self._pf.metadata
            nrg = md.num_row_groups
            rg_sizes = np.fromiter((md.row_group(i).num_rows for i in range(nrg)),
                                   dtype=np.int64, count=nrg)
            self._rg_offsets = np.concatenate(([0], np.cumsum(rg_sizes)))
            self._nrows = int(self._rg_offsets[-1])
            if self._nrows == 0:
                raise ValueError(f"No rows found for '{self._parquet_path}'")

            logger.info(
                "on_demand ready: rows=%d rgs=%d cols=%s",
                self._nrows, nrg, self._cols_order
            )

    def _compute_projection_columns(
            self,
            requested: Sequence[str],
            schema_names: set[str]
        ) -> list[str]:
        """
        Compute the final column projection in a deterministic order.

        For each base column in ``requested``, verifies its presence in the schema
        (raises if missing) and appends a sibling ``<name> + "_mask"`` *if and only if*
        it is present in the schema. Duplicate names are removed while preserving the
        (base, mask) adjacency.

        Parameters
        ----------
        requested : Sequence[str]
            Base columns requested by the user or defaults.
        schema_names : set[str]
            Set of top-level logical column names from the Parquet schema.

        Returns
        -------
        list[str]
            Ordered list of columns to read, with any available mask columns
            inserted immediately after their corresponding base column.

        Raises
        ------
        ValueError
            If any requested base column is not present in the schema.
        """
        out: list[str] = []
        seen: set[str] = set()

        for base in requested:
            if base not in schema_names:
                raise ValueError(f"Requested column not found in parquet: '{base}'")

            if base not in seen:
                out.append(base)
                seen.add(base)

            mask = f"{base}_mask"
            if mask in schema_names and mask not in seen:
                out.append(mask)
                seen.add(mask)

        return out

    def _make_converters(self) -> dict[str, Callable[[object], object]]:
        """
        Build per-column conversion callables.

        Returns
        -------
        dict
            Mapping from column name to a function that converts a Python value
            (produced by ``pyarrow.Scalar.as_py()``) into the desired NumPy
            array or scalar dtype.

        Notes
        -----
        Converters enforce for optimized columns:
            - species -> int64 ndarray
            - coordinates/forces -> float32 ndarray
            - total_charge -> int64 scalar
            - energy fields -> ``energy_dtype`` scalar
        Mask columns (``*_mask``) are converted to boolean ndarrays.
        All other columns use a generic `np.asarray` fallback.
        """
        def default(x):  # fallback for arbitrary columns
            return np.asarray(x)

        def _asarray_dtype(dtype):
            return lambda x, _dt=dtype: np.asarray(x, dtype=_dt)

        conv: dict[str, Callable[[object], object]] = {
            name: _asarray_dtype(dt)
            for name, dt in self._dtype_map.items()
            if dt is not None
        }

        class DefaultDict(dict):
            def __missing__(self, key):
                if key.endswith("_mask"):
                    return lambda x: np.asarray(x, dtype=bool)
                return default

        return DefaultDict(conv)  # type: ignore[return-value]

    def _convert_record_batch_columns(self, rb: pa.RecordBatch) -> dict[str, Sequence[object]]:
        """
        Materialize all projected columns of a RecordBatch once.

        Returns a mapping of column name to a list/ndarray of converted
        values (already run through the per-column converters). This avoids
        calling ``as_py`` for every ``__getitem__`` access to the same
        row-group.
        """
        out: dict[str, Sequence[object]] = {}
        for name in self._cols_order:
            try:
                col = rb.column(name)
            except KeyError:
                raise KeyError(f"Column {name!r} not in batch schema")
            target_dtype = self._dtype_map.get(name)
            conv = self._converters[name]

            # Primitive fast path
            if not pa.types.is_list(col.type) and not pa.types.is_large_list(col.type):
                np_col = col.to_numpy(zero_copy_only=False)
                if target_dtype is not None:
                    out[name] = np.asarray(np_col, dtype=target_dtype)
                else:
                    out[name] = np.asarray(np_col)
                continue

            # List fast path
            if pa.types.is_list(col.type) or pa.types.is_large_list(col.type):
                out[name] = self._split_list_array(col, name=name)
                continue

            # Generic/list path: convert once per element
            values = col.to_pylist()
            out[name] = [conv(v) for v in values]

        return out

    def _split_list_array(self, arr: pa.Array, *, name: str) -> list[np.ndarray]:
        """
        Split a (large) list array into NumPy chunks per parent row.

        Handles:
            - list<primitive>
            - list<fixed_size_list<primitive>>
            - list<list<primitive>> with fixed inner length (fast path)
            - falls back to per-row conversion for ragged / deeper nesting.
        """
        offsets = np.asarray(arr.offsets.to_numpy(), dtype=np.int64)
        values = arr.values  # child of the outer list

        value_type = values.type
        # For nested fixed-size lists, inspect the primitive value type
        if pa.types.is_fixed_size_list(value_type):
            value_type = value_type.value_type

        target_dtype = self._dtype_map.get(name) or self._arrow_primitive_dtype(value_type)

        def _cast_numpy(a: np.ndarray) -> np.ndarray:
            if target_dtype is None:
                return np.asarray(a)
            a = np.asarray(a)
            if a.dtype == object:
                raise TypeError("object-dtype array; cannot safely cast to fixed dtype")
            return a.astype(target_dtype, copy=False)

        if pa.types.is_list(values.type) or pa.types.is_large_list(values.type):
            inner_child = values.values  # child of inner list
            if pa.types.is_list(inner_child.type) or pa.types.is_large_list(inner_child.type):
                py = arr.to_pylist()
                return [
                    np.asarray(v, dtype=target_dtype) if target_dtype is not None else np.asarray(v)
                    for v in py
                ]
            lengths = pc.list_value_length(values)
            if lengths.null_count == 0:
                uniques = pc.unique(lengths)
                unique_lengths = uniques.to_pylist()
                if len(unique_lengths) == 1:
                    inner_size = int(unique_lengths[0])

                    flat_inner = inner_child.to_numpy(zero_copy_only=False)
                    try:
                        flat_inner = _cast_numpy(flat_inner)
                    except TypeError:
                        py = arr.to_pylist()
                        return [
                            np.asarray(v, dtype=target_dtype) if target_dtype is not None else np.asarray(v)
                            for v in py
                        ]
                    flat_inner = flat_inner.reshape(len(values), inner_size)
                    if offsets.size == 2:
                        return [flat_inner]
                    return list(np.split(flat_inner, offsets[1:-1]))
            py = arr.to_pylist()
            return [
                np.asarray(v, dtype=target_dtype) if target_dtype is not None else np.asarray(v)
                for v in py
            ]

        if pa.types.is_fixed_size_list(values.type):
            inner = values.type.list_size
            flat = values.values.to_numpy(zero_copy_only=False)
            flat = _cast_numpy(flat)
            flat = flat.reshape(len(values), inner)
            if offsets.size == 2:
                return [flat]
            return list(np.split(flat, offsets[1:-1]))

        flat = values.to_numpy(zero_copy_only=False)
        flat = _cast_numpy(flat)
        if offsets.size == 2:
            return [flat]
        return list(np.split(flat, offsets[1:-1]))


    @staticmethod
    def _arrow_primitive_dtype(
        pa_type: pa.DataType
        ) -> np.dtype | type[bool] | None:
        """Map an Arrow primitive type to a NumPy dtype for fast conversion."""
        if pa.types.is_integer(pa_type):
            return np.int64
        if pa.types.is_floating(pa_type):
            return np.float32
        if pa.types.is_boolean(pa_type):
            return bool
        return None

    def _infer_dtype_map(
            self,
            schema: pa.Schema
        ) -> dict[str, np.dtype | type[bool] | None]:
        """Derive a preferred NumPy dtype per column from the Arrow schema."""
        out: dict[str, np.dtype | type[bool] | None] = {}
        for field in schema:
            out[field.name] = self._infer_dtype_from_type(field.type, field.name)
        return out

    def _infer_dtype_from_type(
            self,
            pa_type: pa.DataType,
            name: str
        ) -> np.dtype | type[bool] | None:
        """Infer a NumPy dtype from an Arrow type, with light name-based hints."""
        if name.endswith("_mask"):
            return bool
        if "energy" in name and pa.types.is_floating(pa_type):
            return self._energy_dtype
        if pa.types.is_list(pa_type) or pa.types.is_large_list(pa_type):
            return self._infer_dtype_from_type(pa_type.value_type, name)
        if pa.types.is_fixed_size_list(pa_type):
            return self._infer_dtype_from_type(pa_type.value_type, name)
        return self._arrow_primitive_dtype(pa_type)

    def _materialize_record_batch(self, rb: pa.RecordBatch) -> tuple[dict[str, Sequence[object]], int]:
        """Convert a RecordBatch into column data once and return the row count."""
        cols_data = self._convert_record_batch_columns(rb)
        return cols_data, rb.num_rows

    def _rows_from_cols(self, cols_data: dict[str, Sequence[object]], nrows: int) -> list[dict]:
        """Build row dictionaries from already converted column data."""
        return [
            {name: cols_data[name][i] for name in self._cols_order}
            for i in range(nrows)
        ]

    def _ensure_pf(self) -> None:
        """
        Ensure a ``pyarrow.parquet.ParquetFile`` handle is open.

        Notes
        -----
        Used by on-demand mode to lazily (re)open the Parquet file after
        pickling or process forks.
        """
        if self._pf is None:
            self._pf = pq.ParquetFile(self._parquet_path)

    def _ensure_rowgroup_loaded(self, rg: int) -> None:
        """
        Load a specific row-group into the cache as a single RecordBatch.

        Parameters
        ----------
        rg : int
            Zero-based row-group index to load.
        """
        self._ensure_pf()

        if rg == self._cached_rg:
            return

        batch_iter = self._pf.iter_batches(
            row_groups=[rg],
            columns=self._cols_order,
            batch_size=int(self._rg_offsets[rg + 1] - self._rg_offsets[rg]),
            use_threads=True,
        )

        first_batch = next(batch_iter, None)

        if first_batch is None or first_batch.num_rows == 0:
            self._cached_rb = None
            self._cached_cols = None
        else:
            self._cached_rb = first_batch
            self._cached_cols, _ = self._materialize_record_batch(first_batch)

        self._cached_rg = rg

    def __getstate__(self) -> dict:
        """
        Support pickling by dropping non-picklable/transient members.

        Returns
        -------
        dict
            Serializable state dictionary with file/batch handles cleared.
        """
        state = self.__dict__.copy()
        state["_pf"] = None
        state["_cached_rb"] = None
        state["_cached_cols"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        """
        Restore state after unpickling and reset transient caches.

        Parameters
        ----------
        state : dict
            State dictionary produced by ``__getstate__``.
        """
        self.__dict__.update(state)
        self._cached_rg = -1
        self._cached_rb = None
        self._cached_cols = None
        if self._mode == "on_demand":
            self._pf = pq.ParquetFile(self._parquet_path)

    def __len__(self) -> int:
        """
        Return number of rows in the dataset.

        Returns
        -------
        int
            Total row count across all row-groups.
        """
        return self._nrows

    def __getitem__(self, idx: int) -> dict:
        """
        Return a single row by global index.

        Parameters
        ----------
        idx : int
            Zero-based global row index. Negative indices are supported,
            interpreted relative to the end (like Python sequences).

        Returns
        -------
        dict
            Row mapping of requested column names to NumPy arrays/scalars.

        Raises
        ------
        IndexError
            If ``idx`` is out of range.
        RuntimeError
            If a row-group fails to load in on-demand mode.
        """
        import numbers

        if not isinstance(idx, numbers.Integral):
            raise TypeError(f"Row index must be an integer, got {type(idx)!r}")

        # Normalize negative index
        if idx < 0:
            idx += self._nrows
        if not (0 <= idx < self._nrows):
            raise IndexError(idx)

        if self._rows is not None:
            return self._rows[idx]

        rg_offsets = self._rg_offsets
        rg = rg_offsets.searchsorted(idx, side="right") - 1
        rg = int(rg)

        local_idx = idx - int(rg_offsets[rg])

        self._ensure_rowgroup_loaded(rg)
        cols = self._cached_cols
        if cols is None:
            raise RuntimeError("Failed to load row group")
        return {name: cols[name][local_idx] for name in self._cols_order}
