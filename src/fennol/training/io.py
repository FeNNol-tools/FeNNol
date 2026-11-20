import glob
import io
import os
import pickle
import re
import sys
import logging
from collections import defaultdict
from typing import Optional, Sequence, Any, Callable, Literal

import numpy as np
from flax import traverse_util
from scipy.spatial.transform import Rotation

from .databases import DBDataset, H5Dataset, HTTPDataset, ParquetDataset, build_split_indices
from .config import TrainingConfig
from ..models.preprocessing import AtomPadding

logger = logging.getLogger(__name__)

try:
    import tomlkit
except ImportError:
    tomlkit = None

try:
    from torch.utils.data import DataLoader, Subset
except ImportError:
    raise ImportError(
        "PyTorch is required for training models. Install the CPU version following instructions at https://pytorch.org/get-started/locally/"
    )

from ..models import FENNIX


def _identity_batch(batch):
    return batch


class _PrefetchingDataLoader(DataLoader):
    """Fetch batches in a worker and apply preprocessing in the caller."""

    def __init__(self, *args, collate_fn, **kwargs):
        super().__init__(*args, collate_fn=_identity_batch, **kwargs)
        self._parent_collate = collate_fn

    def __iter__(self):
        for batch in super().__iter__():
            yield self._parent_collate(batch)


def load_dataset(
    dspath: str,
    batch_size: int,
    batch_size_val: Optional[int] = None,
    rename_refs: dict = {},
    infinite_iterator: bool = False,
    atom_padding: bool = False,
    ref_keys: Optional[Sequence[str]] = None,
    split_data_inputs: bool = False,
    np_rng: Optional[np.random.Generator] = None,
    train_val_split: bool = True,
    training_config: TrainingConfig | None = None,
    add_flags: Sequence[str] = ["training"],
    fprec: str = "float32",
    log_dataset: Callable | None = None,
    load_mode: Literal["inmemory", "on_demand"] = "on_demand",
    load_chunk_size: int = 4096,
):
    """
    Load a dataset from a pickle file and return two iterators for training and validation batches.

    Args:
        training_config (TrainingConfig): Validated training options.
        rename_refs (list, optional): A list of strings with the names of the reference properties to rename.
            Default is an empty list.
        log_dataset (Callable): Method to register dataset to MLflow.

    Returns:
        tuple: A tuple of two infinite iterators, one for training batches and one for validation batches.
            For each element in the batch, we expect a "species" key with the atomic numbers of the atoms in the system. Arrays are concatenated along the first axis and the following keys are added to distinguish between the systems:
            - 'natoms': np.ndarray. Array with the number of atoms in each system.
            - 'batch_index': np.ndarray. Array with the index of the system to which each atom
            if the keys "forces", "total_energy", "atomic_energies" or any of the elements in rename_refs are present, the keys are renamed by prepending "true_" to the key name.
    """
    # default dataloader values
    num_workers = 0
    prefetch_factor = None
    persistent_workers = False

    if training_config is None:
        training_config = TrainingConfig()
    assert isinstance(
        rename_refs, dict
    ), "rename_refs must be a dictionary with the keys to rename."

    pbc_training = training_config.pbc_training
    minimum_image = training_config.minimum_image

    coordinates_ref_key = training_config.coordinates_ref_key

    input_keys = [
        "species",
        "coordinates",
        "natoms",
        "batch_index",
        "total_charge",
        "flags",
    ]
    input_file_keys = {
        "species",
        "coordinates",
        "total_charge",
    }
    if pbc_training:
        input_keys += ["cells", "reciprocal_cells", "cells_mask"]
        input_file_keys.add("cell")
    if atom_padding:
        input_keys += ["true_atoms", "true_sys"]
    if coordinates_ref_key is not None:
        input_keys += ["system_index", "system_sign"]
        input_file_keys.add(coordinates_ref_key)

    flags = {f: None for f in add_flags}
    if minimum_image and pbc_training:
        flags["minimum_image"] = None

    additional_input_keys = set(training_config.additional_input_keys)
    additional_input_keys_ = set()
    for key in additional_input_keys:
        if key not in input_keys:
            additional_input_keys_.add(key)
    additional_input_keys = additional_input_keys_

    input_file_keys.update(additional_input_keys)
    all_inputs = set(input_keys + list(additional_input_keys))

    extract_all_keys = ref_keys is None
    if ref_keys is not None:
        ref_keys = set(ref_keys)
        ref_keys_ = set()
        for key in ref_keys:
            if key not in all_inputs:
                ref_keys_.add(key)
        input_file_keys.update(ref_keys)
    input_file_keys = list(input_file_keys)

    random_rotation = training_config.random_rotation
    if random_rotation:
        assert np_rng is not None, "np_rng must be provided for adding noise."

        apply_rotation = {
            1: lambda x, r: x @ r,
            -1: lambda x, r: np.einsum("...kn,kj->...jn", x, r),
            2: lambda x, r: np.einsum("li,...lk,kj->...ij", r, x, r),
        }
        def rotate_2f(x,r):
            assert x.shape[-1]==6
            # select from 6 components (xx,yy,zz,xy,xz,yz) to form the 3x3 tensor
            indices = np.array([0,3,4,3,1,5,4,5,2])
            x=x[...,indices].reshape(*x.shape[:-1],3,3)
            x=np.einsum("li,...lk,kj->...ij", r, x, r)
            # select back the 6 components
            indices = np.array([[0,0],[1,1],[2,2],[0,1],[0,2],[1,2]])
            x=x[...,indices[:,0],indices[:,1]]
            return x
        apply_rotation[-2]=rotate_2f

        valid_rotations = tuple(apply_rotation.keys())
        rotated_keys = {
            "coordinates": 1,
            "forces": 1,
            "virial_tensor": 2,
            "stress_tensor": 2,
            "pressure_tensor": 2,
            "virial": 2,
            "stress": 2,
        }
        if pbc_training:
            rotated_keys["cells"] = 1
        user_rotated_keys = dict(training_config.rotated_keys)
        for k, v in user_rotated_keys.items():
            assert (
                v in valid_rotations
            ), f"Invalid rotation type for key {k}. Valid values are {valid_rotations}"
            rotated_keys[k] = v

        print(
            "Applying random rotations to the following keys if present:",
            list(rotated_keys.keys()),
        )

        def apply_random_rotations(output, nbatch):
            euler_angles = np_rng.uniform(0.0, 2 * np.pi, (nbatch, 3))
            r = [
                Rotation.from_euler("xyz", euler_angles[i]).as_matrix().T
                for i in range(nbatch)
            ]
            for k, l in rotated_keys.items():
                if k in output:
                    for i in range(nbatch):
                        output[k][i] = apply_rotation[l](output[k][i], r[i])

    else:

        def apply_random_rotations(output, nbatch):
            pass

    flow_matching = training_config.flow_matching
    if flow_matching:
        if ref_keys is not None:
            ref_keys.add("flow_matching_target")
            if "flow_matching_target" in ref_keys_:
                ref_keys_.remove("flow_matching_target")
        all_inputs.add("flow_matching_time")

        def add_flow_matching(output, nbatch):
            ts = np_rng.uniform(0.0, 1.0, (nbatch,))
            targets = []
            for i in range(nbatch):
                x1 = output["coordinates"][i]
                com = x1.mean(axis=0, keepdims=True)
                x1 = x1 - com
                x0 = np_rng.normal(0.0, 1.0, x1.shape)
                xt = (1 - ts[i]) * x0 + ts[i] * x1
                output["coordinates"][i] = xt
                targets.append(x1 - x0)
            output["flow_matching_target"] = targets
            output["flow_matching_time"] = [np.array(t) for t in ts]

    else:

        def add_flow_matching(output, nbatch):
            pass

    if pbc_training:
        print("Periodic boundary conditions are active.")
        length_nopbc = training_config.length_nopbc

        def add_cell(d, output):
            cell_value = d.get("cell")
            if cell_value is None or np.asarray(cell_value).size == 0:
                cell = np.asarray(
                    [
                        [length_nopbc, 0.0, 0.0],
                        [0.0, length_nopbc, 0.0],
                        [0.0, 0.0, length_nopbc],
                    ],
                    dtype=fprec,
                )
                cells_mask = False
            else:
                cell = np.asarray(cell_value, dtype=fprec)
                cells_mask = True
            output["cells"].append(cell.reshape(1, 3, 3))
            output["cells_mask"].append(np.asarray(cells_mask, dtype=bool))

    else:

        def add_cell(d, output):
            if "cell" in d:
                print(
                    "Warning: 'cell' found in dataset but not training with pbc. Activate pbc_training to use periodic boundary conditions."
                )

    if extract_all_keys:

        def add_other_keys(d, output, atom_shift):
            for k, v in d.items():
                if k in ("cell", "total_charge"):
                    continue
                v_array = np.array(v)
                # Shift atom number if necessary
                if k.endswith("_atidx"):
                    mask = (v_array > 0).astype(int)
                    v_array = v_array + atom_shift*mask
                output[k].append(v_array)

    else:

        def add_other_keys(d, output, atom_shift):
            output["species"].append(np.asarray(d["species"]))
            output["coordinates"].append(np.asarray(d["coordinates"], dtype=fprec))
            for k in additional_input_keys:
                v_array = np.array(d[k])
                # Shift atom number if necessary
                if k.endswith("_atidx"):
                    mask = (v_array > 0).astype(int)
                    v_array =v_array + atom_shift*mask
                output[k].append(v_array)
            for k in ref_keys_:
                v_array = np.array(d[k])
                # Shift atom number if necessary
                if k.endswith("_atidx"):
                    mask = (v_array > 0).astype(int)
                    v_array =v_array + atom_shift*mask
                output[k].append(v_array)
                if k + "_mask" in d:
                    output[k + "_mask"].append(np.asarray(d[k + "_mask"]))

    def add_keys(d, output, atom_shift, batch_index):
        nat = d["species"].shape[0]

        output["natoms"].append(np.asarray([nat]))
        output["batch_index"].append(np.asarray([batch_index] * nat))
        if "total_charge" not in d:
            total_charge = np.asarray(0.0, dtype=fprec)
        else:
            total_charge = np.asarray(d["total_charge"], dtype=fprec)
        output["total_charge"].append(total_charge)

        add_cell(d, output)
        add_other_keys(d, output, atom_shift)

        return atom_shift + nat

    def collate_fn_(batch):
        output = defaultdict(list)
        atom_shift = 0
        batch_index = 0

        for d in batch:
            atom_shift = add_keys(d, output, atom_shift, batch_index)
            batch_index += 1

            if coordinates_ref_key is not None:
                output["system_index"].append(np.asarray([batch_index - 1]))
                output["system_sign"].append(np.asarray([1]))

                cref = d.get(coordinates_ref_key, None)
                if cref is not None:
                    cref_arr = np.asarray(cref)
                    if cref_arr.ndim == 2 and cref_arr.shape[
                        -1] == 3 and cref_arr.size > 0:
                        dref = {**d, "coordinates": cref_arr}
                        atom_shift = add_keys(dref, output, atom_shift,
                                              batch_index)
                        output["system_index"].append(
                            np.asarray([batch_index - 1]))
                        output["system_sign"].append(np.asarray([-1]))
                        batch_index += 1

        nbatch_ = len(output["natoms"])
        apply_random_rotations(output,nbatch_)
        add_flow_matching(output,nbatch_)

        # Stack and concatenate the arrays
        for k, v in output.items():
            try:
                if v[0].ndim == 0:
                    v = np.stack(v)
                else:
                    v = np.concatenate(v, axis=0)
                if np.issubdtype(v.dtype, np.floating):
                    v = v.astype(fprec)
            except Exception as e:
                print(f"Error stacking key '{k}': {e}")
                raise e
            output[k] = v

        if "cells" in output and pbc_training:
            try:
                output["reciprocal_cells"] = np.linalg.inv(output["cells"])
            except np.linalg.LinAlgError:
                raise ValueError(
                    "Encountered singular cell; cannot compute reciprocal_cells.")

        # Rename necessary keys
        # for key in rename_refs:
        #     if key in output:
        #         output["true_" + key] = output.pop(key)
        for kold, knew in rename_refs.items():
            assert (
                knew not in output
            ), f"Cannot rename key {kold} to {knew}. Key {knew} already present."
            if kold in output:
                output[knew] = output.pop(kold)

        output["flags"] = flags
        return output

    collate_layers_train = [collate_fn_]
    collate_layers_valid = [collate_fn_]

    ### collate preprocessing
    # add noise to the training data
    noise_sigma = training_config.noise_sigma
    if noise_sigma is not None:
        assert isinstance(noise_sigma, dict), "noise_sigma should be a dictionary"

        for sigma in noise_sigma.values():
            assert sigma >= 0, "Noise sigma should be a positive number"

        print("Adding noise to the training data:")
        for key, sigma in noise_sigma.items():
            print(f"  - {key} with sigma = {sigma}")

        assert np_rng is not None, "np_rng must be provided for adding noise."

        def collate_with_noise(batch):
            for key, sigma in noise_sigma.items():
                if key in batch and sigma > 0:
                    batch[key] += np_rng.normal(0, sigma, batch[key].shape).astype(
                        batch[key].dtype
                    )
            return batch

        collate_layers_train.append(collate_with_noise)

    if atom_padding:
        padder = AtomPadding(add_sys=training_config.padder_add_sys)
        padder_state = padder.init()

        def collate_with_padding(batch):
            padder_state_up, output = padder(padder_state, batch)
            padder_state.update(padder_state_up)
            return output

        collate_layers_train.append(collate_with_padding)
        collate_layers_valid.append(collate_with_padding)

    if split_data_inputs:

        # input_keys += additional_input_keys
        # input_keys = set(input_keys)
        print("Input keys:", all_inputs)
        print("Ref keys:", ref_keys)

        def collate_split(batch):
            inputs = {}
            refs = {}
            for k, v in batch.items():
                if k in all_inputs:
                    inputs[k] = v
                if k in ref_keys:
                    refs[k] = v
                if k.endswith("_mask") and k[:-5] in ref_keys:
                    refs[k] = v
            return inputs, refs

        collate_layers_train.append(collate_split)
        collate_layers_valid.append(collate_split)

    ### apply all collate preprocessing
    if len(collate_layers_train) == 1:
        collate_fn_train = collate_layers_train[0]
    else:

        def collate_fn_train(batch):
            for layer in collate_layers_train:
                batch = layer(batch)
            return batch

    if len(collate_layers_valid) == 1:
        collate_fn_valid = collate_layers_valid[0]
    else:

        def collate_fn_valid(batch):
            for layer in collate_layers_valid:
                batch = layer(batch)
            return batch

    is_url = dspath.lower().startswith(("http://", "https://"))
    if not is_url and not os.path.exists(dspath):
        raise ValueError(f"Dataset file '{dspath}' not found.")
    # dspath is passed explicitly by the caller.
    print(f"Loading dataset from {dspath}...", end="")
    # print(f"   the following keys will be renamed if present : {rename_refs}")
    sharded_training = False
    if is_url:
        if train_val_split:
            dataset = {
                "training": HTTPDataset(dspath, val=False, batch_size=batch_size),
                "validation": HTTPDataset(dspath, val=True, batch_size=batch_size_val if batch_size_val is not None else batch_size),
            }
        else:
            dataset = HTTPDataset(dspath, val=False, batch_size=batch_size)
    elif dspath.endswith(".db"):
        if train_val_split:
            dataset = {
                "training": DBDataset(dspath, table="training"),
                "validation": DBDataset(dspath, table="validation"),
            }
        else:
            dataset = DBDataset(dspath)
    elif dspath.endswith(".h5") or dspath.endswith(".hdf5"):
        if train_val_split:
            dataset = {
                "training": H5Dataset(dspath, table="training"),
                "validation": H5Dataset(dspath, table="validation"),
            }
        else:
            dataset = H5Dataset(dspath)
    elif dspath.endswith(".pkl") or dspath.endswith(".pickle"):
        with open(dspath, "rb") as f:
            dataset = pickle.load(f)
        if not train_val_split and isinstance(dataset, dict):
            dataset = dataset["training"]

    elif dspath.endswith(".parquet"):
        if train_val_split:
            train_idx, val_idx = build_split_indices(dspath)
            full_ds = ParquetDataset(
                dspath, columns=input_file_keys, mode=load_mode, chunk_size=load_chunk_size
            )
            train_ds = Subset(full_ds, train_idx)
            val_ds = Subset(full_ds, val_idx)
            dataset = {
                "training": train_ds,
                "validation": val_ds,
            }
        else:
            dataset = ParquetDataset(
                dspath, columns=input_file_keys, mode=load_mode, chunk_size=load_chunk_size
            )

    elif os.path.isdir(dspath):
        if train_val_split:
            dataset = {}
            val_pkl = os.path.join(dspath, "validation.pkl")
            val_parquet = os.path.join(dspath, "validation.parquet")
            if os.path.exists(val_pkl):
                with open(val_pkl, "rb") as f:
                    dataset["validation"] = pickle.load(f)
            elif os.path.exists(val_parquet):
                dataset["validation"] = ParquetDataset(
                    val_parquet, columns=input_file_keys, mode=load_mode, chunk_size=load_chunk_size
                )
            else:
                pass
        else:
            dataset = None
        pkl_shards = sorted(glob.glob(os.path.join(dspath, "training_*.pkl")))
        parquet_shards = sorted(
            glob.glob(os.path.join(dspath, "training_*.parquet")))
        if pkl_shards:
            shard_files = pkl_shards
            shard_kind = "Pickle"
        elif parquet_shards:
            shard_files = parquet_shards
            shard_kind = "Parquet"
        else:
            raise ValueError("No dataset shards found.")
        nshards = len(shard_files)
        if nshards == 1:
            shard = shard_files[0]
            if shard.endswith(".pkl"):
                with open(shard, "rb") as f:
                    if train_val_split:
                        dataset["training"] = pickle.load(f)
                    else:
                        dataset = pickle.load(f)
            else:
                if train_val_split:
                    if "validation" not in dataset:
                        tr_idx, va_idx = build_split_indices(shard)
                        full = ParquetDataset(shard, columns=input_file_keys,
                                              mode=load_mode,
                                              chunk_size=load_chunk_size)
                        dataset["training"] = Subset(full, tr_idx)
                        dataset["validation"] = Subset(full, va_idx)
                    else:
                        dataset["training"] = ParquetDataset(
                            shard, columns=input_file_keys, mode=load_mode,
                            chunk_size=load_chunk_size
                        )
                else:
                    dataset = ParquetDataset(
                        shard, columns=input_file_keys, mode=load_mode, chunk_size=load_chunk_size
                    )
        else:
            if shard_kind == "Parquet":
                preloaded_shards = {
                    path: ParquetDataset(
                        path,
                        columns=input_file_keys,
                        mode=load_mode,
                        chunk_size=load_chunk_size
                    ) for path in shard_files
                }
                # DO NOT DELETE.
                # I will come back when collate_fn_ is a separate pickable function
                # num_workers = 4
                # prefetch_factor = 2
                # persistent_workers = True
            print(f"Found {nshards} dataset shards ({shard_kind}).")
            sharded_training = True
    else:
        raise ValueError(
            "Unknown dataset format. Supported formats: '.db', '.h5', '.pkl', '.pickle', '.parquet'. "
        )
    print(" done.")

    ### Log dataset in MLflow.
    if log_dataset is not None:
        log_dataset(dspath, dataset)

    ### BUILD DATALOADERS
    # batch_size is passed explicitly by the caller.
    shuffle = training_config.shuffle_dataset
    num_workers = 0
    prefetch_factor = None
    if is_url: 
        shuffle = None
        num_workers = 1
        prefetch_factor = training_config.batch_prefetch
        print("Using num_workers=1 and prefetch_factor=",prefetch_factor," for HTTP dataset.")

    loader_class = _PrefetchingDataLoader if is_url else DataLoader
    if train_val_split:
        if batch_size_val is None:
            batch_size_val = batch_size
        dataloader_validation = loader_class(
            dataset["validation"],
            batch_size=batch_size_val,
            shuffle=shuffle,
            collate_fn=collate_fn_valid,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )

    if sharded_training:

        def _load_training_shard(path: str):
            if path.endswith(".pkl") or path.endswith(".pickle"):
                with open(path, "rb") as f:
                    return pickle.load(f)
            elif path.endswith(".parquet"):
                return preloaded_shards[path]
            else:
                raise ValueError(f"Unsupported shard type: {path}")

        def iterate_sharded_dataset():
            indices = np.arange(nshards)
            if shuffle:
                assert np_rng is not None, "np_rng must be provided for shuffling."
                np_rng.shuffle(indices)
            for i in indices:
                filename = shard_files[i]
                print(f"# Loading dataset shard from {filename}...", end="")
                dataset_shard = _load_training_shard(filename)
                print(" done.")
                dataloader = DataLoader(
                    dataset_shard,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    collate_fn=collate_fn_train,
                    num_workers=num_workers,
                    prefetch_factor=prefetch_factor,
                    persistent_workers=persistent_workers,
                )
                for batch in dataloader:
                    yield batch

        class DataLoaderSharded:
            def __iter__(self):
                return iterate_sharded_dataset()

        dataloader_training = DataLoaderSharded()
    else:
        dataloader_training = loader_class(
            dataset["training"] if train_val_split else dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn_train,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )

    if not infinite_iterator:
        if train_val_split:
            return dataloader_training, dataloader_validation
        return dataloader_training

    def next_batch_factory(dataloader):
        while True:
            for batch in dataloader:
                yield batch

    training_iterator = next_batch_factory(dataloader_training)
    if train_val_split:
        validation_iterator = next_batch_factory(dataloader_validation)
        return training_iterator, validation_iterator
    return training_iterator


def load_model(
    config_model: dict[str, Any] | None = None,
    model_file: str | None = None,
    rng_key: str | None = None,
    print_model=False,
) -> FENNIX:
    """
    Load a FENNIX model from a file or create a new one.

    Args:
        parameters (dict): A dictionary of parameters for the model.
        model_file (str, optional): The path to a saved model file to load.

    Raise:

        - FileNotFoundError: model_file given does not exist.
        - ValueError: model_file is not given and no rng_key is provided
            to build the model.
        - ValueError: no model_file and no model configuration provided.

    Returns:
        FENNIX: A FENNIX model object.
    """
    # Load model from given file
    if model_file is not None:
        if os.path.exists(model_file):
            print(f"Restored model from '{model_file}'.")
            model = FENNIX.load(model_file, use_atom_padding=False)
        else:
            raise FileNotFoundError(f"Model file {model_file} not found.")

        return model

    # create model from config
    elif config_model is not None:
        if rng_key is None:
            raise ValueError(
                "rng_key must be specified if model_file is not provided.")
        model = FENNIX(**config_model, rng_key=rng_key, use_atom_padding=False)

    else:
        raise ValueError(
            "Configuration Error: No model file path and "
            "no model definition found."
        )

    if print_model:
        print(model.summarize())

    return model


def copy_parameters(variables, variables_ref, params=[".*"]):
    def merge_params(full_path_, v, v_ref):
        full_path = "/".join(full_path_[1:]).lower()
        # status = (False, "")
        for path in params:
            if re.match(path.lower(), full_path):
            # if full_path.startswith(path.lower()) and len(path) > len(status[1]):
                return v_ref
        return v
        # return v_ref if status[0] else v

    flat = traverse_util.flatten_dict(variables, keep_empty_nodes=False)
    flat_ref = traverse_util.flatten_dict(variables_ref, keep_empty_nodes=False)
    return traverse_util.unflatten_dict(
        {
            k: merge_params(k, v, flat_ref[k]) if k in flat_ref else v
            for k, v in flat.items()
        }
    )


class TeeLogger(object):
    def __init__(self, name):
        self.file = io.TextIOWrapper(open(name, "wb"), write_through=True)
        self.stdout = None

    def __del__(self):
        self.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        self.flush()

    def close(self):
        self.file.close()

    def flush(self):
        self.file.flush()

    def bind_stdout(self):
        if isinstance(sys.stdout, TeeLogger):
            raise ValueError("stdout already bound to a Tee instance.")
        if self.stdout is not None:
            raise ValueError("stdout already bound.")
        self.stdout = sys.stdout
        sys.stdout = self

    def unbind_stdout(self):
        if self.stdout is None:
            raise ValueError("stdout is not bound.")
        sys.stdout = self.stdout
