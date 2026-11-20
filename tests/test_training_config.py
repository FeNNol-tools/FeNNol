import ast
import inspect
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

pytest.importorskip("torch", reason="training requires PyTorch")

from fennol.training.config import RunConfig, TrainingConfig


@pytest.fixture
def config_dict_basic() -> dict[str, Any]:
    config_dict = {"rng_seed": 42, "training": {"max_epochs": 8}}

    return config_dict


@pytest.fixture
def config_dict_multi_stages() -> dict[str, Any]:
    config_dict = {
        "training": {"batch_size": 256},
        "stages": {
            "stage_1": {"batch_size": 128},
            "stage_2": {"batch_size": 64},
        },
    }

    return config_dict

@pytest.fixture
def config_dict_model_file() -> dict[str, Any]:
    config_dict = {
        "training": {"batch_size": 256},
        "model": "my_config_model_file"
    }

    return config_dict

def test_config_file_take_precedence_over_default_values(
    tmp_path, config_dict_basic, monkeypatch
):
    """Test configuration sources priority."""
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(
            config_dict_basic,
            f,
            sort_keys=False,  # essential not to mess steps order
        )
    config = RunConfig.from_cli_args(str(config_path))

    assert config.training.max_restore_count == 5  # default value
    assert config.training.batch_size == 16  # default value
    assert config.training.max_epochs == 8
    assert config.rng_seed == 42


def test_config_normalize_device(monkeypatch):
    """Test device environment variable parsing."""
    monkeypatch.setenv("FENNOL_DEVICE", "GPU:42")
    config = RunConfig()
    assert config.device == "gpu:42"

    monkeypatch.setenv("FENNOL_DEVICE", "GPI:42")
    expected_error_pattern = (
        r".*DDevice must be one of \('cpu', 'gpu',"
        r"'cuda', 'gpu|cuda:<CUDA_VISIBLE_DEVICES>'\), got 'gpi:42'"
    )

    with pytest.raises(ValueError, match=expected_error_pattern):
        config = RunConfig()


def test_config_prioritize_cli_model_file_over_config_file_one(
    tmp_path, config_dict_model_file
):
    """Test todo."""
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(
            config_dict_model_file,
            f,
            sort_keys=False,
        )
    cli_model_file = "my_cli_model_file"
    config = RunConfig.from_cli_args(str(config_path), cli_model_file)

    assert config.model_file == cli_model_file


def test_config_detect_curriculum_learning(
    tmp_path, config_dict_multi_stages, monkeypatch
):
    #TODO finish
    """Test todo."""
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(
            config_dict_multi_stages,
            f,
            sort_keys=False,  # essential not to mess steps order
        )
    monkeypatch.setenv("FENNOL_CONFIG_FILE", str(config_path))
    config = RunConfig()

    assert True


def test_training_config_registers_supported_runtime_options():
    config = TrainingConfig(
        weight_decay=0.02,
        stochastic_scheduler=True,
        patience=7,
        noise_sigma={"forces": 0.1},
    )

    assert config.weight_decay == 0.02
    assert config.stochastic_scheduler is True
    assert config.patience == 7
    assert config.noise_sigma == {"forces": 0.1}


def test_training_config_rejects_unregistered_options():
    with pytest.raises(ValidationError, match="unrecognized_training_option"):
        TrainingConfig(unrecognized_training_option=True)


@pytest.mark.parametrize("model_key", ["model", "model_file"])
def test_yaml_model_file_is_used_when_cli_override_is_omitted(tmp_path, model_key):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({model_key: "from_yaml.fnx"}))
    config = RunConfig.from_cli_args(str(config_path))
    assert config.model_file == "from_yaml.fnx"


def test_yaml_model_file_is_used_on_restart_when_cli_override_is_omitted(tmp_path):
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"model_file": "from_yaml.fnx"})
    )
    config = RunConfig.from_cli_args(str(tmp_path))
    assert config.is_restart
    assert config.model_file == "from_yaml.fnx"


def test_run_config_preserves_http_dataset_url(tmp_path):
    address = "https://data.example/Training/SetA"
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump({"training": {"dspath": address}}))

    config = RunConfig.from_cli_args(str(config_file))

    assert config.training.dspath == address


def test_cli_model_file_overrides_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"model_file": "from_yaml.fnx"}))
    config = RunConfig.from_cli_args(str(config_path), "from_cli.fnx")
    assert config.model_file == "from_cli.fnx"


def test_training_config_declares_each_field_once():
    pytest.importorskip("torch", reason="training requires PyTorch")
    from fennol.training import config

    module = ast.parse(inspect.getsource(config))
    training_config = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "TrainingConfig"
    )
    fields = [
        node.target.id
        for node in training_config.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    ]

    assert "load_chunk_size" in fields
    assert len(fields) == len(set(fields))


def test_training_readme_names_the_configured_output_directory():
    with open("examples/training/ani.yaml") as config_file:
        output_directory = yaml.safe_load(config_file)["output_directory"]
    with open("examples/training/README.md") as readme_file:
        readme = readme_file.read()

    assert f"`{output_directory}`" in readme
