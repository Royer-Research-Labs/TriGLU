from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from triglu import ModelConfig
from triglu.runtime import load_experiment_config


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ("12a0t", "9a3t", "6a6t", "3a9t")
SCALED_20L_4K = (
    "20a0t",
    "15a5t",
    "10a10t",
    "9a11t_front_blend",
    "5a15t",
    "15a5t_front_blend",
    "15a5t_late_alternating",
    "15a5t_tail_block",
    "15a5t_final_attention",
)


def scientific_settings(config: dict) -> dict:
    comparable = deepcopy(config)
    comparable["model"].pop("layer_types")
    comparable["training"].pop("output_dir")
    return comparable


def test_canonical_configs_differ_only_in_layer_plan_and_output_path() -> None:
    configs = [
        load_experiment_config(ROOT / "configs" / "12l_1k_100m" / f"{name}.yaml")
        for name in CANONICAL
    ]
    baseline = scientific_settings(configs[0])
    assert all(scientific_settings(config) == baseline for config in configs[1:])

    expected_counts = [(12, 0), (9, 3), (6, 6), (3, 9)]
    for config, (attention, triglu) in zip(configs, expected_counts, strict=True):
        plan = config["model"]["layer_types"]
        assert plan.count("attention") == attention
        assert plan.count("triglu") == triglu
        ModelConfig.from_dict(config["model"])

    triglu_sets = [
        {
            index
            for index, layer_type in enumerate(config["model"]["layer_types"])
            if layer_type == "triglu"
        }
        for config in configs
    ]
    assert triglu_sets[0] < triglu_sets[1] < triglu_sets[2] < triglu_sets[3]


def test_canonical_token_budget_is_fixed() -> None:
    config = load_experiment_config(
        ROOT / "configs" / "12l_1k_100m" / "12a0t.yaml"
    )
    training = config["training"]
    tokens_per_step = (
        training["batch_size"]
        * training["sequence_length"]
        * training["gradient_accumulation_steps"]
    )
    assert tokens_per_step == 65_536
    assert tokens_per_step * training["max_steps"] == 100_007_936
    assert training["checkpoint_interval"] == 0
    assert training["output_dir"] == "runs/12l_1k_100m/12a0t"


def test_scaled_20l_4k_configs_are_controlled_and_valid() -> None:
    configs = [
        load_experiment_config(ROOT / "configs" / "20l_4k_1b" / f"{name}.yaml")
        for name in SCALED_20L_4K
    ]
    baseline = scientific_settings(configs[0])
    assert all(scientific_settings(config) == baseline for config in configs[1:])

    expected_counts = [
        (20, 0),
        (15, 5),
        (10, 10),
        (9, 11),
        (5, 15),
        (15, 5),
        (15, 5),
        (15, 5),
        (15, 5),
    ]
    for config, (attention, triglu) in zip(configs, expected_counts, strict=True):
        model = config["model"]
        training = config["training"]
        assert model["n_layers"] == 20
        assert model["d_model"] == 512
        assert model["context_length"] == 4096
        assert model["layer_types"].count("attention") == attention
        assert model["layer_types"].count("triglu") == triglu
        assert training["sequence_length"] == 4096
        assert training["gradient_accumulation_steps"] == 1
        ModelConfig.from_dict(model)

    front_blend_9a11t = configs[SCALED_20L_4K.index("9a11t_front_blend")]
    triglu_indices = {
        index
        for index, layer_type in enumerate(front_blend_9a11t["model"]["layer_types"])
        if layer_type == "triglu"
    }
    assert triglu_indices == {6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19}

    final_attention = configs[SCALED_20L_4K.index("15a5t_final_attention")]
    final_plan = final_attention["model"]["layer_types"]
    assert final_plan[-1] == "attention"
    assert {
        index for index, layer_type in enumerate(final_plan) if layer_type == "triglu"
    } == {8, 12, 15, 17, 18}


def test_scaled_20l_4k_token_budget_is_one_billion() -> None:
    config = load_experiment_config(
        ROOT / "configs" / "20l_4k_1b" / "20a0t.yaml"
    )
    training = config["training"]
    tokens_per_step = (
        training["batch_size"]
        * training["sequence_length"]
        * training["gradient_accumulation_steps"]
    )
    assert tokens_per_step == 65_536
    assert tokens_per_step * training["max_steps"] == 1_000_013_824
    assert training["checkpoint_interval"] == 0
    assert training["output_dir"] == "runs/20l_4k_1b/20a0t"


def test_smoke_config_is_offline_small_and_mixed() -> None:
    config = load_experiment_config(ROOT / "configs" / "smoke.yaml")
    assert config["data"]["synthetic"] is True
    assert config["training"]["device"] == "cpu"
    assert config["training"]["compile"] is False
    assert config["training"]["max_steps"] == 3
    assert config["model"]["layer_types"] == ["attention", "triglu"]
    ModelConfig.from_dict(config["model"])
