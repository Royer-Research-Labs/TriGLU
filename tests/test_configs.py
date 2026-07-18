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
PRIOR_ART_ABLATIONS = (
    ("15a5_triglu_no_rope_front_blend", "triglu_no_rope", None),
    ("15a5_mb_mlp_front_blend", "mb_mlp", None),
    ("15a5_swiglu_front_blend", "swiglu_mixer", 683),
)
PLACEMENT_AMOUNT_LADDER = (
    ("18a2_triglu_no_rope_nested", frozenset({15, 19})),
    ("15a5_triglu_no_rope_nested", frozenset({7, 11, 15, 17, 19})),
    (
        "12a8_triglu_no_rope_nested",
        frozenset({6, 7, 10, 11, 14, 15, 17, 19}),
    ),
    (
        "9a11_triglu_no_rope_nested",
        frozenset({6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19}),
    ),
    ("6a14_triglu_no_rope_nested", frozenset(range(6, 20))),
)
PLACEMENT_CONTROLS = (
    ("15a5_triglu_no_rope_nested", frozenset({7, 11, 15, 17, 19})),
    ("15a5_triglu_no_rope_repeating", frozenset({3, 7, 11, 15, 19})),
    ("15a5_triglu_no_rope_tail_block", frozenset(range(15, 20))),
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


def test_prior_art_ablation_configs_match_the_selected_front_blend_plan() -> None:
    reference = load_experiment_config(
        ROOT / "configs" / "20l_4k_1b" / "15a5t_front_blend.yaml"
    )
    replacement_indices = {8, 12, 15, 17, 19}

    for name, replacement_type, hidden_size in PRIOR_ART_ABLATIONS:
        config = load_experiment_config(
            ROOT / "configs" / "20l_4k_1b" / "ablations" / f"{name}.yaml"
        )
        model = config["model"]
        training = config["training"]

        comparable = deepcopy(config)
        comparable["model"]["layer_types"] = list(reference["model"]["layer_types"])
        comparable["model"]["swiglu_mixer_hidden_size"] = reference["model"][
            "swiglu_mixer_hidden_size"
        ]
        comparable["training"]["output_dir"] = reference["training"]["output_dir"]
        assert comparable == reference

        assert model["layer_types"].count("attention") == 15
        assert model["layer_types"].count(replacement_type) == 5
        assert {
            index
            for index, layer_type in enumerate(model["layer_types"])
            if layer_type == replacement_type
        } == replacement_indices
        assert model["swiglu_mixer_hidden_size"] == hidden_size
        assert training["batch_size"] * training["sequence_length"] == 65_536
        assert training["gradient_accumulation_steps"] == 1
        assert training["max_steps"] == 15_259
        assert training["output_dir"] == f"runs/20l_4k_1b/{name}"
        ModelConfig.from_dict(model)


def test_no_rope_placement_amount_configs_are_controlled_and_nested() -> None:
    reference = load_experiment_config(
        ROOT
        / "configs"
        / "20l_4k_1b"
        / "ablations"
        / "15a5_triglu_no_rope_front_blend.yaml"
    )
    config_dir = ROOT / "configs" / "20l_4k_1b" / "placement_amount"
    ladder_sets: list[frozenset[int]] = []

    for name, expected_replacements in PLACEMENT_AMOUNT_LADDER:
        config = load_experiment_config(config_dir / f"{name}.yaml")
        model = config["model"]
        training = config["training"]
        actual_replacements = frozenset(
            index
            for index, layer_type in enumerate(model["layer_types"])
            if layer_type == "triglu_no_rope"
        )

        comparable = deepcopy(config)
        comparable["model"]["layer_types"] = list(reference["model"]["layer_types"])
        comparable["training"]["output_dir"] = reference["training"]["output_dir"]
        assert comparable == reference

        assert actual_replacements == expected_replacements
        assert model["layer_types"].count("attention") == 20 - len(
            expected_replacements
        )
        assert set(model["layer_types"]) <= {"attention", "triglu_no_rope"}
        assert training["output_dir"] == f"runs/20l_4k_1b/{name}"
        assert training["seed"] == 1337
        assert training["batch_size"] * training["sequence_length"] == 65_536
        assert training["gradient_accumulation_steps"] == 1
        assert training["max_steps"] == 15_259
        ModelConfig.from_dict(model)
        ladder_sets.append(actual_replacements)

    assert all(
        smaller < larger
        for smaller, larger in zip(
            ladder_sets[:-1], ladder_sets[1:], strict=True
        )
    )
    assert ladder_sets[1] - ladder_sets[0] == {7, 11, 17}
    assert ladder_sets[2] - ladder_sets[1] == {6, 10, 14}
    assert ladder_sets[3] - ladder_sets[2] == {9, 13, 18}
    assert ladder_sets[4] - ladder_sets[3] == {8, 12, 16}


def test_no_rope_fixed_count_placement_controls_are_exact() -> None:
    config_dir = ROOT / "configs" / "20l_4k_1b" / "placement_amount"
    placements = {
        "front_blend": frozenset({8, 12, 15, 17, 19}),
    }

    for name, expected_replacements in PLACEMENT_CONTROLS:
        config = load_experiment_config(config_dir / f"{name}.yaml")
        replacements = frozenset(
            index
            for index, layer_type in enumerate(config["model"]["layer_types"])
            if layer_type == "triglu_no_rope"
        )
        assert replacements == expected_replacements
        assert len(replacements) == 5
        placements[name] = replacements

    assert len(set(placements.values())) == len(placements)


def test_smoke_config_is_offline_small_and_mixed() -> None:
    config = load_experiment_config(ROOT / "configs" / "smoke.yaml")
    assert config["data"]["synthetic"] is True
    assert config["training"]["device"] == "cpu"
    assert config["training"]["compile"] is False
    assert config["training"]["max_steps"] == 3
    assert config["model"]["layer_types"] == ["attention", "triglu"]
    ModelConfig.from_dict(config["model"])


def test_ablation_smoke_config_exercises_every_new_mixer_offline() -> None:
    config = load_experiment_config(ROOT / "configs" / "smoke_ablations.yaml")
    assert config["data"]["synthetic"] is True
    assert config["training"]["device"] == "cpu"
    assert config["training"]["compile"] is False
    assert config["training"]["max_steps"] == 2
    assert config["model"]["layer_types"] == [
        "triglu_no_rope",
        "mb_mlp",
        "swiglu_mixer",
        "attention",
    ]
    assert config["model"]["swiglu_mixer_hidden_size"] == 85
    ModelConfig.from_dict(config["model"])
