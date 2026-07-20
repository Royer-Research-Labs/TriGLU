from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from triglu import DecoderLM, ModelConfig
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
    (
        "15a5_triglu_no_rope_early_intrusion",
        frozenset({3, 12, 15, 17, 19}),
    ),
)
SWIGLU_PLACEMENT_PROBES = (
    ("15a5_swiglu_repeating", frozenset({3, 7, 11, 15, 19})),
    (
        "9a11_swiglu_nested",
        frozenset({6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19}),
    ),
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


def test_triglu_ffn_control_matches_canonical_all_attention_budget() -> None:
    reference = load_experiment_config(
        ROOT / "configs" / "20l_4k_1b" / "20a0t.yaml"
    )
    control = load_experiment_config(
        ROOT
        / "configs"
        / "20l_4k_1b"
        / "ablations"
        / "20a0t_triglu_no_rope_ffn.yaml"
    )
    model = control["model"]
    training = control["training"]

    comparable = deepcopy(control)
    comparable["model"]["ffn_hidden_size"] = reference["model"]["ffn_hidden_size"]
    comparable["model"]["ffn_type"] = reference["model"]["ffn_type"]
    comparable["training"]["output_dir"] = reference["training"]["output_dir"]
    assert comparable == reference

    assert model["ffn_type"] == "triglu_no_rope"
    assert model["ffn_hidden_size"] == 1032
    assert reference["model"]["ffn_type"] == "swiglu"
    assert reference["model"]["ffn_hidden_size"] == 1376
    assert model["layer_types"] == ["attention"] * 20
    assert model["bias"] is False
    assert (
        4 * model["d_model"] * model["ffn_hidden_size"]
        == 3
        * reference["model"]["d_model"]
        * reference["model"]["ffn_hidden_size"]
        == 2_113_536
    )
    assert training["output_dir"] == (
        "runs/20l_4k_1b/20a0t_triglu_no_rope_ffn"
    )
    ModelConfig.from_dict(model)


def test_single_residual_and_grouped_depth_configs_are_explicit_controls() -> None:
    ablation_dir = ROOT / "configs" / "20l_4k_1b" / "ablations"
    single = load_experiment_config(
        ablation_dir
        / "15a5_wide_swiglu_single_residual_front_blend.yaml"
    )
    grouped = load_experiment_config(
        ablation_dir
        / "9l_9a0t_grouped_swiglu_9a11_nested_collapse.yaml"
    )
    front_reference = load_experiment_config(
        ROOT / "configs" / "20l_4k_1b" / "15a5t_front_blend.yaml"
    )
    nested_reference = load_experiment_config(
        ROOT
        / "configs"
        / "20l_4k_1b"
        / "placement_amount"
        / "9a11_triglu_no_rope_nested.yaml"
    )

    for control, reference in (
        (single, front_reference),
        (grouped, nested_reference),
    ):
        assert control["data"] == reference["data"]
        training = deepcopy(control["training"])
        training["output_dir"] = reference["training"]["output_dir"]
        assert training == reference["training"]
        for key in (
            "vocab_size",
            "d_model",
            "n_heads",
            "ffn_hidden_size",
            "ffn_type",
            "context_length",
            "dropout",
            "bias",
            "rope_theta",
            "norm_eps",
            "init_std",
            "tie_embeddings",
        ):
            assert control["model"][key] == reference["model"][key]
        ModelConfig.from_dict(control["model"])

    single_model = single["model"]
    single_positions = {
        index
        for index, layer_type in enumerate(single_model["layer_types"])
        if layer_type == "ffn_only"
    }
    assert single_model["n_layers"] == 20
    assert single_model["layer_types"].count("attention") == 15
    assert single_positions == {8, 12, 15, 17, 19}
    assert single_model["ffn_hidden_sizes"] == [
        2059 if index in single_positions else 1376
        for index in range(20)
    ]
    assert single_model["residual_init_depth"] == 20
    C = single_model["d_model"]
    assert C + 3 * C * 2059 == 2 * C + 4 * C * C + 3 * C * 1376
    assert single["training"]["output_dir"].endswith(
        "15a5_wide_swiglu_single_residual_front_blend"
    )

    grouped_model = grouped["model"]
    widths = [1376, 1376, 1376, 1376, 1376, 5495, 7554, 7554, 7554]
    assert grouped_model["n_layers"] == 9
    assert grouped_model["layer_types"] == ["attention"] * 9
    assert grouped_model["ffn_hidden_sizes"] == widths
    assert grouped_model["residual_init_depth"] == 20
    original_block_parameters = 4 * C * C + 3 * C * 1376 + 2 * C
    grouped_parameters = sum(
        4 * C * C + 3 * C * width + 2 * C for width in widths
    )
    assert grouped_parameters == 20 * original_block_parameters + 512
    assert grouped["training"]["output_dir"].endswith(
        "9l_9a0t_grouped_swiglu_9a11_nested_collapse"
    )


def test_nine_layer_ffn_controls_are_explicit_and_parameter_checked() -> None:
    ablation_dir = ROOT / "configs" / "20l_4k_1b" / "ablations"
    grouped_swiglu = load_experiment_config(
        ablation_dir
        / "9l_9a0t_grouped_swiglu_9a11_nested_collapse.yaml"
    )
    grouped_triglu = load_experiment_config(
        ablation_dir / "9l_9a0t_grouped_triglu_no_rope_ffn.yaml"
    )
    standard_nine = load_experiment_config(
        ablation_dir / "9l_9a0t_standard_swiglu.yaml"
    )
    baseline = load_experiment_config(
        ROOT / "configs" / "20l_4k_1b" / "20a0t.yaml"
    )

    for control in (grouped_triglu, standard_nine):
        assert control["data"] == baseline["data"]
        training = deepcopy(control["training"])
        training["output_dir"] = baseline["training"]["output_dir"]
        assert training == baseline["training"]
        assert control["model"]["n_layers"] == 9
        assert control["model"]["layer_types"] == ["attention"] * 9

    triglu_model = grouped_triglu["model"]
    swiglu_model = grouped_swiglu["model"]
    standard_model = standard_nine["model"]
    triglu_widths = [1032, 1032, 1032, 1032, 1032, 4121, 5666, 5666, 5665]
    swiglu_widths = [1376, 1376, 1376, 1376, 1376, 5495, 7554, 7554, 7554]

    assert triglu_model["ffn_type"] == "triglu_no_rope"
    assert triglu_model["ffn_hidden_size"] == 1032
    assert triglu_model["ffn_hidden_sizes"] == triglu_widths
    assert triglu_model["residual_init_depth"] == 20
    assert swiglu_model["ffn_type"] == "swiglu"
    assert swiglu_model["ffn_hidden_sizes"] == swiglu_widths
    assert swiglu_model["residual_init_depth"] == 20

    # TriGLU FFNs have four C-by-H projection matrices; SwiGLU FFNs have
    # three. Integer hidden widths make a 512-parameter remainder unavoidable.
    C = triglu_model["d_model"]
    assert 4 * C * sum(triglu_widths) == 3 * C * sum(swiglu_widths) + 512

    assert standard_model["ffn_type"] == "swiglu"
    assert standard_model["ffn_hidden_size"] == 1376
    assert standard_model["ffn_hidden_sizes"] is None
    assert standard_model["residual_init_depth"] == 9

    grouped_triglu_config = ModelConfig.from_dict(triglu_model)
    grouped_swiglu_config = ModelConfig.from_dict(swiglu_model)
    standard_config = ModelConfig.from_dict(standard_model)
    assert DecoderLM(grouped_triglu_config).num_parameters() == 89_019_904
    assert DecoderLM(grouped_swiglu_config).num_parameters() == 89_019_392
    assert DecoderLM(standard_config).num_parameters() == 54_224_384
    assert grouped_triglu_config.effective_residual_init_depth == 20
    assert standard_config.effective_residual_init_depth == 9

    assert grouped_triglu["training"]["output_dir"].endswith(
        "9l_9a0t_grouped_triglu_no_rope_ffn"
    )
    assert standard_nine["training"]["output_dir"].endswith(
        "9l_9a0t_standard_swiglu"
    )


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


def test_early_intrusion_is_a_single_swap_from_front_blend() -> None:
    reference = load_experiment_config(
        ROOT
        / "configs"
        / "20l_4k_1b"
        / "ablations"
        / "15a5_triglu_no_rope_front_blend.yaml"
    )
    probe = load_experiment_config(
        ROOT
        / "configs"
        / "20l_4k_1b"
        / "placement_amount"
        / "15a5_triglu_no_rope_early_intrusion.yaml"
    )
    reference_replacements = frozenset(
        index
        for index, layer_type in enumerate(reference["model"]["layer_types"])
        if layer_type == "triglu_no_rope"
    )
    probe_replacements = frozenset(
        index
        for index, layer_type in enumerate(probe["model"]["layer_types"])
        if layer_type == "triglu_no_rope"
    )

    comparable = deepcopy(probe)
    comparable["model"]["layer_types"] = list(reference["model"]["layer_types"])
    comparable["training"]["output_dir"] = reference["training"]["output_dir"]
    assert comparable == reference
    assert reference_replacements == {8, 12, 15, 17, 19}
    assert probe_replacements == {3, 12, 15, 17, 19}
    assert reference_replacements ^ probe_replacements == {3, 8}


def test_swiglu_placement_probe_configs_match_the_parameter_control() -> None:
    reference = load_experiment_config(
        ROOT
        / "configs"
        / "20l_4k_1b"
        / "ablations"
        / "15a5_swiglu_front_blend.yaml"
    )
    config_dir = ROOT / "configs" / "20l_4k_1b" / "placement_amount"

    for name, expected_replacements in SWIGLU_PLACEMENT_PROBES:
        config = load_experiment_config(config_dir / f"{name}.yaml")
        model = config["model"]
        training = config["training"]
        replacements = frozenset(
            index
            for index, layer_type in enumerate(model["layer_types"])
            if layer_type == "swiglu_mixer"
        )

        comparable = deepcopy(config)
        comparable["model"]["layer_types"] = list(reference["model"]["layer_types"])
        comparable["training"]["output_dir"] = reference["training"]["output_dir"]
        assert comparable == reference

        assert replacements == expected_replacements
        assert model["layer_types"].count("attention") == 20 - len(replacements)
        assert set(model["layer_types"]) == {"attention", "swiglu_mixer"}
        assert model["swiglu_mixer_hidden_size"] == 683
        assert training["output_dir"] == f"runs/20l_4k_1b/{name}"
        assert training["seed"] == 1337
        assert config["data"]["seed"] == 1337
        assert (
            training["batch_size"]
            * training["sequence_length"]
            * training["gradient_accumulation_steps"]
            == 65_536
        )
        assert training["max_steps"] == 15_259
        assert training["checkpoint_interval"] == 0
        ModelConfig.from_dict(model)

        no_rope = load_experiment_config(
            config_dir
            / (
                "15a5_triglu_no_rope_repeating.yaml"
                if name == "15a5_swiglu_repeating"
                else "9a11_triglu_no_rope_nested.yaml"
            )
        )
        no_rope_replacements = frozenset(
            index
            for index, layer_type in enumerate(no_rope["model"]["layer_types"])
            if layer_type == "triglu_no_rope"
        )
        assert no_rope_replacements == replacements


def test_every_placement_amount_config_has_explicit_coverage() -> None:
    config_dir = ROOT / "configs" / "20l_4k_1b" / "placement_amount"
    expected = {
        name
        for name, _replacements in (
            *PLACEMENT_AMOUNT_LADDER,
            *PLACEMENT_CONTROLS,
            *SWIGLU_PLACEMENT_PROBES,
        )
    }
    actual = {path.stem for path in config_dir.glob("*.yaml")}
    assert actual == expected


def test_combination_config_matches_flagship_with_matched_triglu_ffn() -> None:
    reference = load_experiment_config(
        ROOT / "configs" / "20l_4k_1b" / "placement_amount"
        / "9a11_triglu_no_rope_nested.yaml"
    )
    config = load_experiment_config(
        ROOT / "configs" / "20l_4k_1b" / "ablations"
        / "9a11_triglu_no_rope_nested_triglu_ffn.yaml"
    )
    # Identical to the flagship except the FFN form and its matched width.
    comparable = deepcopy(config)
    comparable["model"]["ffn_type"] = reference["model"].get("ffn_type", "swiglu")
    comparable["model"]["ffn_hidden_size"] = reference["model"]["ffn_hidden_size"]
    comparable["training"]["output_dir"] = reference["training"]["output_dir"]
    assert comparable == reference

    assert config["model"]["ffn_type"] == "triglu_no_rope"
    assert config["model"]["ffn_hidden_size"] == 1032
    # 4*C*1032 == 3*C*1376, so the model stays exactly parameter-matched.
    combined = DecoderLM(ModelConfig.from_dict(config["model"])).num_parameters()
    flagship = DecoderLM(ModelConfig.from_dict(reference["model"])).num_parameters()
    assert combined == flagship == 89_018_880


PARALLEL_BLOCK_CONTROLS = (
    ("20a0t_parallel_block", "20a0t.yaml"),
    (
        "9a11_triglu_no_rope_nested_parallel_block",
        "placement_amount/9a11_triglu_no_rope_nested.yaml",
    ),
)


def test_parallel_block_configs_change_only_block_mode() -> None:
    for name, reference_rel in PARALLEL_BLOCK_CONTROLS:
        reference = load_experiment_config(
            ROOT / "configs" / "20l_4k_1b" / reference_rel
        )
        config = load_experiment_config(
            ROOT / "configs" / "20l_4k_1b" / "ablations" / f"{name}.yaml"
        )
        comparable = deepcopy(config)
        comparable["model"]["block_mode"] = reference["model"]["block_mode"]
        comparable["training"]["output_dir"] = reference["training"]["output_dir"]
        assert comparable == reference

        assert config["model"]["block_mode"] == "parallel"
        assert reference["model"]["block_mode"] == "sequential"
        assert config["training"]["output_dir"] == f"runs/20l_4k_1b/{name}"
        ModelConfig.from_dict(config["model"])


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
    assert config["model"]["ffn_type"] == "triglu_no_rope"
    assert config["model"]["ffn_hidden_size"] == 132
    assert config["model"]["bias"] is False
    assert 4 * 64 * 132 == 3 * 64 * 176 == 33_792
    ModelConfig.from_dict(config["model"])


def test_residual_control_smoke_exercises_both_new_topologies_offline() -> None:
    config = load_experiment_config(
        ROOT / "configs" / "smoke_residual_controls.yaml"
    )
    assert config["data"]["synthetic"] is True
    assert config["training"]["device"] == "cpu"
    assert config["training"]["compile"] is False
    assert config["training"]["max_steps"] == 2
    assert config["model"]["layer_types"] == [
        "attention",
        "ffn_only",
        "attention",
    ]
    assert config["model"]["ffn_hidden_sizes"] == [176, 261, 400]
    assert config["model"]["residual_init_depth"] == 4
    ModelConfig.from_dict(config["model"])
