from __future__ import annotations

import struct
from types import SimpleNamespace

import pytest
import torch

from triglu.data import SyntheticTokenStream, UInt16TokenStream
from triglu.prepare_data import DataPreparationConfig
from triglu import prepare_data
from triglu.runtime import ConfigurationError, record_data_provenance, verify_token_manifest


def test_synthetic_stream_is_deterministic_and_has_consistent_successors() -> None:
    left = SyntheticTokenStream(128, 32, seed=4, pattern_length=16)
    right = SyntheticTokenStream(128, 32, seed=4, pattern_length=16)
    torch.testing.assert_close(left.read_tokens(0, 128), right.read_tokens(0, 128))

    tokens = left.read_tokens(0, 64)
    successor_by_token: dict[int, int] = {}
    for current, following in zip(tokens[:-1].tolist(), tokens[1:].tolist(), strict=True):
        successor_by_token.setdefault(current, following)
        assert successor_by_token[current] == following


def test_random_batches_are_reproducible_from_generator_state() -> None:
    stream = SyntheticTokenStream(256, 64, seed=5, pattern_length=32)
    generator = torch.Generator().manual_seed(99)
    state = generator.get_state()
    first = stream.sample_batch(4, 8, generator)
    generator.set_state(state)
    repeated = stream.sample_batch(4, 8, generator)
    torch.testing.assert_close(first[0], repeated[0])
    torch.testing.assert_close(first[1], repeated[1])
    torch.testing.assert_close(first[0][:, 1:], first[1][:, :-1])


def test_eval_batches_are_sequential_without_duplicate_targets() -> None:
    stream = SyntheticTokenStream(25, 25, seed=6, pattern_length=25)
    batches = list(stream.iter_eval_batches(2, 4))
    all_inputs = torch.cat([inputs for inputs, _ in batches], dim=0)
    all_targets = torch.cat([targets for _, targets in batches], dim=0)
    torch.testing.assert_close(all_inputs.flatten(), stream.read_tokens(0, 24))
    torch.testing.assert_close(all_targets.flatten(), stream.read_tokens(1, 24))


def test_uint16_stream_reads_little_endian_and_checks_bounds(tmp_path) -> None:
    path = tmp_path / "tokens.bin"
    values = [0, 1, 255, 256, 50_256]
    path.write_bytes(struct.pack(f"<{len(values)}H", *values))
    stream = UInt16TokenStream(path, vocab_size=50_304)
    assert stream.num_tokens == len(values)
    assert stream.read_tokens(1, 3).tolist() == values[1:4]

    with pytest.raises(ValueError, match="outside vocabulary"):
        UInt16TokenStream(path, vocab_size=1_000)


def test_uint16_stream_rejects_malformed_files(tmp_path) -> None:
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        UInt16TokenStream(empty)

    odd = tmp_path / "odd.bin"
    odd.write_bytes(b"\x00")
    with pytest.raises(ValueError, match="odd byte length"):
        UInt16TokenStream(odd)


def test_data_preparation_requires_a_pinned_revision(tmp_path) -> None:
    with pytest.raises(ValueError, match="revision"):
        DataPreparationConfig(
            dataset="example/dataset",
            dataset_config="default",
            revision="",
            output_dir=tmp_path,
            train_tokens=100,
            val_tokens=100,
        )


def test_preparation_writes_exact_disjoint_splits_and_hash_manifest(
    tmp_path, monkeypatch
) -> None:
    stream_closed = False

    def rows():
        nonlocal stream_closed
        try:
            yield from ({"text": str(index)} for index in range(10))
        finally:
            stream_closed = True

    class FakeDatasets:
        __version__ = "test"

        @staticmethod
        def get_dataset_config_names(*args, **kwargs):
            assert args == ("example/dataset",)
            assert kwargs["revision"] == "deadbeef"
            return ["default", "pinned"]

        @staticmethod
        def load_dataset(*args, **kwargs):
            assert args[:2] == ("example/dataset", "pinned")
            assert kwargs["revision"] == "deadbeef"
            assert kwargs["streaming"] is True
            return rows()

    class FakeTokenizer:
        eot_token = 9

        @staticmethod
        def encode_ordinary(text: str) -> list[int]:
            return [int(text)] * 3

    fake_tiktoken = SimpleNamespace(
        __version__="test", get_encoding=lambda name: FakeTokenizer()
    )
    monkeypatch.setattr(
        prepare_data,
        "_load_optional_dependencies",
        lambda: (FakeDatasets, fake_tiktoken),
    )

    config = DataPreparationConfig(
        dataset="example/dataset",
        dataset_config="pinned",
        revision="deadbeef",
        output_dir=tmp_path,
        train_tokens=8,
        val_tokens=7,
    )
    manifest = prepare_data.prepare_dataset(config)
    assert stream_closed is True

    assert UInt16TokenStream(tmp_path / "val.bin").read_tokens(0, 7).tolist() == [
        0,
        0,
        0,
        9,
        1,
        1,
        9,
    ]
    assert UInt16TokenStream(tmp_path / "train.bin").read_tokens(0, 8).tolist() == [
        2,
        2,
        2,
        9,
        3,
        3,
        3,
        9,
    ]
    assert manifest["source"]["revision"] == "deadbeef"
    assert manifest["splits"]["validation"]["num_tokens"] == 7
    assert manifest["splits"]["train"]["num_tokens"] == 8
    checksums = (tmp_path / "SHA256SUMS").read_text(encoding="utf-8")
    assert manifest["splits"]["train"]["sha256"] in checksums
    assert manifest["splits"]["validation"]["sha256"] in checksums

    verified = verify_token_manifest(
        tmp_path / "train.bin", "train", expected_num_tokens=8
    )
    assert verified["sha256"] == manifest["splits"]["train"]["sha256"]
    provenance_dir = tmp_path / "run"
    provenance = record_data_provenance(
        {
            "synthetic": False,
            "vocab_size": 10,
            "train_path": str(tmp_path / "train.bin"),
            "val_path": str(tmp_path / "val.bin"),
            "train_tokens": 8,
            "val_tokens": 7,
        },
        provenance_dir,
        splits=("train", "val"),
    )
    assert provenance["kind"] == "prepared_uint16"
    assert (provenance_dir / "data_manifest.json").is_file()
    assert (provenance_dir / "data_provenance.json").is_file()

    with (tmp_path / "train.bin").open("r+b") as handle:
        handle.write(b"\xff\xff")
    with pytest.raises(ConfigurationError, match="SHA256 mismatch"):
        verify_token_manifest(tmp_path / "train.bin", "train")


def test_preparation_rejects_config_missing_from_pinned_revision(
    tmp_path, monkeypatch
) -> None:
    load_called = False

    class FakeDatasets:
        @staticmethod
        def get_dataset_config_names(*args, **kwargs):
            assert args == ("example/dataset",)
            assert kwargs == {"revision": "old-revision"}
            return ["default", *(f"snapshot-{index}" for index in range(12))]

        @staticmethod
        def load_dataset(*args, **kwargs):
            nonlocal load_called
            load_called = True
            raise AssertionError("load_dataset must not run after a failed preflight")

    fake_tiktoken = SimpleNamespace(get_encoding=lambda name: None)
    monkeypatch.setattr(
        prepare_data,
        "_load_optional_dependencies",
        lambda: (FakeDatasets, fake_tiktoken),
    )
    config = DataPreparationConfig(
        dataset="example/dataset",
        dataset_config="sample-10BT",
        revision="old-revision",
        output_dir=tmp_path,
        train_tokens=8,
        val_tokens=7,
    )

    with pytest.raises(prepare_data.DatasetSourceError) as caught:
        prepare_data.prepare_dataset(config)

    message = str(caught.value)
    assert "sample-10BT" in message
    assert "old-revision" in message
    assert "Available configs (13)" in message
    assert "snapshot-11" not in message
    assert load_called is False


def test_main_reports_dataset_source_error_without_traceback(
    tmp_path, monkeypatch, capsys
) -> None:
    def fail(_config):
        raise prepare_data.DatasetSourceError("config and revision do not match")

    monkeypatch.setattr(prepare_data, "prepare_dataset", fail)
    result = prepare_data.main(
        [
            "--dataset",
            "example/dataset",
            "--dataset-config",
            "missing",
            "--revision",
            "deadbeef",
            "--output-dir",
            str(tmp_path),
            "--train-tokens",
            "8",
            "--val-tokens",
            "7",
        ]
    )

    captured = capsys.readouterr()
    assert result == 2
    assert captured.out == ""
    assert captured.err == "error: config and revision do not match\n"
    assert "Traceback" not in captured.err
