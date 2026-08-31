# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "convert_deepseek_v4_to_w8a8.py"
SPEC = importlib.util.spec_from_file_location("convert_to_w8a8", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
convert = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(convert)


def _write_source_metadata(tmp_path, weight_map, *, create_shards=True):
    config = {
        "model_type": "deepseek_v4",
        "quantization_config": {"quant_method": "fp8"},
        "num_hidden_layers": 1,
        "compress_ratios": [0, 0],
        "n_routed_experts": 1,
    }
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    if create_shards:
        for shard_name in set(weight_map.values()):
            (tmp_path / shard_name).touch()


def test_mxfp4_dequantization_unpacks_low_nibble_first():
    packed = torch.tensor([[0x21, -0x68]], dtype=torch.int8)
    scale = torch.full((1, 1), 2.0, dtype=torch.float32)

    actual = convert._dequantize_hybrid_weight(packed, scale)

    assert torch.equal(actual, torch.tensor([[1.0, 2.0, -0.0, -1.0]]))


def test_fp8_dequantization_expands_128_by_128_block_scale():
    weight = torch.ones((129, 129), dtype=torch.float8_e4m3fn)
    scale = torch.tensor([[1.0, 2.0], [4.0, 8.0]], dtype=torch.float32)

    actual = convert._dequantize_hybrid_weight(weight, scale)

    assert actual[0, 0] == 1.0
    assert actual[0, 128] == 2.0
    assert actual[128, 0] == 4.0
    assert actual[128, 128] == 8.0


def test_fp8_dequantization_rejects_other_one_byte_dtypes():
    weight = torch.ones((1, 1), dtype=torch.uint8)
    scale = torch.ones((1, 1), dtype=torch.float32)

    with pytest.raises(ValueError, match="unsupported source quantized dtype"):
        convert._dequantize_hybrid_weight(weight, scale)


def test_load_e8m0_scale_decodes_exponents(tmp_path):
    header = {
        "w.scale": {"dtype": "F8_E8M0", "shape": [2, 2], "data_offsets": [0, 4]}
    }
    encoded_header = json.dumps(header).encode()
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(
        len(encoded_header).to_bytes(8, "little")
        + encoded_header
        + bytes([127, 128, 126, 129])
    )

    actual = convert._load_e8m0_scale(checkpoint, "w.scale")

    assert torch.equal(actual, torch.tensor([[1.0, 2.0], [0.5, 4.0]]))


def test_load_e8m0_scale_rejects_nan(tmp_path):
    header = {"w.scale": {"dtype": "F8_E8M0", "shape": [1], "data_offsets": [0, 1]}}
    encoded_header = json.dumps(header).encode()
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(len(encoded_header).to_bytes(8, "little") + encoded_header + b"\xff")

    with pytest.raises(ValueError, match="contains NaN"):
        convert._load_e8m0_scale(checkpoint, "w.scale")


def test_serving_quantization_is_symmetric_per_output_channel():
    weight = torch.tensor([[0.0, 1.0, -2.0], [0.5, -0.5, 0.25]], dtype=torch.float32)

    quantized, scale = convert._quantize_weight_per_output(weight)

    assert torch.equal(quantized, torch.tensor([[0, 64, -127], [127, -127, 64]], dtype=torch.int8))
    assert torch.allclose(scale, torch.tensor([2.0 / 127.0, 0.5 / 127.0]))


def test_quantized_weight_selection_matches_deepseek_v4_serving_contract():
    ratios = [0, 0, 4, 128]

    assert convert._is_serving_quantized_weight("layers.0.attn.wq_b.weight", ratios)
    assert convert._is_serving_quantized_weight("layers.0.ffn.experts.17.w2.weight", ratios)
    assert convert._is_serving_quantized_weight("layers.2.attn.indexer.wq_b.weight", ratios)
    assert convert._is_serving_quantized_weight("mtp.0.e_proj.weight", ratios)
    assert not convert._is_serving_quantized_weight("layers.3.attn.indexer.wq_b.weight", ratios)
    assert not convert._is_serving_quantized_weight("layers.0.attn.wq_a.weight", ratios)


def test_quantized_weight_selection_includes_all_dspark_draft_layers():
    ratios = [0, 0, 4, 128]

    assert convert._is_serving_quantized_weight(
        "mtp.2.ffn.experts.17.w2.weight", ratios, mtp_layer_count=3
    )
    assert convert._is_serving_quantized_weight(
        "mtp.1.attn.wq_b.weight", ratios, mtp_layer_count=3
    )
    assert not convert._is_serving_quantized_weight(
        "mtp.3.attn.wq_b.weight", ratios, mtp_layer_count=3
    )
    assert not convert._is_serving_quantized_weight(
        "mtp.2.main_proj.weight", ratios, mtp_layer_count=3
    )


def test_mtp_layer_count_uses_dspark_target_layers():
    assert convert._mtp_layer_count({"num_hidden_layers": 43}) == 1
    assert (
        convert._mtp_layer_count(
            {"num_hidden_layers": 43, "dspark_target_layer_ids": [40, 41, 42]}
        )
        == 3
    )


@pytest.mark.parametrize(
    "target_layer_ids",
    [[], [40, 40], [-1], [43], ["40"]],
)
def test_mtp_layer_count_rejects_invalid_dspark_target_layers(target_layer_ids):
    with pytest.raises(ValueError, match="invalid dspark_target_layer_ids"):
        convert._mtp_layer_count(
            {"num_hidden_layers": 43, "dspark_target_layer_ids": target_layer_ids}
        )


def test_dspark_quantization_metadata_ignores_bf16_projections():
    ignore = convert._quantization_ignore(43, [0] * 43, mtp_layer_count=3)

    assert "mtp.1.attn.wq_a" in ignore
    assert "mtp.2.attn.wkv" in ignore
    assert "mtp.0.main_proj" in ignore
    assert "mtp.2.markov_head.markov_w1" in ignore
    assert "mtp.2.markov_head.markov_w2" in ignore
    assert "mtp.2.confidence_head.proj" in ignore


def test_output_config_keeps_all_dspark_ratios_by_default():
    source_config = {
        "num_hidden_layers": 43,
        "compress_ratios": [0] * 46,
    }

    output_config = convert._build_output_config(
        source_config,
        source_config["compress_ratios"],
        mtp_layer_count=3,
        no_spec_compatible=False,
    )

    assert len(output_config["compress_ratios"]) == 46


def test_no_spec_output_config_matches_current_serving_loader_contract():
    from pypto_serving.model.model_loader import _validate_deepseek_v4_weight_index

    source_config = {
        "num_hidden_layers": 43,
        "compress_ratios": [0] * 46,
    }
    output_config = convert._build_output_config(
        source_config,
        source_config["compress_ratios"],
        mtp_layer_count=3,
        no_spec_compatible=True,
    )
    required_weights = {
        name: "model.safetensors"
        for name in (
            "embed.weight",
            "norm.weight",
            "head.weight",
            "layers.0.attn.wq_b.weight",
            "layers.0.attn.wq_b.scale",
            "layers.0.attn.wo_b.weight",
            "layers.0.attn.wo_b.scale",
            "layers.0.ffn.experts.0.w1.weight",
            "layers.0.ffn.experts.0.w1.scale",
        )
    }

    assert len(output_config["compress_ratios"]) == 44
    _validate_deepseek_v4_weight_index(required_weights, output_config)


def test_output_weight_map_only_adds_scales_available_in_the_source():
    source_map = {
        "layers.0.attn.wq_b.weight": "model-1.safetensors",
        "layers.0.attn.wq_b.scale": "model-1.safetensors",
        "layers.0.attn.wo_b.weight": "model-1.safetensors",
        "embed.weight": "model-1.safetensors",
    }

    output_map = convert._build_output_weight_map(source_map, [0])

    assert output_map == {
        "layers.0.attn.wq_b.weight": "model-1.safetensors",
        "layers.0.attn.wq_b.scale": "model-1.safetensors",
        "layers.0.attn.wo_b.weight": "model-1.safetensors",
        "embed.weight": "model-1.safetensors",
    }


def test_source_validation_rejects_a_selected_weight_without_scale(tmp_path):
    weight_name = "layers.0.attn.wq_b.weight"
    shard_name = "model-1.safetensors"
    _write_source_metadata(tmp_path, {weight_name: shard_name})

    with pytest.raises(ValueError, match="missing source scale for serving-quantized weight"):
        convert._validate_source(tmp_path)


def test_source_validation_requires_exact_dspark_compress_ratio_count(tmp_path):
    shard_name = "model-1.safetensors"
    _write_source_metadata(tmp_path, {"embed.weight": shard_name})
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config["num_hidden_layers"] = 2
    config["compress_ratios"] = [0, 0]
    config["dspark_target_layer_ids"] = [0]
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="expected 3, got 2"):
        convert._validate_source(tmp_path)


def test_source_validation_rejects_an_incomplete_dspark_layer(tmp_path):
    shard_name = "model-1.safetensors"
    required_names = convert._required_dspark_tensor_names(
        mtp_layer_count=1, n_routed_experts=1
    )
    assert len(required_names) == 45
    assert "mtp.0.main_proj.weight" in required_names
    assert "mtp.0.markov_head.markov_w2.weight" in required_names
    missing_name = "mtp.0.ffn.experts.0.w2.weight"
    weight_map = {name: shard_name for name in required_names - {missing_name}}
    _write_source_metadata(tmp_path, weight_map)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config["num_hidden_layers"] = 2
    config["compress_ratios"] = [0, 0, 0]
    config["dspark_target_layer_ids"] = [0]
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match=missing_name):
        convert._validate_source(tmp_path)

    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard_name for name in required_names}})
    )
    convert._validate_source(tmp_path)


@pytest.mark.parametrize("shard_name", ["..", "../escaped.safetensors", "/tmp/escaped.safetensors"])
def test_source_validation_rejects_unsafe_shard_paths(tmp_path, shard_name):
    _write_source_metadata(tmp_path, {"embed.weight": shard_name}, create_shards=False)

    with pytest.raises(ValueError, match="unsafe shard filename"):
        convert._validate_source(tmp_path)


def test_dry_run_opens_and_validates_source_shards(tmp_path, monkeypatch):
    shard_name = "model-1.safetensors"
    _write_source_metadata(tmp_path, {"embed.weight": shard_name})
    (tmp_path / shard_name).write_bytes(b"not a safetensors file")

    def reject_invalid_shard(*args, **kwargs):
        raise RuntimeError("invalid header")

    monkeypatch.setattr(convert, "safe_open", reject_invalid_shard)
    monkeypatch.setattr(convert, "load_file", object())
    monkeypatch.setattr(convert, "save_file", object())

    with pytest.raises(ValueError, match="invalid safetensors shard"):
        convert.convert_checkpoint(tmp_path, tmp_path / "output", resume=False, dry_run=True)


def test_resume_validation_rejects_a_wrong_tensor_dtype(tmp_path, monkeypatch):
    shard = tmp_path / "model-1.safetensors"
    monkeypatch.setattr(
        convert,
        "_read_tensor_specs",
        lambda path: {"layers.0.attn.wq_b.weight": ("BF16", (4, 8))},
    )

    with pytest.raises(ValueError, match="dtypes, or shapes differ"):
        convert._validate_resumable_shard(
            shard, {"layers.0.attn.wq_b.weight": ("I8", (4, 8))}
        )


def test_source_header_validation_rejects_a_wrong_scale_dtype(tmp_path, monkeypatch):
    source_map = {"w.weight": "model.safetensors", "w.scale": "model.safetensors"}
    monkeypatch.setattr(
        convert,
        "_read_tensor_specs",
        lambda path: {"w.weight": ("F8_E4M3", (128, 128)), "w.scale": ("F32", (1, 1))},
    )

    with pytest.raises(ValueError, match="invalid source scale contract"):
        convert._expected_output_specs(tmp_path, source_map, source_map, [0])


def test_auxiliary_files_are_copied_atomically(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    (input_dir / "tokenizer.json").write_text('{"version": 1}')

    convert._copy_auxiliary_files(input_dir, output_dir)

    assert (output_dir / "tokenizer.json").read_text() == '{"version": 1}'
    assert not (output_dir / ".tokenizer.json.tmp").exists()


def test_input_and_output_directories_are_required():
    with pytest.raises(SystemExit):
        convert._parse_args([])


def test_no_spec_compatible_cli_option_is_explicit():
    args = convert._parse_args(
        ["--input-dir", "input", "--output-dir", "output", "--no-spec-compatible"]
    )

    assert args.no_spec_compatible is True


def test_safetensors_total_size_sums_tensor_data_offsets(tmp_path):
    header = {
        "first": {"dtype": "I8", "shape": [4], "data_offsets": [0, 4]},
        "second": {"dtype": "F32", "shape": [2], "data_offsets": [4, 12]},
        "__metadata__": {"format": "pt"},
    }
    encoded_header = json.dumps(header).encode()
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(len(encoded_header).to_bytes(8, "little") + encoded_header + bytes(12))

    assert convert._safetensors_total_size(checkpoint) == 12


def test_resume_allows_source_directory_relocation(tmp_path):
    source_a = tmp_path / "source-a"
    source_b = tmp_path / "source-b"
    output = tmp_path / "output"
    for source in (source_a, source_b):
        _write_source_metadata(source, {})
    output.mkdir()
    convert._write_conversion_marker(output, convert._conversion_marker(source_a, complete=False))

    convert._validate_conversion_marker(source_b, output)


def test_resume_rejects_a_different_source_index(tmp_path):
    source_a = tmp_path / "source-a"
    source_b = tmp_path / "source-b"
    output = tmp_path / "output"
    output.mkdir()
    _write_source_metadata(source_a, {"a": "a"}, create_shards=False)
    _write_source_metadata(source_b, {"b": "b"}, create_shards=False)
    convert._write_conversion_marker(output, convert._conversion_marker(source_a, complete=False))

    with pytest.raises(ValueError, match="source_index_sha256"):
        convert._validate_conversion_marker(source_b, output)


def test_resume_rejects_changed_source_shard_payload(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    output.mkdir()
    _write_source_metadata(source, {})
    shard_name = "model-1.safetensors"
    shard = source / shard_name
    shard.write_bytes(b"checkpoint-a")
    marker = convert._conversion_marker(source, shard_names=[shard_name], complete=False)
    convert._write_conversion_marker(output, marker)
    shard.write_bytes(b"checkpoint-b")

    with pytest.raises(ValueError, match="source_shards_sha256"):
        convert._validate_conversion_marker(source, output, [shard_name])
