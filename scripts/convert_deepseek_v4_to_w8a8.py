# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Convert DeepSeek-V4-Flash Hybrid FP8/MXFP4 weights for PyPTO W8A8 serving.

Run this repository-local utility before starting PyPTO serving with the
released DeepSeek-V4-Flash checkpoint.

The PyPTO kernels consume static, symmetric, per-output-channel INT8 weights
and FP32 dequantization scales. Activations are quantized dynamically per
token at runtime. The source checkpoint mixes 128x128-block FP8 weights and
32-element-group MXFP4 expert weights, so it must be dequantized before the
serving quantization is applied.

The conversion is shard-by-shard and can be resumed. Existing output shards
are never overwritten.

Pass ``--no-spec-compatible`` to write the ``num_hidden_layers + 1``
``compress_ratios`` contract required by current PyPTO serving when MTP is
disabled. All configured MTP/DSpark draft weights are still converted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Mapping, Sequence

import torch

try:
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
except ImportError:
    safe_open = None
    load_file = None
    save_file = None


FP8_BLOCK_SIZE = 128
MXFP4_GROUP_SIZE = 32
INT8_MAX = 127.0
INT8_AMAX_EPS = 1e-4
CONVERSION_FORMAT = "pypto-deepseek-v4-w8a8-v3"
CONVERSION_MARKER = ".pypto-w8a8-conversion.json"
TensorSpec = tuple[str, tuple[int, ...]]

_LAYER_QUANT_RE = re.compile(
    r"^layers\.(?P<layer>\d+)\.(?:"
    r"attn\.(?:wq_b|wo_b)|"
    r"attn\.indexer\.wq_b|"
    r"ffn\.shared_experts\.w[123]|"
    r"ffn\.experts\.\d+\.w[123]"
    r")\.weight$"
)
_MTP_QUANT_RE = re.compile(
    r"^mtp\.(?P<mtp_layer>\d+)\.(?:"
    r"attn\.(?:wq_b|wo_b)|"
    r"ffn\.shared_experts\.w[123]|"
    r"ffn\.experts\.\d+\.w[123]|"
    r"(?:e_proj|h_proj)"
    r")\.weight$"
)

_DSPARK_COMMON_TENSOR_SUFFIXES = (
    "attn.attn_sink",
    "attn.q_norm.weight",
    "attn.kv_norm.weight",
    "attn_norm.weight",
    "ffn_norm.weight",
    "ffn.gate.weight",
    "ffn.gate.bias",
    "hc_attn_base",
    "hc_attn_fn",
    "hc_attn_scale",
    "hc_ffn_base",
    "hc_ffn_fn",
    "hc_ffn_scale",
)
_DSPARK_SCALED_MODULE_SUFFIXES = (
    "attn.wq_a",
    "attn.wq_b",
    "attn.wkv",
    "attn.wo_a",
    "attn.wo_b",
    "ffn.shared_experts.w1",
    "ffn.shared_experts.w2",
    "ffn.shared_experts.w3",
)
_DSPARK_FIRST_LAYER_TENSOR_SUFFIXES = (
    "main_norm.weight",
    "main_proj.weight",
    "main_proj.scale",
)
_DSPARK_FINAL_LAYER_TENSOR_SUFFIXES = (
    "confidence_head.proj.weight",
    "markov_head.markov_w1.weight",
    "markov_head.markov_w2.weight",
    "norm.weight",
    "hc_head_base",
    "hc_head_fn",
    "hc_head_scale",
)

_MXFP4_VALUES = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc


def _require_safetensors() -> None:
    if safe_open is None or load_file is None or save_file is None:
        raise RuntimeError("safetensors is required: python -m pip install safetensors")


def _safe_shard_path(root: Path, filename: str) -> Path:
    relative_path = Path(filename)
    if filename in {"", ".", ".."} or relative_path.is_absolute() or relative_path.name != filename:
        raise ValueError(f"unsafe shard filename in model index: {filename!r}")
    return root / relative_path


def _mtp_layer_count(config: Mapping[str, object]) -> int:
    target_layer_ids = config.get("dspark_target_layer_ids")
    if target_layer_ids is None:
        return 1
    num_layers = int(config.get("num_hidden_layers", 0))
    if (
        not isinstance(target_layer_ids, list)
        or not target_layer_ids
        or not all(isinstance(layer_id, int) for layer_id in target_layer_ids)
        or len(set(target_layer_ids)) != len(target_layer_ids)
        or any(layer_id < 0 or layer_id >= num_layers for layer_id in target_layer_ids)
    ):
        raise ValueError("config.json has invalid dspark_target_layer_ids")
    return len(target_layer_ids)


def _required_dspark_tensor_names(
    mtp_layer_count: int, n_routed_experts: int
) -> set[str]:
    required: set[str] = set()
    for layer_id in range(mtp_layer_count):
        prefix = f"mtp.{layer_id}"
        suffixes = set(_DSPARK_COMMON_TENSOR_SUFFIXES)
        for module_suffix in _DSPARK_SCALED_MODULE_SUFFIXES:
            suffixes.update((f"{module_suffix}.weight", f"{module_suffix}.scale"))
        for expert_id in range(n_routed_experts):
            for weight_id in (1, 2, 3):
                module_suffix = f"ffn.experts.{expert_id}.w{weight_id}"
                suffixes.update((f"{module_suffix}.weight", f"{module_suffix}.scale"))
        if layer_id == 0:
            suffixes.update(_DSPARK_FIRST_LAYER_TENSOR_SUFFIXES)
        if layer_id == mtp_layer_count - 1:
            suffixes.update(_DSPARK_FINAL_LAYER_TENSOR_SUFFIXES)
        required.update(f"{prefix}.{suffix}" for suffix in suffixes)
    return required


def _validate_source(input_dir: Path) -> tuple[dict, dict[str, str]]:
    config = _read_json(input_dir / "config.json")
    model_type = str(config.get("model_type", "")).lower()
    architectures = {str(item).lower() for item in config.get("architectures", [])}
    if model_type != "deepseek_v4" and "deepseekv4forcausallm" not in architectures:
        raise ValueError(f"{input_dir} is not a DeepSeek-V4 checkpoint")

    source_quant = config.get("quantization_config") or {}
    if source_quant.get("quant_method") != "fp8":
        raise ValueError(
            "expected the original Hybrid FP8/MXFP4 checkpoint, "
            f"got quant_method={source_quant.get('quant_method')!r}"
        )

    num_layers = int(config.get("num_hidden_layers", 0))
    compress_ratios = config.get("compress_ratios")
    if num_layers <= 0 or not isinstance(compress_ratios, list):
        raise ValueError("config.json has an invalid num_hidden_layers/compress_ratios contract")
    mtp_layer_count = _mtp_layer_count(config)
    expected_ratio_count = num_layers + mtp_layer_count
    if len(compress_ratios) != expected_ratio_count:
        raise ValueError(
            "config.json compress_ratios must include one entry per hidden layer and "
            f"MTP/DSpark layer: expected {expected_ratio_count}, got {len(compress_ratios)}"
        )
    n_routed_experts = int(config.get("n_routed_experts", 0))
    if n_routed_experts <= 0:
        raise ValueError("config.json has an invalid n_routed_experts value")

    index = _read_json(input_dir / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("model.safetensors.index.json has no weight_map")
    normalized_map = {str(name): str(filename) for name, filename in weight_map.items()}
    shard_filenames = set(normalized_map.values())
    shard_paths = {filename: _safe_shard_path(input_dir, filename) for filename in shard_filenames}
    missing_shards = sorted(filename for filename, path in shard_paths.items() if not path.is_file())
    if missing_shards:
        raise ValueError(f"missing source shard: {shard_paths[missing_shards[0]]}")
    compress_ratios = tuple(int(value) for value in compress_ratios)
    missing_dspark_tensors = (
        sorted(
            _required_dspark_tensor_names(mtp_layer_count, n_routed_experts)
            - normalized_map.keys()
        )
        if config.get("dspark_target_layer_ids") is not None
        else []
    )
    if missing_dspark_tensors:
        raise ValueError(
            "checkpoint is missing required MTP/DSpark tensor: "
            f"{missing_dspark_tensors[0]}"
        )
    missing_scales = sorted(
        name
        for name in normalized_map
        if name.endswith(".weight")
        and _is_serving_quantized_weight(name, compress_ratios, mtp_layer_count)
        and _source_scale_name(name) not in normalized_map
    )
    if missing_scales:
        raise ValueError(f"missing source scale for serving-quantized weight: {missing_scales[0]}")
    return config, normalized_map


def _is_serving_quantized_weight(
    name: str, compress_ratios: Sequence[int], mtp_layer_count: int = 1
) -> bool:
    mtp_match = _MTP_QUANT_RE.fullmatch(name)
    if mtp_match:
        return int(mtp_match.group("mtp_layer")) < mtp_layer_count
    match = _LAYER_QUANT_RE.fullmatch(name)
    if match is None:
        return False
    if ".attn.indexer.wq_b." not in name:
        return True
    layer_id = int(match.group("layer"))
    return layer_id < len(compress_ratios) and int(compress_ratios[layer_id]) == 4


def _source_scale_name(weight_name: str) -> str:
    return f"{weight_name[:-len('.weight')]}.scale"


def _source_weight_name(scale_name: str) -> str:
    return f"{scale_name[:-len('.scale')]}.weight"


def _unpack_mxfp4(weight: torch.Tensor) -> torch.Tensor:
    if weight.dtype != torch.int8 or weight.ndim != 2:
        raise ValueError(f"MXFP4 weight must be rank-2 INT8, got {weight.dtype}/{tuple(weight.shape)}")
    packed = weight.view(torch.uint8)
    indices = torch.stack((packed & 0x0F, (packed >> 4) & 0x0F), dim=-1)
    return _MXFP4_VALUES[indices.long()].reshape(weight.shape[0], weight.shape[1] * 2)


def _dequantize_hybrid_weight(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Return the source FP8 or packed-MXFP4 weight in FP32."""
    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError(
            f"source quantized weight/scale must be rank-2, got {tuple(weight.shape)}/{tuple(scale.shape)}"
        )
    scale_f32 = scale.float()

    if weight.dtype == torch.int8:
        unpacked = _unpack_mxfp4(weight)
        expected_scale_shape = (
            unpacked.shape[0],
            (unpacked.shape[1] + MXFP4_GROUP_SIZE - 1) // MXFP4_GROUP_SIZE,
        )
        if tuple(scale.shape) != expected_scale_shape:
            raise ValueError(
                f"MXFP4 scale shape mismatch: expected {expected_scale_shape}, got {tuple(scale.shape)}"
            )
        expanded_scale = scale_f32.repeat_interleave(MXFP4_GROUP_SIZE, dim=1)
        return unpacked * expanded_scale[:, : unpacked.shape[1]]

    if weight.dtype != torch.float8_e4m3fn:
        raise ValueError(f"unsupported source quantized dtype: {weight.dtype}")
    expected_scale_shape = (
        (weight.shape[0] + FP8_BLOCK_SIZE - 1) // FP8_BLOCK_SIZE,
        (weight.shape[1] + FP8_BLOCK_SIZE - 1) // FP8_BLOCK_SIZE,
    )
    if tuple(scale.shape) != expected_scale_shape:
        raise ValueError(
            f"FP8 scale shape mismatch: expected {expected_scale_shape}, got {tuple(scale.shape)}"
        )
    expanded_scale = scale_f32.repeat_interleave(FP8_BLOCK_SIZE, dim=0).repeat_interleave(
        FP8_BLOCK_SIZE, dim=1
    )
    return weight.float() * expanded_scale[: weight.shape[0], : weight.shape[1]]


def _quantize_weight_per_output(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the exact per-output-channel policy used by the PyPTO kernels."""
    weight_f32 = weight.float()
    amax = weight_f32.abs().amax(dim=-1).clamp_min(INT8_AMAX_EPS)
    scale = amax / INT8_MAX
    quantized = torch.round(weight_f32 / scale.unsqueeze(-1))
    quantized = quantized.clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
    return quantized.contiguous(), scale.to(torch.float32).contiguous()


def _build_output_weight_map(
    source_map: Mapping[str, str],
    compress_ratios: Sequence[int],
    mtp_layer_count: int = 1,
) -> dict[str, str]:
    output_map: dict[str, str] = {}
    for name, filename in source_map.items():
        if name.endswith(".scale") and _source_weight_name(name) in source_map:
            continue
        output_map[name] = filename
        if (
            name.endswith(".weight")
            and _source_scale_name(name) in source_map
            and _is_serving_quantized_weight(name, compress_ratios, mtp_layer_count)
        ):
            output_map[_source_scale_name(name)] = filename
    return output_map


def _quantization_ignore(
    num_layers: int, compress_ratios: Sequence[int], mtp_layer_count: int = 1
) -> list[str]:
    """Build compressed-tensors metadata matching the converted tensor policy."""
    ignore: list[str] = []
    for layer_id in range(num_layers):
        prefix = f"layers.{layer_id}.attn"
        ignore.extend((f"{prefix}.wq_a", f"{prefix}.wkv", f"{prefix}.wo_a"))
        ratio = int(compress_ratios[layer_id])
        if ratio == 4:
            ignore.extend(
                (
                    f"{prefix}.indexer.weights_proj",
                    f"{prefix}.indexer.compressor.wgate",
                    f"{prefix}.indexer.compressor.wkv",
                    f"{prefix}.compressor.wgate",
                    f"{prefix}.compressor.wkv",
                )
            )
        elif ratio == 128:
            ignore.extend((f"{prefix}.compressor.wgate", f"{prefix}.compressor.wkv"))
    for mtp_layer in range(mtp_layer_count):
        prefix = f"mtp.{mtp_layer}"
        ignore.extend(
            (
                f"{prefix}.attn.wq_a",
                f"{prefix}.attn.wkv",
                f"{prefix}.attn.wo_a",
                f"{prefix}.head",
            )
        )
    if mtp_layer_count > 1:
        ignore.extend(
            (
                "mtp.0.main_proj",
                f"mtp.{mtp_layer_count - 1}.markov_head.markov_w1",
                f"mtp.{mtp_layer_count - 1}.markov_head.markov_w2",
                f"mtp.{mtp_layer_count - 1}.confidence_head.proj",
            )
        )
    ignore.append("head")
    return ignore


def _serving_quantization_config(
    num_layers: int, compress_ratios: Sequence[int], mtp_layer_count: int = 1
) -> dict:
    return {
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "input_activations": {
                    "actorder": None,
                    "block_structure": None,
                    "dynamic": True,
                    "group_size": None,
                    "num_bits": 8,
                    "observer": "memoryless",
                    "observer_kwargs": {},
                    "strategy": "token",
                    "symmetric": True,
                    "type": "int",
                },
                "activation_use_clip": False,
                "output_activations": None,
                "weights": {
                    "actorder": None,
                    "block_structure": None,
                    "dynamic": False,
                    "group_size": None,
                    "num_bits": 8,
                    "observer": "minmax",
                    "observer_kwargs": {},
                    "strategy": "channel",
                    "symmetric": True,
                    "type": "int",
                },
            }
        },
        "format": "int-quantized",
        "global_compression_ratio": 1,
        "ignore": _quantization_ignore(num_layers, compress_ratios, mtp_layer_count),
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
        "kv_cache_scheme": None,
        "li_cache_scheme": {"type": "int", "num_bits": 8},
    }


def _build_output_config(
    config: Mapping[str, object],
    compress_ratios: Sequence[int],
    mtp_layer_count: int,
    *,
    no_spec_compatible: bool,
) -> dict:
    num_layers = int(config["num_hidden_layers"])
    output_ratios = tuple(int(value) for value in compress_ratios)
    if no_spec_compatible:
        output_ratios = output_ratios[: num_layers + 1]

    output_config = dict(config)
    output_config["compress_ratios"] = list(output_ratios)
    output_config["quantization_config"] = _serving_quantization_config(
        num_layers, compress_ratios, mtp_layer_count
    )
    return output_config


def _load_scale(
    input_dir: Path,
    source_map: Mapping[str, str],
    current_filename: str,
    current_tensors: Mapping[str, torch.Tensor],
    scale_name: str,
) -> torch.Tensor:
    scale_filename = source_map.get(scale_name)
    if scale_filename is None:
        raise ValueError(f"missing source scale for quantized weight: {scale_name}")
    return _load_e8m0_scale(_safe_shard_path(input_dir, scale_filename), scale_name)


def _load_e8m0_scale(path: Path, name: str) -> torch.Tensor:
    """Load an F8_E8M0 tensor on Torch versions without that dtype."""
    with path.open("rb") as file:
        header_size_bytes = file.read(8)
        if len(header_size_bytes) != 8:
            raise ValueError(f"invalid safetensors header in {path}")
        header_size = int.from_bytes(header_size_bytes, byteorder="little", signed=False)
        try:
            header = json.loads(file.read(header_size))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid safetensors header JSON in {path}: {exc}") from exc
        metadata = header.get(name) if isinstance(header, dict) else None
        if not isinstance(metadata, dict) or metadata.get("dtype") != "F8_E8M0":
            raise ValueError(f"missing F8_E8M0 tensor {name!r} in {path}")
        shape = metadata.get("shape")
        offsets = metadata.get("data_offsets")
        if (
            not isinstance(shape, list)
            or not all(isinstance(size, int) and size >= 0 for size in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(offset, int) for offset in offsets)
        ):
            raise ValueError(f"invalid F8_E8M0 tensor metadata for {name!r} in {path}")
        element_count = 1
        for size in shape:
            element_count *= size
        if offsets[0] < 0 or offsets[1] - offsets[0] != element_count:
            raise ValueError(f"invalid F8_E8M0 data offsets for {name!r} in {path}")
        file.seek(8 + header_size + offsets[0])
        payload = file.read(element_count)
    if len(payload) != element_count:
        raise ValueError(f"truncated F8_E8M0 tensor {name!r} in {path}")
    encoded = torch.frombuffer(bytearray(payload), dtype=torch.uint8).clone()
    if torch.any(encoded == 0xFF):
        raise ValueError(f"F8_E8M0 tensor {name!r} in {path} contains NaN")
    scale = torch.ldexp(torch.ones(element_count, dtype=torch.float32), encoded.int() - 127)
    return scale.reshape(shape)


def _load_shard_tensors(
    path: Path, source_map: Mapping[str, str]
) -> dict[str, torch.Tensor]:
    """Load a shard without asking Torch to materialize F8_E8M0 scales."""
    assert safe_open is not None
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt", device="cpu") as reader:
        for name in reader.keys():
            if name.endswith(".scale") and _source_weight_name(name) in source_map:
                continue
            tensors[name] = reader.get_tensor(name)
    return tensors


def _read_tensor_specs(path: Path) -> dict[str, TensorSpec]:
    assert safe_open is not None
    try:
        with safe_open(str(path), framework="pt", device="cpu") as reader:
            result = {}
            for name in reader.keys():
                tensor_slice = reader.get_slice(name)
                result[name] = (str(tensor_slice.get_dtype()), tuple(tensor_slice.get_shape()))
            return result
    except Exception as exc:
        raise ValueError(f"invalid safetensors shard {path}: {exc}") from exc


def _expected_output_specs(
    input_dir: Path,
    source_map: Mapping[str, str],
    output_map: Mapping[str, str],
    compress_ratios: Sequence[int],
    mtp_layer_count: int = 1,
) -> dict[str, dict[str, TensorSpec]]:
    source_specs: dict[str, TensorSpec] = {}
    for filename in set(source_map.values()):
        path = _safe_shard_path(input_dir, filename)
        shard_specs = _read_tensor_specs(path)
        expected_names = {name for name, shard in source_map.items() if shard == filename}
        if set(shard_specs) != expected_names:
            raise ValueError(f"tensor set in {path} differs from model.safetensors.index.json")
        source_specs.update(shard_specs)

    dequantized_shapes: dict[str, tuple[int, int]] = {}
    for weight_name in source_map:
        scale_name = _source_scale_name(weight_name) if weight_name.endswith(".weight") else ""
        if scale_name not in source_map:
            continue
        weight_dtype, weight_shape = source_specs[weight_name]
        scale_dtype, scale_shape = source_specs[scale_name]
        if len(weight_shape) != 2 or len(scale_shape) != 2:
            raise ValueError(f"source quantized weight/scale must be rank-2: {weight_name}")
        if weight_dtype == "I8":
            output_shape = (weight_shape[0], weight_shape[1] * 2)
            expected_scale_shape = (
                output_shape[0],
                (output_shape[1] + MXFP4_GROUP_SIZE - 1) // MXFP4_GROUP_SIZE,
            )
        elif weight_dtype == "F8_E4M3":
            output_shape = weight_shape
            expected_scale_shape = tuple(
                (size + FP8_BLOCK_SIZE - 1) // FP8_BLOCK_SIZE for size in output_shape
            )
        else:
            raise ValueError(f"unsupported source quantized dtype for {weight_name}: {weight_dtype}")
        if scale_dtype != "F8_E8M0" or scale_shape != expected_scale_shape:
            raise ValueError(
                f"invalid source scale contract for {weight_name}: "
                f"{scale_dtype}/{scale_shape}, expected F8_E8M0/{expected_scale_shape}"
            )
        dequantized_shapes[weight_name] = output_shape

    result: dict[str, dict[str, TensorSpec]] = {}
    for name, filename in output_map.items():
        weight_name = _source_weight_name(name) if name.endswith(".scale") else name
        if name.endswith(".scale") and weight_name in dequantized_shapes:
            spec = ("F32", (dequantized_shapes[weight_name][0],))
        elif name in dequantized_shapes:
            dtype = (
                "I8"
                if _is_serving_quantized_weight(name, compress_ratios, mtp_layer_count)
                else "BF16"
            )
            spec = (dtype, dequantized_shapes[name])
        else:
            spec = source_specs[name]
        result.setdefault(filename, {})[name] = spec
    return result


def _validate_resumable_shard(
    path: Path, expected_specs: Mapping[str, TensorSpec]
) -> None:
    try:
        actual_specs = _read_tensor_specs(path)
    except ValueError as exc:
        raise ValueError(f"cannot resume from invalid output shard {path}: {exc}") from exc
    if actual_specs != expected_specs:
        raise ValueError(
            f"cannot resume from {path}: tensor names, dtypes, or shapes differ from the "
            "expected conversion output"
        )


def _convert_shard(
    input_dir: Path,
    output_dir: Path,
    filename: str,
    source_map: Mapping[str, str],
    compress_ratios: Sequence[int],
    mtp_layer_count: int = 1,
) -> None:
    assert save_file is not None
    source_path = _safe_shard_path(input_dir, filename)
    output_path = _safe_shard_path(output_dir, filename)
    source_tensors = _load_shard_tensors(source_path, source_map)
    converted: dict[str, torch.Tensor] = {}

    for name, tensor in source_tensors.items():
        if name.endswith(".scale") and _source_weight_name(name) in source_map:
            continue
        scale_name = _source_scale_name(name) if name.endswith(".weight") else ""
        if scale_name not in source_map:
            converted[name] = tensor.contiguous()
            continue

        source_scale = _load_scale(input_dir, source_map, filename, source_tensors, scale_name)
        dequantized = _dequantize_hybrid_weight(tensor, source_scale)
        if _is_serving_quantized_weight(name, compress_ratios, mtp_layer_count):
            quantized, dequant_scale = _quantize_weight_per_output(dequantized)
            converted[name] = quantized
            converted[scale_name] = dequant_scale
        else:
            converted[name] = dequantized.to(torch.bfloat16).contiguous()

    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    save_file(converted, str(temporary_path), metadata={"format": "pt"})
    os.replace(temporary_path, output_path)


def _safetensors_total_size(path: Path) -> int:
    """Return the sum of tensor payload bytes without loading tensor data."""
    with path.open("rb") as file:
        header_size_bytes = file.read(8)
        if len(header_size_bytes) != 8:
            raise ValueError(f"invalid safetensors header in {path}")
        header_size = int.from_bytes(header_size_bytes, byteorder="little", signed=False)
        try:
            header = json.loads(file.read(header_size))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid safetensors header JSON in {path}: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"invalid safetensors header object in {path}")

    total_size = 0
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(metadata, dict):
            raise ValueError(f"invalid tensor metadata for {name!r} in {path}")
        offsets = metadata.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(value, int) for value in offsets)
            or offsets[0] < 0
            or offsets[1] < offsets[0]
        ):
            raise ValueError(f"invalid data offsets for {name!r} in {path}")
        total_size += offsets[1] - offsets[0]
    return total_size


def _copy_auxiliary_files(input_dir: Path, output_dir: Path) -> None:
    for source_path in input_dir.iterdir():
        if not source_path.is_file():
            continue
        if source_path.name in {"config.json", "model.safetensors.index.json"}:
            continue
        if source_path.suffix not in {".json", ".py", ".jinja", ".model"}:
            continue
        target_path = output_dir / source_path.name
        if not target_path.exists():
            temporary_path = target_path.with_name(f".{target_path.name}.tmp")
            shutil.copy2(source_path, temporary_path)
            os.replace(temporary_path, target_path)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_shards_digest(input_dir: Path, shard_names: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for filename in sorted(shard_names):
        digest.update(filename.encode())
        digest.update(bytes.fromhex(_file_digest(_safe_shard_path(input_dir, filename))))
    return digest.hexdigest()


def _conversion_marker(
    input_dir: Path, *, shard_names: Sequence[str] = (), complete: bool
) -> dict:
    return {
        "format": CONVERSION_FORMAT,
        "input_dir": str(input_dir),
        "source_config_sha256": _file_digest(input_dir / "config.json"),
        "source_index_sha256": _file_digest(input_dir / "model.safetensors.index.json"),
        "source_shards_sha256": _source_shards_digest(input_dir, shard_names),
        "complete": complete,
    }


def _write_conversion_marker(output_dir: Path, marker: Mapping[str, object]) -> None:
    marker_path = output_dir / CONVERSION_MARKER
    temporary_path = marker_path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(marker, indent=2) + "\n")
    os.replace(temporary_path, marker_path)


def _validate_conversion_marker(
    input_dir: Path, output_dir: Path, shard_names: Sequence[str] = ()
) -> dict:
    marker_path = output_dir / CONVERSION_MARKER
    if not marker_path.is_file():
        raise ValueError(
            f"cannot resume from {output_dir}: {CONVERSION_MARKER} is missing; "
            "use a new output directory"
        )
    marker = _read_json(marker_path)
    expected = _conversion_marker(
        input_dir, shard_names=shard_names, complete=bool(marker.get("complete"))
    )
    for key in (
        "format",
        "source_config_sha256",
        "source_index_sha256",
        "source_shards_sha256",
    ):
        if marker.get(key) != expected[key]:
            raise ValueError(
                f"cannot resume from {output_dir}: conversion marker field {key!r} does not match"
            )
    return marker


def convert_checkpoint(
    input_dir: Path,
    output_dir: Path,
    *,
    resume: bool,
    dry_run: bool,
    no_spec_compatible: bool = False,
) -> None:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if input_dir == output_dir:
        raise ValueError("input and output directories must be different")

    config, source_map = _validate_source(input_dir)
    num_layers = int(config["num_hidden_layers"])
    compress_ratios = tuple(int(value) for value in config["compress_ratios"])
    mtp_layer_count = _mtp_layer_count(config)
    output_config = _build_output_config(
        config,
        compress_ratios,
        mtp_layer_count,
        no_spec_compatible=no_spec_compatible,
    )
    output_map = _build_output_weight_map(source_map, compress_ratios, mtp_layer_count)
    shard_names = sorted(set(source_map.values()))
    _require_safetensors()
    expected_by_shard = _expected_output_specs(
        input_dir, source_map, output_map, compress_ratios, mtp_layer_count
    )
    selected_count = sum(
        name.endswith(".weight")
        and _is_serving_quantized_weight(name, compress_ratios, mtp_layer_count)
        for name in source_map
    )
    dequantized_count = sum(
        name.endswith(".weight")
        and _source_scale_name(name) in source_map
        and not _is_serving_quantized_weight(name, compress_ratios, mtp_layer_count)
        for name in source_map
    )

    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Shards: {len(shard_names)}")
    print(f"MTP/DSpark layers: {mtp_layer_count}")
    print(f"Output compress ratios: {len(output_config['compress_ratios'])}")
    print(f"Serving INT8 weights: {selected_count}")
    print(f"BF16 fallback weights: {dequantized_count}")
    if dry_run:
        return

    if output_dir.exists() and not resume:
        raise ValueError(f"output directory already exists: {output_dir}; use --resume or a new path")
    print("Verifying source checkpoint payloads...", flush=True)
    if output_dir.exists():
        marker = _validate_conversion_marker(input_dir, output_dir, shard_names)
    else:
        output_dir.mkdir(parents=True)
        marker = _conversion_marker(input_dir, shard_names=shard_names, complete=False)
        _write_conversion_marker(output_dir, marker)
    _copy_auxiliary_files(input_dir, output_dir)

    for shard_index, filename in enumerate(shard_names, start=1):
        output_path = _safe_shard_path(output_dir, filename)
        if output_path.exists():
            if not resume:
                raise ValueError(f"refusing to overwrite existing output shard: {output_path}")
            _validate_resumable_shard(output_path, expected_by_shard[filename])
            print(f"[{shard_index:02d}/{len(shard_names):02d}] resume: {filename}", flush=True)
            continue
        print(f"[{shard_index:02d}/{len(shard_names):02d}] convert: {filename}", flush=True)
        _convert_shard(
            input_dir, output_dir, filename, source_map, compress_ratios, mtp_layer_count
        )

    (output_dir / "config.json").write_text(json.dumps(output_config, indent=2) + "\n")
    total_size = sum(
        _safetensors_total_size(_safe_shard_path(output_dir, filename)) for filename in shard_names
    )
    output_index = {"metadata": {"total_size": total_size}, "weight_map": output_map}
    (output_dir / "model.safetensors.index.json").write_text(json.dumps(output_index, indent=2) + "\n")
    marker["complete"] = True
    _write_conversion_marker(output_dir, marker)
    print(f"Conversion complete: {output_dir}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="path to the original DeepSeek-V4-Flash checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory for the converted PyPTO W8A8 checkpoint",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="validate and skip completed output shards after an interrupted conversion",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the source and print the conversion plan without writing files",
    )
    parser.add_argument(
        "--no-spec-compatible",
        action="store_true",
        help=(
            "write the current serving config contract for use with --no-enable-mtp; "
            "all configured MTP/DSpark weights are still converted"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        convert_checkpoint(
            args.input_dir,
            args.output_dir,
            resume=args.resume,
            dry_run=args.dry_run,
            no_spec_compatible=args.no_spec_compatible,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
