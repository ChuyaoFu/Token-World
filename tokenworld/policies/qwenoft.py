# Adapted for the TokenWorld RoboTwin release: local imports and portable entrypoints.
# Source attribution and original licenses are listed in THIRD_PARTY_NOTICES.md.
from __future__ import annotations
import os
import sys
from pathlib import Path
import torch
from omegaconf import OmegaConf
from PIL import Image

def _add_path(path: Path) -> None:
    path = path.expanduser().resolve()
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path), device="cpu")
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unexpected policy checkpoint payload: {type(state)!r}")
    return {
        key[len("module.") :] if key.startswith("module.") else key: value
        for key, value in state.items()
    }


def _load_policy(bundle: Path, device: torch.device):
    """Load the exported QwenOFT policy and its static prompt template."""

    starvla_root = bundle / "code" / "starvla"
    _add_path(starvla_root)
    from starVLA.model.framework.base_framework import build_framework
    from starVLA.model.framework.share_tools import apply_config_compat

    config_path = bundle / "model" / "config.full.yaml"
    checkpoint_path = bundle / "model" / "checkpoints" / "steps_50000_pytorch_model.pt"
    if not checkpoint_path.is_file():
        # Make the script convenient for later exported checkpoints.
        candidates = sorted((bundle / "model" / "checkpoints").glob("*.pt"))
        if not candidates:
            raise FileNotFoundError(f"No policy checkpoint under {bundle / 'model' / 'checkpoints'}")
        checkpoint_path = candidates[-1]
    config = apply_config_compat(OmegaConf.load(config_path))
    if os.environ.get("QWEN_MODEL"):
        config.framework.qwenvl.base_vlm = os.environ["QWEN_MODEL"]
    policy = build_framework(config)
    incompatible = policy.load_state_dict(_load_state_dict(checkpoint_path), strict=False)
    missing = [key for key in incompatible.missing_keys if "num_batches_tracked" not in key]
    unexpected = [key for key in incompatible.unexpected_keys if "num_batches_tracked" not in key]
    if missing or unexpected:
        raise RuntimeError(f"Policy checkpoint mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}")
    policy.to(device).eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    return policy, config_path, checkpoint_path


def _action_instruction(policy, task: str, outcome: str) -> str:
    tokens = policy.action_token * policy.chunk_len
    return (
        f"{task}\nTrajectory outcome: {outcome}."
        f" Please predict the next {policy.chunk_len} robot actions: "
        f"<action>{tokens}<action>."
    )


def _build_policy_template(policy, task: str, outcome: str, image_size: tuple[int, int]):
    height, width = image_size
    dummy = Image.new("RGB", (width, height))
    inputs = policy.qwen_vl_interface.build_qwenvl_inputs(
        images=[[dummy]],
        instructions=[_action_instruction(policy, task, outcome)],
    )
    return {key: inputs[key] for key in ("input_ids", "attention_mask", "image_grid_thw")}


@torch.inference_mode()
def _policy_actions(policy, template: dict[str, torch.Tensor], features: torch.Tensor) -> torch.Tensor:
    """Run the policy with decoded Qwen merger features injected in-place."""

    causal_model = policy.qwen_vl_interface.model
    backbone = causal_model.model
    input_ids = template["input_ids"].to(next(policy.parameters()).device)
    attention_mask = template["attention_mask"].to(input_ids.device)
    image_grid_thw = template["image_grid_thw"].to(input_ids.device)
    inputs_embeds = causal_model.get_input_embeddings()(input_ids)
    flat_features = features.reshape(-1, features.shape[-1]).to(
        device=inputs_embeds.device,
        dtype=inputs_embeds.dtype,
    )
    image_mask, _ = backbone.get_placeholder_mask(
        input_ids,
        inputs_embeds=inputs_embeds,
        image_features=flat_features,
    )
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, flat_features)
    try:
        position_ids, _ = backbone.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            attention_mask=attention_mask,
        )
    except TypeError:
        position_ids = None
    forward_inputs = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "use_cache": False,
        "output_hidden_states": False,
        "return_dict": True,
    }
    if position_ids is not None:
        forward_inputs["position_ids"] = position_ids
    with torch.autocast(
        device_type=inputs_embeds.device.type,
        dtype=torch.bfloat16,
        enabled=inputs_embeds.device.type == "cuda",
    ):
        output = causal_model.model(**forward_inputs)
    queries = policy._gather_action_token_embeddings(
        output.last_hidden_state,
        input_ids,
        action_token_id=policy.action_token_id,
    )
    with torch.autocast(
        device_type=inputs_embeds.device.type,
        dtype=torch.float32,
        enabled=inputs_embeds.device.type == "cuda",
    ):
        actions = policy.action_model.predict_action(queries)
    return actions.float()

