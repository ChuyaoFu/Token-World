"""Frozen-latent DeepStack adapter and Qwen3-VL feature injection."""
import torch
from torch import nn

class DeepStackAdapter(nn.Module):
    """Frozen decoded merger + latent, with learned spatial/global mixing."""
    def __init__(self, width=512):
        super().__init__()
        self.merger_proj = nn.Sequential(nn.LayerNorm(2560), nn.Linear(2560, width))
        self.latent_proj = nn.Sequential(nn.LayerNorm(16), nn.Linear(16, width))
        self.position = nn.Parameter(torch.zeros(1, 108, width))
        layer = nn.TransformerEncoderLayer(width, 8, 4*width, dropout=0.,
                                           activation='gelu', batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 3*2560))
        self.register_buffer('target_mean', torch.zeros(3, 2560))
        self.register_buffer('target_std', torch.ones(3, 2560))

    def forward(self, latent, merger):
        x = self.merger_proj(merger) + self.latent_proj(latent) + self.position
        return self.head(self.blocks(x)).view(-1, 108, 3, 2560).permute(0, 2, 1, 3)

    def features(self, latent, merger):
        return self(latent, merger).float()*self.target_std[None,:,None,:] + self.target_mean[None,:,None,:]


def inject_actions(policy, template, merger, deep=None, *, readout="legacy_post_norm"):
    """Use the Qwen text backbone directly, preserving visual masks and RoPE."""
    dev = next(policy.parameters()).device
    model = policy.qwen_vl_interface.model
    backbone = model.model
    ids = template['input_ids'].to(dev)
    attention = template['attention_mask'].to(dev)
    grid = template['image_grid_thw'].to(dev)
    embeds = model.get_input_embeddings()(ids)
    merger = merger.reshape(-1, 2560).to(embeds)
    mask, _ = backbone.get_placeholder_mask(ids, inputs_embeds=embeds, image_features=merger)
    embeds = embeds.masked_scatter(mask, merger)
    positions, _ = backbone.get_rope_index(input_ids=ids, image_grid_thw=grid,
                                           video_grid_thw=None, attention_mask=attention)
    deep = None if deep is None else [d.reshape(-1,2560).to(embeds) for d in deep]
    text_model = backbone.language_model
    if readout not in {"native_pre_norm", "legacy_post_norm"}:
        raise ValueError(f"Unknown action readout {readout!r}")
    before_norm = []
    handle = None
    if readout == "native_pre_norm":
        # QwenOFT.predict_action uses the final decoder block output in this bundle.
        def capture_input(module, args):
            before_norm.append(args[0])
        handle = text_model.norm.register_forward_pre_hook(capture_input)
    try:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            hidden = text_model(inputs_embeds=embeds, attention_mask=attention,
                position_ids=positions, visual_pos_masks=mask[...,0], deepstack_visual_embeds=deep,
                use_cache=False, return_dict=True).last_hidden_state
    finally:
        if handle is not None:
            handle.remove()
    if readout == "native_pre_norm":
        hidden, = before_norm
    queries = policy._gather_action_token_embeddings(hidden, ids, action_token_id=policy.action_token_id)
    with torch.autocast('cuda', dtype=torch.float32):
        return policy.action_model.predict_action(queries).float()


@torch.no_grad()
def adapter_policy_actions(policy, codec, adapter, template, raw):
    """Query the deployed policy on one denormalized, predicted cam_high latent."""
    raw = raw.reshape(1, 108, 16)
    # Match the singleton codec kernel used in simulator evaluation.
    with torch.enable_grad():
        merger = codec.decode(raw, [108])[0].reshape(1, 108, 2560).detach()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        merger = merger.bfloat16().float()
        deep = adapter.features(raw.bfloat16().float(), merger)[0]
        return inject_actions(policy, template, merger, deep, readout="native_pre_norm")


def native_actions(policy, inputs):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        hidden = policy.qwen_vl_interface.model.model(**inputs, use_cache=False, return_dict=True).last_hidden_state
    queries = policy._gather_action_token_embeddings(hidden, inputs['input_ids'], action_token_id=policy.action_token_id)
    with torch.autocast('cuda', dtype=torch.float32):
        return policy.action_model.predict_action(queries).float()


