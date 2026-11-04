"""Load RGB decoder weights from current and legacy research checkpoints."""
from tokenworld.decoder.model import VQGAN
from tokenworld.utils.checkpoint_compat import load_checkpoint_payload


def _convert_to_dict(obj):
    if isinstance(obj, dict):
        return {key: _convert_to_dict(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_to_dict(value) for value in obj]
    if hasattr(obj, "__dict__"):
        return {key: _convert_to_dict(value) for key, value in vars(obj).items()}
    return obj


def load_external_decoder(decoder_path, device="cuda", *, freeze=False):
    saved = load_checkpoint_payload(decoder_path)
    config = _convert_to_dict(saved["config"])
    config = config.get("decoder", config)
    keys = ("in_channel", "emb_dim", "channel", "channel_mult", "n_res_block",
            "resolution", "num_groups", "use_attention")
    decoder = VQGAN(**{key: config[key] for key in keys},
                    grid_size=config.get("grid_size") or config.get("latent_grid_size") or [9, 12])
    # Prefer the explicit decoder key when both are present.
    key = "decoder_state_dict" if "decoder_state_dict" in saved else "state_dict"
    decoder.load_state_dict(saved[key], strict=True)
    decoder = decoder.to(device).eval()
    if freeze:
        decoder.requires_grad_(False)
    return decoder
