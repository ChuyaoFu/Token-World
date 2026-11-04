"""
Utility functions for loading checkpoints and creating world models.

Supports two checkpoint formats:
1. FSDP checkpoints (old format): accelerate-based distributed checkpoints
2. Standard checkpoints (new format): single .pt files with model state dict
"""

import os
import yaml
import torch
from tokenworld.utils.checkpoint_compat import load_checkpoint_payload
from typing import Optional, Dict, Any, Tuple

# Add project root to path
import sys

from tokenworld.dynamics.model import DynamicsWorldModel


def exists(v):
    return v is not None


def load_config_from_ckpt(ckpt_path: str) -> Optional[Dict[str, Any]]:
    """
    Load config from checkpoint path.
    
    For FSDP checkpoints: looks for config.yaml in ckpt_path or parent directory
    For .pt checkpoints: config should be stored inside the checkpoint
    
    Args:
        ckpt_path: Path to checkpoint (directory for FSDP, file for .pt)
    
    Returns:
        config dict or None if not found
    """
    # If it's a .pt file, try to load config from it
    if ckpt_path.endswith('.pt'):
        try:
            checkpoint = load_checkpoint_payload(ckpt_path, map_location='cpu')
            if 'config' in checkpoint:
                return checkpoint['config']
        except Exception as e:
            print(f"Warning: Could not load config from .pt file: {e}")
        
        # Try config.yaml in same directory
        config_path = os.path.join(os.path.dirname(ckpt_path), 'config.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return None
    
    # For FSDP checkpoint directories
    config_path = os.path.join(ckpt_path, 'config.yaml')
    if not os.path.exists(config_path):
        # Try parent directory (e.g. if ckpt_path is .../best, config might be in ...)
        parent_path = os.path.dirname(ckpt_path)
        config_path = os.path.join(parent_path, 'config.yaml')
    
    if os.path.exists(config_path):
        print(f"Loading config from {config_path}")
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    return None


def create_world_model_from_config(config: Dict[str, Any]) -> DynamicsWorldModel:
    """
    Create DynamicsWorldModel instance from configuration dict.
    
    Args:
        config: Configuration dictionary with 'model', 'decoder', 'data' sections
    
    Returns:
        Initialized DynamicsWorldModel (weights not loaded)
    """
    model_config = config.get('model', {})
    data_config = config.get('data', {})
    
    # Handle actions - need to multiply by frameskip like in training
    frameskip = data_config.get('frameskip', 1)
    
    # Get original action dims
    original_discrete = model_config.get('original_discrete_actions', 
                                         model_config.get('num_discrete_actions', 0))
    if original_discrete is None:
        original_discrete = 0
    if isinstance(original_discrete, list):
        original_discrete = tuple(original_discrete)
    
    original_continuous = model_config.get('original_continuous_actions',
                                           model_config.get('num_continuous_actions', 0))
    
    # Multiply by frameskip to get actual dimensions used in training
    if isinstance(original_discrete, tuple):
        num_discrete_actions = original_discrete * frameskip if original_discrete != () else original_discrete
    elif isinstance(original_discrete, int):
        num_discrete_actions = original_discrete * frameskip if original_discrete != 0 else original_discrete
    else:
        num_discrete_actions = original_discrete
        
    num_continuous_actions = original_continuous * frameskip if original_continuous != 0 else original_continuous

    world_model = DynamicsWorldModel(
        dim=model_config.get('dim', 2048),
        dim_latent=model_config.get('dim_latent', 2048),
        num_latent_tokens=model_config.get('num_latent_tokens', 256),
        num_spatial_tokens=model_config.get('num_spatial_tokens', None),
        video_tokenizer=None,
        depth=model_config.get('depth', 4),
        num_discrete_actions=num_discrete_actions,
        num_continuous_actions=num_continuous_actions,
        dim_proprio=model_config.get('dim_proprio', 8),
        max_steps=model_config.get('max_steps', 64),
        attn_heads=model_config.get('attn_heads', 8),
        attn_dim_head=model_config.get('attn_dim_head', 64),
        use_adaln=model_config.get('use_adaln', False),
        num_video_views=model_config.get('num_video_views', 1),
        num_agents=model_config.get('num_agent_tokens', model_config.get('num_agents', 1)),
        spatial_grid_size=model_config.get('spatial_grid_size'),
        time_block_every=model_config.get('time_block_every', 4),
        use_time_rnn=model_config.get('use_time_rnn', True),
        pred_orig_latent=model_config.get('pred_orig_latent', True),
        use_idm=model_config.get('use_idm', False),
    )
    return world_model


def load_decoder_from_ckpt(decoder_path: str, device: str = 'cpu'):
    """Load and freeze the RGB decoder through the shared compatibility loader."""
    from tokenworld.decoder.checkpoint import load_external_decoder
    return load_external_decoder(decoder_path, device=device, freeze=True)


def _convert_dtensor_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert DTensor (distributed tensor) state dict to regular Tensor state dict.
    
    This is needed when loading checkpoints saved with FSDP that contain DTensor
    weights into a non-distributed model.
    
    Args:
        state_dict: State dict potentially containing DTensor values
        
    Returns:
        State dict with all DTensor converted to regular Tensor
    """
    converted = {}
    has_dtensor = False
    has_incomplete_shard = False
    
    for key, value in state_dict.items():
        if type(value).__name__ == 'DTensor' or hasattr(value, '_local_tensor'):
            has_dtensor = True
            # Priority: use full_tensor() to get complete tensor
            if hasattr(value, 'full_tensor'):
                try:
                    converted[key] = value.full_tensor().clone()
                    continue
                except Exception as e:
                    print(f"  Warning: full_tensor() failed for {key}: {e}")
            
            # Fallback: to_local() 
            if hasattr(value, 'to_local'):
                try:
                    converted[key] = value.to_local().clone()
                    continue
                except Exception as e:
                    print(f"  Warning: to_local() failed for {key}: {e}")
            
            # Last resort: _local_tensor (may be incomplete shard!)
            if hasattr(value, '_local_tensor'):
                converted[key] = value._local_tensor.clone()
                has_incomplete_shard = True
                print(f"  Warning: {key} extracted from _local_tensor (may be incomplete shard)")
            else:
                # Really last resort
                converted[key] = torch.tensor(value.tolist())
        else:
            converted[key] = value
    
    if has_dtensor:
        print("  Note: Converted DTensor state dict to regular Tensor format")
    if has_incomplete_shard:
        print("  WARNING: Some tensors may be incomplete shards from FSDP!")
        print("  WARNING: The checkpoint may have been saved incorrectly with distributed training.")
        print("  WARNING: Consider re-training or using the correct checkpoint loading method.")
    
    return converted


def load_checkpoint(
    ckpt_path: str,
    config: Optional[Dict[str, Any]] = None,
    device: str = 'cpu',
    with_decoder: bool = False,
    decoder_path: Optional[str] = None,
) -> Tuple[DynamicsWorldModel, Dict[str, Any]]:
    """
    Load model from standard .pt checkpoint (new format).
    
    Args:
        ckpt_path: Path to .pt checkpoint file
        config: Optional config dict. If None, will try to load from checkpoint or config.yaml
        device: Device to load model to
        with_decoder: Whether to load external decoder
        decoder_path: Path to external decoder checkpoint
    
    Returns:
        (model, config) tuple
    """
    print(f"Loading checkpoint from {ckpt_path}")
    
    # Load checkpoint
    checkpoint = load_checkpoint_payload(ckpt_path, map_location='cpu')
    
    # Get config
    if config is None:
        config = checkpoint.get('config')
        if config is None:
            config = load_config_from_ckpt(ckpt_path)
        if config is None:
            raise ValueError(f"Config not found in checkpoint or {ckpt_path}")
    
    # Create model
    model = create_world_model_from_config(config)
    
    # Load state dict (with DTensor conversion if needed)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        raise ValueError("Checkpoint does not contain 'model_state_dict' or 'state_dict'")
    
    # Check if state dict contains DTensor and convert if needed
    sample_value = next(iter(state_dict.values()), None)
    if sample_value is not None and (
        type(sample_value).__name__ == 'DTensor' or hasattr(sample_value, '_local_tensor')
    ):
        print("  Detected DTensor format in checkpoint, converting to regular Tensor...")
        state_dict = _convert_dtensor_state_dict(state_dict)
    
    incompatible = model.load_state_dict(state_dict, strict=False)
    known_legacy_auxiliary_keys = {
        "policy_head.layers.4.1.gamma",
        "value_head.layers.4.1.gamma",
    }
    allowed_unexpected = [
        key
        for key in incompatible.unexpected_keys
        if key in known_legacy_auxiliary_keys
    ]
    critical_unexpected = [
        key
        for key in incompatible.unexpected_keys
        if key not in allowed_unexpected
    ]
    if incompatible.missing_keys or critical_unexpected:
        missing_preview = incompatible.missing_keys[:20]
        unexpected_preview = critical_unexpected[:20]
        raise RuntimeError(
            "World-model checkpoint is not architecture-compatible. "
            f"Missing keys ({len(incompatible.missing_keys)}): {missing_preview}; "
            f"unexpected keys ({len(critical_unexpected)}): "
            f"{unexpected_preview}"
        )
    if allowed_unexpected:
        print(
            "  Ignoring known legacy auxiliary-head parameters: "
            f"{allowed_unexpected}"
        )
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}, "
          f"global_step {checkpoint.get('global_step', 'unknown')}")
    
    # Load external decoder if requested
    if with_decoder and decoder_path and os.path.exists(decoder_path):
        model.active_decoder = load_decoder_from_ckpt(decoder_path, device)
        print(f"Loaded external decoder from {decoder_path}")
    
    model.to(device)
    model.eval()
    
    return model, config


def load_fsdp_checkpoint(
    ckpt_path: str,
    device: str = 'cpu',
    load_external_decoder: bool = True,
) -> Tuple[DynamicsWorldModel, Dict[str, Any]]:
    """
    Load FSDP checkpoint and return a clean, unwrapped model.
    This is for compatibility with older checkpoints saved with accelerate FSDP.
    
    Args:
        ckpt_path: Path to checkpoint directory
        device: Device to load to ('cpu' recommended for re-training with accelerator)
        load_external_decoder: Whether to load external decoder from config
    
    Returns:
        (model, config) tuple
    """
    config = load_config_from_ckpt(ckpt_path)
    if config is None:
        raise ValueError(f"Config not found in {ckpt_path} or parent")
    
    print("Creating model...")
    model = create_world_model_from_config(config)
    
    fsdp_path = os.path.join(ckpt_path, "pytorch_model_fsdp_0")
    
    if os.path.exists(fsdp_path):
        print(f"Found FSDP checkpoint at {fsdp_path}")
        print("Loading using torch.distributed.checkpoint...")
        
        import torch.distributed.checkpoint as dcp
        
        # Load to CPU first to avoid device conflicts
        model.cpu()
        
        # Create a state_dict with the structure accelerate uses
        state_dict = {"model": model.state_dict()}
        
        try:
            dcp.load(
                state_dict=state_dict,
                checkpoint_id=fsdp_path,
            )
            model.load_state_dict(state_dict["model"], strict=False)
            print("FSDP checkpoint loaded successfully")
            
        except Exception as e:
            print(f"Error loading FSDP checkpoint: {e}")
            raise e
            
    else:
        print("FSDP folder not found, attempting standard accelerate load...")
        from accelerate import Accelerator
        
        # Create temporary accelerator just for loading
        accelerator = Accelerator(cpu=True)
        wrapped_model = accelerator.prepare(model)
        
        try:
            accelerator.load_state(ckpt_path)
            print("Checkpoint loaded successfully")
            model = accelerator.unwrap_model(wrapped_model)
            
        except Exception as e:
            print(f"Error loading state with accelerator: {e}")
            raise e
    
    # Load external decoder if configured
    if load_external_decoder:
        training_config = config.get('training', {})
        external_decoder_path = training_config.get('external_decoder_path')
        if external_decoder_path and os.path.exists(external_decoder_path):
            print(f"Loading external decoder from {external_decoder_path}")
            model.active_decoder = load_decoder_from_ckpt(external_decoder_path, device)
            print("External decoder loaded successfully")
        elif external_decoder_path:
            print(f"Warning: External decoder path specified but not found: {external_decoder_path}")
    
    model.to(device)
    model.eval()
    
    print(f"Returning clean, unwrapped model on {device}")
    return model, config


def auto_load_checkpoint(
    ckpt_path: str,
    device: str = 'cpu',
    config: Optional[Dict[str, Any]] = None,
    with_decoder: bool = True,
    decoder_path: Optional[str] = None,
) -> Tuple[DynamicsWorldModel, Dict[str, Any]]:
    """
    Automatically detect checkpoint format and load accordingly.
    
    Args:
        ckpt_path: Path to checkpoint (file or directory)
        device: Device to load to
        config: Optional config override
        with_decoder: Whether to load decoder
        decoder_path: Optional explicit decoder path
    
    Returns:
        (model, config) tuple
    """
    # Detect format: .pt file = new format, directory = FSDP format
    if ckpt_path.endswith('.pt') and os.path.isfile(ckpt_path):
        print(f"Detected standard .pt checkpoint format")
        return load_checkpoint(
            ckpt_path=ckpt_path,
            config=config,
            device=device,
            with_decoder=with_decoder,
            decoder_path=decoder_path,
        )
    elif os.path.isdir(ckpt_path):
        print(f"Detected FSDP checkpoint format")
        return load_fsdp_checkpoint(
            ckpt_path=ckpt_path,
            device=device,
            load_external_decoder=with_decoder,
        )
    else:
        raise ValueError(f"Unknown checkpoint format: {ckpt_path}")
