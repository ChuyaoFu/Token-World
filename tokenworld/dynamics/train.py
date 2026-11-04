# train_dreamer_vlm.py
"""Train TokenWorld on RoboTwin sequence caches. Run as a Python module."""
import sys
import os
import argparse
import yaml
from datetime import datetime

from tokenworld.dynamics.model import DynamicsWorldModel
from tokenworld.dynamics.trainer import BehaviorCloneTrainer
from tokenworld.dynamics.task_tokenizer import TaskTokenizer

# RobotWin dataset imports
from tokenworld.data.dataset import load_robotwin_svae_train_val


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_args_to_config(config: dict, args: argparse.Namespace) -> dict:
    """Override config values with command line arguments if provided."""
    # Override training params if specified
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.num_epochs is not None:
        config['training']['num_epochs'] = args.num_epochs
    if args.learning_rate is not None:
        config['training']['learning_rate'] = args.learning_rate
    if args.checkpoint_folder is not None:
        config['training']['checkpoint_folder'] = args.checkpoint_folder
    if args.resume_from is not None:
        config['training']['resume_from'] = args.resume_from
    
    # Override data params if specified
    if args.data_path is not None:
        config['data']['data_path'] = args.data_path
    if args.frameskip is not None:
        config['data']['frameskip'] = args.frameskip
    if args.addskip is not None:
        config['data']['addskip'] = args.addskip
    
    return config

def _save_config(config, output_dir: str):
        """
        Save training configuration to YAML file.
        """
        config_path = os.path.join(output_dir, 'config.yaml')
        
        os.makedirs(output_dir, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"Saved config to {config_path}")

def main():
    parser = argparse.ArgumentParser(description='Train Dreamer4 VLM World Model')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML configuration file')
    
    # Optional overrides
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_epochs', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--checkpoint_folder', type=str, default=f'./checkpoints/{datetime.now().strftime("%Y%m%d_%H%M")}')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--frameskip', type=int, default=None)
    parser.add_argument('--addskip', type=int, default=None)
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint file for resuming training (e.g., ./checkpoints/checkpoint_epoch_0010.pt)')
    
    args = parser.parse_args()
    
    # Load and merge config
    config = load_config(args.config)
    config = merge_args_to_config(config, args)
    _save_config(config=config, output_dir=args.checkpoint_folder)
    
    # Extract configurations
    data_cfg = config['data']
    model_cfg = config['model']
    decoder_cfg = config['decoder']
    training_cfg = config['training']
    wandb_cfg = config.get('wandb', {})
    
    frameskip = data_cfg['frameskip']
    addskip = data_cfg.get('addskip', 0)
    dataset_type = data_cfg.get('dataset_type', 'robotwin_svae')
    
    # Get resume_from (command line arg takes precedence over config)
    resume_from = training_cfg.get('resume_from', None)
    
    print("=" * 60)
    print("Dreamer4 VLM Training")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Dataset type: {dataset_type}")
    print(f"Frameskip: {frameskip}")
    print(f"Addskip: {addskip}")
    print(f"Batch size: {training_cfg['batch_size']}")
    print(f"Mixed precision: {training_cfg.get('mixed_precision', 'no')}")
    print(f"DataLoader workers: {training_cfg.get('num_workers', 0)}")
    print(f"Num epochs: {training_cfg['num_epochs']}")
    if training_cfg.get('max_train_steps') is not None:
        print(f"Max train steps: {training_cfg['max_train_steps']}")
    if training_cfg.get('checkpoint_every_steps') is not None:
        print(
            "Checkpoint every optimizer steps: "
            f"{training_cfg['checkpoint_every_steps']}"
        )
    if resume_from:
        print(f"Resuming from: {resume_from}")
    print("=" * 60)
    
    # 1. Load dataset based on type
    print("\n1. Loading dataset...")
    
    if dataset_type == 'robotwin_svae':
        if frameskip != 1 or addskip != 0:
            raise ValueError(
                "robotwin_svae samples temporal spacing before loading actions; "
                "set frameskip=1 and addskip=0, then use the sequence interval/"
                "random stride options instead"
            )

        tasks = data_cfg.get('tasks', None)
        outcomes = data_cfg.get('outcomes', None)
        views = data_cfg.get('views', None)
        print("   Loading native RoboWin latent token cache...")
        print(f"   Tasks: {tasks if tasks else 'all'}")
        print(f"   Outcomes: {outcomes if outcomes else 'all'}")
        print(f"   Views: {views if views is not None else 'all'}")

        datasets, traj_dset = load_robotwin_svae_train_val(
            cache_dir=data_cfg['data_path'],
            tasks=tasks,
            outcomes=outcomes,
            views=views,
            latent_method=data_cfg.get('latent_method', 'symmetric_qwen'),
            normalize_latents=data_cfg.get('normalize_latents', True),
            latent_stats_scope=data_cfg.get('latent_stats_scope', 'per_view'),
            normalize_action=data_cfg.get('normalize_action', True),
            verify_action_equals_state=data_cfg.get('verify_action_equals_state', True),
            num_hist=data_cfg['num_hist'],
            num_pred=data_cfg['num_pred'],
            random_window_per_episode=data_cfg.get('random_window_per_episode', True),
            random_stride_min=data_cfg.get('random_stride_min', 1),
            random_stride_max=data_cfg.get('random_stride_max', 8),
            train_sequence_interval=data_cfg.get('train_sequence_interval', 1),
            train_window_stride=data_cfg.get('train_window_stride', 1),
            val_sequence_interval=data_cfg.get('val_sequence_interval', 1),
            val_window_stride=data_cfg.get('val_window_stride'),
            tail_align=data_cfg.get('tail_align', True),
            lru_episodes=data_cfg.get('lru_episodes', 8),
            action_schema=data_cfg.get('action_schema', 'robotwin14'),
        )
        train_dataset = datasets['train']
        val_dataset = datasets['valid']

        dataset_contract = {
            'dim_latent': train_dataset.latent_channels,
            'num_latent_tokens': train_dataset.tokens_per_view,
            'num_video_views': train_dataset.num_views,
        }
        for config_key, actual_value in dataset_contract.items():
            configured_value = int(model_cfg[config_key])
            if configured_value != actual_value:
                raise ValueError(
                    f"Latent cache/model mismatch for {config_key}: "
                    f"cache={actual_value}, model={configured_value}"
                )

        configured_grid = model_cfg.get('spatial_grid_size')
        if configured_grid is not None:
            configured_grid = tuple(map(int, configured_grid))
            cache_grid = (train_dataset.grid_height, train_dataset.grid_width)
            if configured_grid != cache_grid:
                raise ValueError(
                    "Latent cache/model mismatch for spatial_grid_size: "
                    f"cache={list(cache_grid)}, model={list(configured_grid)}"
                )

        print(
            "   Native token shape: "
            f"[{train_dataset.num_views}, {train_dataset.tokens_per_view}, "
            f"{train_dataset.latent_channels}]"
        )
    else:
        raise ValueError("This release accepts the robotwin_svae cache interface for S-VAE and raw Qwen features.")

    print(f"   Train samples: {len(train_dataset)}, Val samples: {len(val_dataset) if val_dataset else 0}")

    # Initialize task tokenizer
    task_tokenizer = None
    task_tokenizer_cfg = model_cfg.get('task_tokenizer', {})
    if task_tokenizer_cfg.get('enabled', False):
        task_tokenizer = TaskTokenizer(
            **task_tokenizer_cfg
        )
        print(f"   Task Tokenizer loaded, vocab_size={task_tokenizer.vocab_size}")
        task_prompt = data_cfg.get('task_prompt', None)
        print(f"   Task Prompt: {task_prompt}")
    else:
        print("   Task Tokenizer disabled")
    
    
    if task_tokenizer is not None: # if task tokenizer is enabled, then use task cross attention
        task_cross_attn_cfg = model_cfg.get('task_cross_attn', {})
        has_task_tokens = task_cross_attn_cfg.get('enabled', False)
        task_cross_attn_every = task_cross_attn_cfg.get('cross_attn_every', 1)
    else:
        has_task_tokens = False
        task_cross_attn_every = 1

    # 3. Initialize Dreamer4 World Model
    print("\n2. Initializing world model...")
    
    # Handle discrete actions - convert list to tuple and multiply by skip factor
    # addskip sums actions (dim unchanged); frameskip concatenates (dim multiplied)
    original_discrete = model_cfg['original_discrete_actions']
    if isinstance(original_discrete, list):
        original_discrete = tuple(original_discrete)
    if addskip > 0:
        num_discrete_actions = original_discrete
        num_continuous_actions = model_cfg['original_continuous_actions']
    else:
        num_discrete_actions = original_discrete * frameskip
        num_continuous_actions = model_cfg['original_continuous_actions'] * frameskip
    
    print(f"   num_discrete_actions: {num_discrete_actions}")
    print(f"   num_continuous_actions: {num_continuous_actions}")
    
    world_model = DynamicsWorldModel(
        dim=model_cfg['dim'],
        dim_latent=model_cfg['dim_latent'],
        num_latent_tokens=model_cfg['num_latent_tokens'],
        num_spatial_tokens=model_cfg['num_spatial_tokens'],
        num_agents=model_cfg['num_agent_tokens'],
        num_video_views=model_cfg.get('num_video_views',1),
        spatial_grid_size=model_cfg.get('spatial_grid_size'),
        video_tokenizer=None,
        depth=model_cfg['depth'],
        time_block_every=model_cfg.get('time_block_every', 4),
        num_discrete_actions=num_discrete_actions,
        num_continuous_actions=num_continuous_actions,
        dim_proprio=model_cfg['dim_proprio'],
        max_steps=model_cfg['max_steps'],
        attn_heads=model_cfg['attn_heads'],
        attn_dim_head=model_cfg['attn_dim_head'],
        # TODO: Check why use adaln doesn't work in generate
        use_adaln=model_cfg.get('use_adaln', False),
        use_idm=model_cfg.get('use_idm', False),
        use_view_action_proj=model_cfg.get('use_view_action_proj', False),
        use_multi_view_action_embed=model_cfg.get('use_multi_view_action_embed', False),
        task_prompt=data_cfg.get('task_prompt', None),
        task_tokenizer=task_tokenizer,
        has_task_tokens=has_task_tokens,
        task_cross_attn_every=task_cross_attn_every,
        prob_no_shortcut_train=model_cfg.get('prob_no_shortcut_train', None),
        use_time_rnn=model_cfg.get('use_time_rnn', True),
        pred_orig_latent=model_cfg.get('pred_orig_latent', True),
    )
    
    # 4. Create trainer
    print("\n3. Creating trainer...")
    accelerate_kwargs = {}
    if wandb_cfg.get('enabled', False):
        accelerate_kwargs["log_with"] = "wandb"
    if training_cfg.get('mixed_precision'):
        accelerate_kwargs["mixed_precision"] = training_cfg['mixed_precision']
    
    # Handle decoder usage flag
    use_decoder = training_cfg.get('use_decoder', True)
    if not use_decoder:
        print("   use_decoder=False: decoder will NOT be loaded, RGB visualization disabled.")

    # Handle dual decoder configuration for RobotWin (view0 vs view1+view2)
    dual_decoder_cfg = training_cfg.get('dual_decoder', None)
    external_decoder_path = training_cfg.get('external_decoder_path') if use_decoder else None
    external_decoder_paths = None
    
    if use_decoder and dual_decoder_cfg is not None and dual_decoder_cfg.get('enabled', False):
        # Dual decoder mode: different decoders for different views
        external_decoder_paths = {
            'view0': dual_decoder_cfg.get('decoder_view0_path'),
            'view1_2': dual_decoder_cfg.get('decoder_view1_2_path'),
        }
        print(f"   Using dual decoders:")
        print(f"     View0 decoder: {external_decoder_paths['view0']}")
        print(f"     View1+2 decoder: {external_decoder_paths['view1_2']}")
        external_decoder_path = None  # Disable single decoder when using dual
    
    # NEW: Get training mode from config
    training_mode = training_cfg.get('training_mode', 'world_model')
    print(f"   Training mode: {training_mode}")
    if training_mode == 'policy':
        print("   Policy mode: model will predict actions from task instruction + current state")
    else:
        print("   World model mode: model will predict future state from actions + current state")
    
    trainer = BehaviorCloneTrainer(
        world_model,
        dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=training_cfg['batch_size'],
        num_workers=training_cfg.get('num_workers', 0),
        val_num_workers=training_cfg.get('val_num_workers'),
        pin_memory=training_cfg.get('pin_memory', False),
        persistent_workers=training_cfg.get('persistent_workers', False),
        prefetch_factor=training_cfg.get('prefetch_factor', 2),
        drop_last=training_cfg.get('drop_last', True),
        find_unused_parameters=training_cfg.get('find_unused_parameters', True),
        num_epochs=training_cfg['num_epochs'],
        max_train_steps=training_cfg.get('max_train_steps'),
        learning_rate=training_cfg['learning_rate'],
        # reward_loss_weight=training_cfg.get('reward_loss_weight', 1.0),
        decoder_learning_rate=training_cfg['decoder_learning_rate'],
        train_decoder=training_cfg['train_decoder'],
        external_decoder_path=external_decoder_path,
        external_decoder_paths=external_decoder_paths,  # New: dual decoder support
        checkpoint_every=training_cfg.get('checkpoint_every'),
        checkpoint_every_steps=training_cfg.get('checkpoint_every_steps'),
        checkpoint_folder=training_cfg.get(
            'checkpoint_folder', args.checkpoint_folder
        ),
        val_every=training_cfg['val_every'],
        viz_every=training_cfg.get('viz_every'),
        accelerate_kwargs=accelerate_kwargs,
        use_idm=model_cfg.get('use_idm', False),
        training_mode=training_mode,  # NEW: pass training mode
    )
    
    # 5. Initialize wandb and train
    print("\n4. Starting training...")
    if wandb_cfg.get('enabled', False):
        trainer.accelerator.init_trackers(project_name=wandb_cfg['project_name'])
    
    trainer(resume_from=resume_from)
    
    if wandb_cfg.get('enabled', False):
        trainer.accelerator.end_training()
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
