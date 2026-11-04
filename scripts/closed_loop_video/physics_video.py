"""Execute WM actions; write simulation RGB locally without sending visual feedback."""
import argparse
import base64
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import random
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--sim-root', type=Path, required=True)
    parser.add_argument('--frames', type=Path, required=True)
    parser.add_argument('--max-steps', type=int, default=0, help='Diagnostic only; zero uses native task limit')
    parser.add_argument('--reset-only', action='store_true', help='CPU preparation check; no policy or action')
    args = parser.parse_args()
    channel = os.fdopen(os.dup(1), 'w', buffering=1)
    os.dup2(2, 1)  # Route native renderer and planner output away from the JSON stream.
    root = args.sim_root.resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))
    import numpy as np
    from PIL import Image
    import yaml
    module = importlib.import_module('envs.' + args.task)
    task = getattr(module, args.task)()
    config = yaml.safe_load(Path('task_config/demo_clean.yml').read_text())
    body = './assets/embodiments/aloha-agilex/'
    robot = yaml.safe_load((Path(body) / 'config.yml').read_text())
    robot['planner'] = 'mplib_RRT'
    config.update(task_name=args.task, left_robot_file=body, right_robot_file=body,
                  left_embodiment_config=robot, right_embodiment_config=robot, dual_arm=True, dual_arm_embodied=True,
                  eval_mode=True, eval_video_log=False, save_data=False)
    random.seed(args.seed)
    from envs._base_task import UnStableError
    try:
        task.setup_demo(now_ep_num=0, seed=args.seed, is_test=True, **config)
    except UnStableError:
        task.close_env()
        channel.write(json.dumps(dict(invalid_reset=True, task=args.task, seed=args.seed))+'\n')
        return
    try:
        if args.task == 'put_object_cabinet':
            # Native evaluation normally retains these fields from the same-seed expert reset.
            pose = task.object.get_pose().p
            task.arm_tag = module.ArmTag('right' if pose[0] > 0 else 'left')
            task.origin_z = float(pose[2])
        elif args.task == 'open_laptop':
            # Same deterministic fields set by play_once, without executing expert actions.
            face_prod = module.get_face_prod(task.laptop.get_pose().q, [1, 0, 0], [1, 0, 0])
            task.arm_tag = module.ArmTag('left' if face_prod > 0 else 'right')
        elif args.task == 'place_object_scale':
            task.arm_tag = module.ArmTag('right' if task.object.get_pose().p[0] > 0 else 'left')
        if args.reset_only:
            # The preflight needs the physical reset, head renderer and success
            # predicate, without collecting unneeded wrist-camera observations.
            initial_success = bool(task.check_success())
            task._update_render()
            head = task.cameras.static_camera_list[task.cameras.static_camera_name.index('head_camera')]
            head.take_picture()
            color = head.get_picture('Color')
            assert color.ndim == 3 and color.shape[-1] == 4 and np.isfinite(color).all()
            joints = task.robot.get_left_arm_jointState() + task.robot.get_right_arm_jointState()
            assert np.asarray(joints).shape == (14,) and np.isfinite(joints).all()
            channel.write(json.dumps(dict(task=args.task, seed=args.seed, limit=int(task.step_lim),
                                          initial_success=initial_success, head_render_checked=True))+'\n')
            return
        obs = task.get_obs()
        rgb = obs['observation']['head_camera']['rgb']
        encoded = io.BytesIO()
        Image.fromarray(rgb).save(encoded, format='PNG')
        qpos = np.asarray(obs['joint_action']['vector'], dtype=np.float32)
        assert qpos.shape == (14,) and np.isfinite(qpos).all()
        # Match WM RGB export resolution; frame ids refer to the 50 Hz command clock.
        def small(frame):
            return np.asarray(Image.fromarray(frame).resize((224, 224), Image.Resampling.BILINEAR))
        head = task.cameras.static_camera_list[task.cameras.static_camera_name.index('head_camera')]
        def head_frame():
            # Render only the head camera after initialization. Other get_obs fields
            # are not consumed by either policy or WM during the physical replay.
            task._update_render()
            head.take_picture()
            return (head.get_picture('Color')[:, :, :3] * 255).clip(0, 255).astype('uint8')
        frames, frame_ids = [small(rgb)], [0]
        limit = min(int(task.step_lim), args.max_steps) if args.max_steps else int(task.step_lim)
        def pose(actor):
            return np.concatenate([actor.get_pose().p, actor.get_pose().q]).tolist()
        physical = dict(entities=sorted([(a.name, pose(a)) for a in task.scene.get_all_actors()]),
                        articulations=sorted([(a.name, pose(a), a.get_qpos().tolist(), a.get_qvel().tolist()) for a in task.scene.get_all_articulations()]),
                        task_assets={k:str(getattr(task,k)) for k in ('model_name','model_id','selected_modelname','selected_model_id','bottle_id') if hasattr(task,k)},
                        qpos=qpos.tolist())
        channel.write(json.dumps(dict(image=base64.b64encode(encoded.getvalue()).decode(), qpos=qpos.tolist(),
                                physics=physical, limit=limit, native_limit=int(task.step_lim), timestep=float(task.scene.get_timestep()),
                                rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()))+'\n')
        for line in sys.stdin:
            actions=np.asarray(json.loads(line), dtype=np.float32)
            assert actions.ndim==2 and actions.shape[1]==14 and 0<len(actions)<=50 and np.isfinite(actions).all()
            consumed=0
            for action in actions:
                task.take_action(action, action_type='qpos')
                consumed+=1
                if task.take_action_cnt % 2 == 0:
                    frames.append(small(head_frame()))
                    frame_ids.append(int(task.take_action_cnt))
                if task.eval_success or task.take_action_cnt>=limit:
                    break
            reply = dict(consumed=consumed, steps=int(task.take_action_cnt), success=bool(task.eval_success))
            channel.write(json.dumps(reply)+'\n')
            if task.eval_success or task.take_action_cnt>=limit:
                break
        args.frames.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.frames.with_suffix('.tmp.npz')
        np.savez_compressed(temporary, rgb=np.stack(frames), frame_ids=np.asarray(frame_ids))
        temporary.replace(args.frames)
    finally:
        task.close_env()
if __name__=='__main__':
    main()
