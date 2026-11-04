"""One fresh RoboTwin scene. The parent receives only the reset observation and terminal flags."""
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
    parser.add_argument('--live-rgb', action='store_true',
                        help='Return a fresh simulated head-camera frame after a nonterminal action chunk')
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
        obs = task.get_obs()
        rgb = obs['observation']['head_camera']['rgb']
        encoded = io.BytesIO()
        Image.fromarray(rgb).save(encoded, format='PNG')
        qpos = np.asarray(obs['joint_action']['vector'], dtype=np.float32)
        assert qpos.shape == (14,) and np.isfinite(qpos).all()
        def pose(actor):
            return np.concatenate([actor.get_pose().p, actor.get_pose().q]).tolist()
        physical = dict(entities=sorted([(a.name, pose(a)) for a in task.scene.get_all_actors()]),
                        articulations=sorted([(a.name, pose(a), a.get_qpos().tolist(), a.get_qvel().tolist()) for a in task.scene.get_all_articulations()]),
                        task_assets={k:str(getattr(task,k)) for k in ('model_name','model_id','selected_modelname','selected_model_id','bottle_id') if hasattr(task,k)},
                        qpos=qpos.tolist())
        channel.write(json.dumps(dict(image=base64.b64encode(encoded.getvalue()).decode(), qpos=qpos.tolist(),
                                physics=physical, limit=int(task.step_lim), timestep=float(task.scene.get_timestep()),
                                rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()))+'\n')
        for line in sys.stdin:
            actions=np.asarray(json.loads(line), dtype=np.float32)
            assert actions.ndim==2 and actions.shape[1]==14 and 0<len(actions)<=50 and np.isfinite(actions).all()
            consumed=0
            for action in actions:
                task.take_action(action, action_type='qpos')
                consumed+=1
                if task.eval_success or task.take_action_cnt>=task.step_lim:
                    break
            reply = dict(consumed=consumed, steps=int(task.take_action_cnt), success=bool(task.eval_success))
            if args.live_rgb and not task.eval_success and task.take_action_cnt < task.step_lim:
                frame = task.get_obs()['observation']['head_camera']['rgb']
                buffer = io.BytesIO()
                Image.fromarray(frame).save(buffer, format='PNG')
                reply['image'] = base64.b64encode(buffer.getvalue()).decode()
            channel.write(json.dumps(reply)+'\n')
            if task.eval_success or task.take_action_cnt>=task.step_lim:
                break
    finally:
        task.close_env()
if __name__=='__main__':
    main()
