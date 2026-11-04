"""Evaluate the adapter with QwenOFT's native action readout on the saved paired validation set."""
import argparse
import json
from pathlib import Path
import torch
from PIL import Image
from evaluate_rgb_bridge_similarity import measure, sha
from tokenworld.codec.common import load_codec_checkpoint
from tokenworld.codec.deepstack_adapter import DeepStackAdapter, inject_actions
from tokenworld.data.dataset import RobotwinSVAETokenDataset
from tokenworld.evaluation import closed_loop as cl


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    root = args.run_dir
    config = json.loads((root / 'run.json').read_text())['contract']
    episodes = [e for e in json.loads((root / 'manifest.json').read_text())['episodes'] if e['split'] == 'val']
    refs = {}
    for path in sorted((root / 'rgb_bridge_similarity').glob('action_outputs.rank*.pt')):
        for row in torch.load(path, weights_only=True):
            key = (row['dataset'], row['episode_index'], row['frame'])
            assert key not in refs
            refs[key] = row
    assert len(refs) == 2000
    checkpoint = root / 'adapter_best.pt'
    digest = sha(checkpoint)
    assert digest == json.loads((root / 'training_complete.json').read_text())['best_sha256']
    policy, _, _ = cl.load_policy(Path(config['policy_bundle']), device)
    codec, _ = load_codec_checkpoint(Path(config['codec_checkpoint']), device=device)
    codec.eval().requires_grad_(False)
    adapter = DeepStackAdapter().to(device)
    adapter.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'])
    adapter.eval().requires_grad_(False)
    ds = RobotwinSVAETokenDataset(config['cache'], raw_data_path=config['raw_data_root'],
                                  split='val', num_hist=1, num_pred=1, views=[0])
    indices = {(r['dataset'], r['episode_index']): i for i, r in enumerate(ds.records)}
    stats = json.loads((Path(config['policy_bundle']) / 'model/dataset_statistics.json').read_text())
    rows = []
    parity = []
    for entry in episodes:
        task = entry['task']
        index = indices[(entry['dataset'], entry['episode_index'])]
        payload = ds._load_episode(index)
        template = cl.build_policy_template(policy, task, 'success')
        first = entry['frames'][0]
        raw = ds._load_raw_images(index, ds.records[index], [first])[0, 0]
        image = Image.fromarray(raw.numpy()).convert('RGB').resize((384, 288), Image.Resampling.BILINEAR)
        inputs = policy.qwen_vl_interface.build_qwenvl_inputs(images=[[image]], instructions=[cl.action_instruction(policy, task, 'success')])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            merger, deep = policy.qwen_vl_interface.model.model.get_image_features(inputs['pixel_values'].to(device), inputs['image_grid_thw'].to(device))
        oracle = inject_actions(policy, template, torch.cat(list(merger)), deep, readout='native_pre_norm').cpu()[0]
        key = (entry['dataset'], entry['episode_index'])
        err = (oracle - refs[(*key, first)]['predict_action_teacher']).abs().max().item()
        assert err < 1e-5, (key, err)
        parity.append(err)
        for frame in entry['frames']:
            ref = refs.pop((*key, frame))
            latent = payload['latents']['symmetric_qwen'][frame, :1].to(device).float()
            merger = codec.decode(latent, [108])[0].reshape(1, 108, 2560).bfloat16().float()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                deep = adapter.features(latent.bfloat16().float(), merger)[0]
            action = inject_actions(policy, template, merger, deep, readout='native_pre_norm').cpu()[0]
            assert torch.isfinite(action).all()
            rows.append(dict(task=task, dataset=key[0], episode_index=key[1], frame=frame,
                             metrics=measure(action, ref['predict_action_teacher'], stats)))
        if len(parity) == 1 or len(parity) % 50 == 0:
            print(f'queries={len(rows)}, RGB injection parity max abs={max(parity)}', flush=True)
    assert not refs and len(rows) == 2000
    result = dict(rows=rows, checkpoint_sha256=digest, native_parity_frames=len(parity), native_parity_max_abs=max(parity))
    (args.output / 'validation.json').write_text(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
