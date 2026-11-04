#!/usr/bin/env python3
"""Separate Qwen weights, PIL-vs-tensor resize, and cached latent precision."""
import argparse,json,sys,gc
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'scripts/table1_robotwin')]
import evaluate as ev
from tokenworld.evaluation import closed_loop as cl
from tokenworld.codec.train import FrozenQwenVision
from tokenworld.codec.common import load_codec_checkpoint,resolve_video_paths
from tokenworld.data.video import _decode_selected_frames


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--device',default='cuda:1');a=p.parse_args();torch.set_num_threads(4);torch.cuda.set_device(a.device)
    manifest=json.loads((a.run/'pilot/manifest.json').read_text());cache=Path(manifest['arguments']['cache']);raw=Path(manifest['arguments']['data_root']);spec=manifest['cache_spec']
    records=json.loads((cache/'index.json').read_text())['records'];ri,r=ev.selected_reference_records({'episodes':[dict(dataset='robotwin_fail',episode_index=1954,task='put_object_cabinet',outcome='failure')]},records,1)[0]
    payload=ev._torch_load(cache/r['path']);ids=[1,91,181,271];sourceids=[int(payload['frame_indices'][i]) for i in ids]
    pil=[x[0] for x in _decode_selected_frames(resolve_video_paths(raw,r,['observation.images.cam_high']),sourceids,tuple(spec['image_size']))]
    tensor=ev.to_pil(torch.cat([ev.raw_cam_high_rgb(ev.RawCacheDatasetAdapter(spec,raw),r,payload,i,i+1) for i in ids]))
    policy,_,_=cl.load_policy(Path(manifest['arguments']['policy_bundle']),torch.device(a.device))
    fpolicy=ev.rgb_qwen_merger_features(policy,tensor,4)[:,0]
    inputs=policy.qwen_vl_interface.build_qwenvl_inputs(images=[[im] for im in tensor],instructions=['']*len(tensor))
    del policy;gc.collect();torch.cuda.empty_cache()
    teacher=FrozenQwenVision(ROOT.parent/'IRASim/pretrained_models/Qwen3-VL-4B-Instruct',torch.device(a.device))
    ftensor=teacher(tensor);fpil=teacher(pil)
    codec,_=load_codec_checkpoint(Path(manifest['arguments']['codec']),device=torch.device(a.device));codec.eval()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        z=codec.encode(list(fpil));zt=codec.encode(list(ftensor))
    cached=payload['latents']['symmetric_qwen'][ids,0].float().to(a.device)
    def err(x,y):return dict(mse=(x.float()-y.float()).square().mean().item(),max_abs=(x.float()-y.float()).abs().max().item(),cosine=torch.nn.functional.cosine_similarity(x.float(),y.float(),dim=-1).mean().item())
    result=dict(dataset=r['dataset'],episode_index=r['episode_index'],frames=ids,policy_vs_base_visual='Output parity measured below on identical input pixels',grid_thw=inputs['image_grid_thw'].tolist(),policy_vs_teacher_same_pixels=err(fpolicy,ftensor),pil_vs_tensor_resize_merger=err(fpil,ftensor),cached_vs_pil_reencoded_latent=err(cached,z),cached_vs_tensor_reencoded_latent=err(cached,zt),historical_export='PyAV -> RGB PIL resize BILINEAR 384x288 -> Qwen image_processor',table1_path='decord -> tensor bilinear antialias384x288 -> uint8 PIL -> policy image processor')
    (a.run/'preprocessing_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
