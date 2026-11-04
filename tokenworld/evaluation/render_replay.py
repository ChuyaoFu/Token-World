"""Render matched raw / codec GT / recorded-action replay / learned closed loop.
Closed-loop frames are qualitative only: actions diverge from the recorded episode.
"""
import argparse,json,os,shutil,subprocess
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image,ImageDraw,ImageFont
from tokenworld.data.dataset import load_robotwin_svae_train_val
from tokenworld.codec.decoder import load_symmetric_qwen_decoder
from tokenworld.decoder.checkpoint import load_external_decoder
from tokenworld.decoder.train import compute_metrics,lpips
from tokenworld.evaluation.closed_loop import episode_name,write_video

def parse_args():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['manifest','cache','raw-data-root','codec-checkpoint','decoder-checkpoint','open-loop-root','closed-loop-root','output-root']:
  p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--device',default='cuda:0');p.add_argument('--batch-size',type=int,default=8)
 return p.parse_args()

@torch.inference_mode()
def main():
 a=parse_args();a.output_root.mkdir(parents=True,exist_ok=True)
 ds,_=load_robotwin_svae_train_val(cache_dir=a.cache,raw_data_path=a.raw_data_root,num_hist=4,num_pred=16,views=[0],load_raw_images=True)
 ds=ds['valid'];lookup={(r['dataset'],r['episode_index']):i for i,r in enumerate(ds.records)}
 codec,_=load_symmetric_qwen_decoder(a.codec_checkpoint,a.device)
 decoder=load_external_decoder(a.decoder_checkpoint,device=a.device,freeze=True)
 def decode(z):
  output=[]
  for b in z.split(a.batch_size):
   with torch.autocast('cuda',dtype=torch.bfloat16):
    f=codec(b[:,0].to(a.device));p,_=decoder(f)
   output.append(p.float().cpu())
  return torch.cat(output)
 def pixels(x):return (((x+1)/2).clamp(0,1).permute(0,2,3,1).numpy()*255).round().astype(np.uint8)
 def metrics(p,t):
  sums={};n=0
  for lo in range(1,len(p),a.batch_size):
   pp=p[lo:lo+a.batch_size].to(a.device);tt=t[lo:lo+a.batch_size].to(a.device)
   vals=compute_metrics(pp,tt,lpips)
   for k,v in vals.items():sums[k]=sums.get(k,0)+v*len(pp)
   n+=len(pp)
  return {k:v/n for k,v in sums.items()}
 try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
 except OSError:font=ImageFont.load_default()
 reports=[]
 for entry in json.loads(a.manifest.read_text())['episodes']:
  name=episode_name(entry);print('render',name,flush=True)
  op=a.open_loop_root/'tokenworld'/name
  cl=a.closed_loop_root/'episodes'/name
  oz=torch.load(op/'predicted_svae_latents.pt',map_location='cpu',weights_only=True)
  cz=torch.load(cl/'predicted_svae_latents.pt',map_location='cpu',weights_only=True)
  assert torch.isfinite(oz).all() and torch.isfinite(cz).all()
  assert oz.shape==cz.shape and len(oz)==entry['num_frames']
  ri=lookup[(entry['dataset'],entry['episode_index'])];rec=ds.records[ri]
  z=ds._load_episode(ri)['latents']['symmetric_qwen'][:len(oz),:1]
  assert torch.equal(oz[0],z[0].float()) and torch.equal(cz[0],z[0].float())
  raw=ds._load_raw_images(ri,rec,list(range(len(z))))[:,0].permute(0,3,1,2).float()/255*2-1
  raw=F.interpolate(raw,size=(224,224),mode='bilinear',align_corners=False)
  preds={'codec_gt':decode(z),'open_loop':decode(oz),'closed_loop':decode(cz)}
  result={'episode':entry,'frames':len(oz),'metrics_exclude_initial_frame':True,'codec_gt_vs_raw':metrics(preds['codec_gt'],raw),'open_loop_vs_raw':metrics(preds['open_loop'],raw),'open_loop_vs_codec_gt':metrics(preds['open_loop'],preds['codec_gt'])}
  acts=np.load(cl/'actions_closed_loop.npy');assert np.isfinite(acts).all() and acts.shape==(len(z)-1,14)
  cm=json.loads((cl/'metadata.json').read_text());om=json.loads((op/'metadata.json').read_text());assert cm['policy_queries']==int(np.ceil((len(z)-1)/cm['query_every']))
  result['closed_loop_queries']=cm['policy_queries'];result['closed_loop_success']=None
  result['protocol']={'open_loop':{k:om[k] for k in ['execution_horizon','num_denoise_steps','checkpoint','latent_normalization']},'closed_loop':{k:cm[k] for k in ['execution_horizon','num_denoise_steps','query_every','world_model_checkpoint','policy_bundle']},'initial_context_frames':1,'policy_prompt_outcome':'success','action_feature_interface':'merger_only','codec_checkpoint':str(a.codec_checkpoint.resolve()),'decoder_checkpoint':str(a.decoder_checkpoint.resolve())}
  assert Path(cm['decoder_checkpoint']).resolve()==a.decoder_checkpoint.resolve()
  assert Path(cm['codec_checkpoint']).resolve()==a.codec_checkpoint.resolve()==Path(om['codec_checkpoint']).resolve()
  assert Path(cm['world_model_checkpoint']).resolve()==Path(om['checkpoint']).resolve()
  tracks={'Raw GT':pixels(raw),'GT latent decode':pixels(preds['codec_gt']),'Open-loop replay':pixels(preds['open_loop']),'Closed-loop (WM)':pixels(preds['closed_loop'])}
  fps=float(json.loads((a.cache/'cache_spec.json').read_text()).get('export_frame_rate_hz',50))
  video=np.concatenate(list(tracks.values()),axis=2)
  head=Image.new('RGB',(896,32),'white');draw=ImageDraw.Draw(head)
  for j,label in enumerate(tracks):draw.text((j*224+5,7),label,fill='black',font=font)
  header=np.asarray(head)
  video=np.concatenate([np.broadcast_to(header,(len(z),*header.shape)),video],axis=1)
  out=a.output_root/name;out.mkdir(exist_ok=True)
  write_video(video,out/'comparison.mp4',int(fps))
  if shutil.which('ffmpeg'):
   encoded=out/'comparison_h264.mp4'
   subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(out/'comparison.mp4'),'-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(encoded)],check=True)
   encoded.replace(out/'comparison.mp4')
  ids=np.linspace(0,len(z)-1,7).round().astype(int)
  sheet=Image.new('RGB',(160+7*224,50+4*224),'white');draw=ImageDraw.Draw(sheet)
  draw.text((5,5),f"{entry['task']} / source {entry['outcome']} | closed-loop prompt: success",fill='black',font=font)
  for j,i in enumerate(ids):draw.text((160+j*224,29),f'frame {i}',fill='black',font=font)
  for row,(label,frames) in enumerate(tracks.items()):
   draw.text((5,50+row*224+100),label,fill='black',font=font)
   for j,i in enumerate(ids):sheet.paste(Image.fromarray(frames[i]),(160+j*224,50+row*224))
  sheet.save(out/'comparison.png')
  result['video']=str((out/'comparison.mp4').resolve());result['thumbnail']=str((out/'comparison.png').resolve())
  (out/'metrics.json').write_text(json.dumps(result,indent=2));reports.append(result)
  print(json.dumps(result),flush=True)
 summary={'protocol':'Original block replay with predicted proprio feedback; per-episode parameters and checkpoint paths recorded in episodes[].protocol','scope':'diagnostic subset, not full paper/simulator evaluation','episodes':reports}
 summary['macro_over_episodes']={group:{k:float(np.mean([r[group][k] for r in reports])) for k in reports[0][group]} for group in ['codec_gt_vs_raw','open_loop_vs_raw','open_loop_vs_codec_gt']}
 (a.output_root/'summary.json').write_text(json.dumps(summary,indent=2))
 print('DONE',json.dumps(summary['macro_over_episodes']),flush=True)
if __name__=='__main__':main()
