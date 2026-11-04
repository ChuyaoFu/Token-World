#!/usr/bin/env python3
"""Per-frame paired image/crop evidence on immutable decoder probes."""
import argparse,json,sys,os
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from decoder_improvement.models import load_decoder
from tokenworld.decoder.metrics.lpips.modules.lpips import LPIPS
from diagnose import sha


def resized(x):return F.interpolate(x[None].float(),size=(224,224),mode='bilinear',align_corners=False)[0]
def image(x):return Image.fromarray(((x.detach().float().cpu().clamp(-1,1)+1)*127.5).round().byte().permute(1,2,0).numpy())
def edge(x):
    return (x[...,1:,:]-x[...,:-1,:],x[...,:,1:]-x[...,:,:-1])

def metrics(pred,gt,net):
    assert pred.shape==gt.shape and pred.shape==(3,224,224)
    assert torch.isfinite(pred).all() and torch.isfinite(gt).all()
    def calc(a,b):
        mse=((a-b)/2).square().mean();ea,eb=edge(a),edge(b)
        return {'mse_01':mse.item(),'psnr':float(-10*mse.clamp_min(1e-12).log10()),'lpips':net(a[None].float(),b[None].float()).item(),'edge_l1':sum((x-y).abs().mean().item() for x,y in zip(ea,eb))/2,'color_mae_01':((a-b)/2).abs().mean().item()}
    return calc(pred,gt)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--probes',type=Path,required=True);p.add_argument('--decoder',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--roi-json',type=Path);p.add_argument('--layout-controls',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);torch.cuda.set_device(a.device);a.output.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('TORCH_HOME',str(ROOT/'outputs/pretrained/torch'))
    probes=torch.load(a.probes,map_location='cpu',weights_only=False)
    model=load_decoder(str(a.decoder),a.device,freeze=True)
    net=LPIPS('alex','0.1').to(a.device).eval().requires_grad_(False)
    rois=json.loads(a.roi_json.read_text()) if a.roi_json else {}
    rows=[];saved=[]
    for i,c in enumerate(probes):
        if c['split']=='train':continue
        gt=resized(c['target']).to(a.device);views={'GT':gt};row={k:v for k,v in c.items() if not torch.is_tensor(v)};row['metrics']={}
        for key in ['raw','reencoded','cached','wm']:
            if key not in c:continue
            with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(c[key][None].to(a.device))[0][0].float()
            views[key]=pred;row['metrics'][key]=metrics(pred,gt,net)
        if a.layout_controls:
            # Negative control only: transposed spatial ordering, no training.
            shuffled=c['raw'].reshape(9,12,2560).transpose(0,1).reshape(108,2560)
            with torch.autocast('cuda',dtype=torch.bfloat16):bad=model(shuffled[None].to(a.device))[0][0].float()
            row['metrics']['transposed_negative_control']=metrics(bad,gt,net)
        if 'historical_wm_rgb' in c:
            views['historical_WM']=resized(c['historical_wm_rgb']).to(a.device)
            row['metrics']['historical_WM']=metrics(views['historical_WM'],gt,net)
        rid=f"{c.get('dataset','seed')}_{c.get('episode_index',c.get('seed'))}_{c['frame']}"
        roi=rois.get(rid,c.get('roi_xyxy_224'))
        if roi:
            x,y,x1,y1=map(int,roi);assert 0<=x<x1<=224 and 0<=y<y1<=224
            row['roi_xyxy_224']=[x,y,x1,y1]
            cuts={k:v[:,y:y1,x:x1] for k,v in views.items()}
            # LPIPS needs enough spatial support: resize identical fixed crop to 128.
            cutgt=F.interpolate(cuts['GT'][None],size=(128,128),mode='bilinear',align_corners=False)[0]
            for k,v in cuts.items():
                if k=='GT':continue
                cv=F.interpolate(v[None],size=(128,128),mode='bilinear',align_corners=False)[0]
                m=row['metrics'][k]
                m['roi_mse_01']=((v-cuts['GT'])/2).square().mean().item()
                m['roi_lpips_128']=net(cv[None],cutgt[None]).item()
                # GT-defined red foreground diagnostic; only meaningful for the red stapler.
                if c.get('seed')==100014:
                    mask=lambda q:(q[0]>.05)&(q[0]-q[1]>.25)&(q[0]-q[2]>.25)
                    ma,mb=mask(v),mask(cuts['GT']);union=(ma|mb).sum().item()
                    m['roi_red_iou']=float((ma&mb).sum())/union if union else None
                    m['roi_red_area_ratio']=float(ma.sum())/max(1,int(mb.sum()))
        width=len(views)*228;h=492 if roi else 260
        sheet=Image.new('RGB',(width,h),'white');draw=ImageDraw.Draw(sheet)
        for j,(k,v) in enumerate(views.items()):
            draw.text((j*228+3,3),k,fill='black');sheet.paste(image(v),(j*228,25))
            if roi:
                x,y,x1,y1=roi;draw.rectangle((j*228+x,25+y,j*228+x1,25+y1),outline='cyan',width=1)
                sheet.paste(image(v[:,y:y1,x:x1]).resize((224,224),Image.Resampling.NEAREST),(j*228,267))
        sheet.save(a.output/f'{rid}.png');rows.append(row)
        saved.append({'id':rid,'views':{k:v.detach().cpu().bfloat16() for k,v in views.items()}})
        print('scored',rid,flush=True)
    groups={}
    for split in sorted({r['split'] for r in rows}):
        keys=sorted({k for r in rows if r['split']==split for k in r['metrics']})
        groups[split]={}
        for key in keys:
            rr=[r['metrics'][key] for r in rows if r['split']==split and key in r['metrics']]
            fields=sorted({k for r in rr for k,v in r.items() if v is not None})
            groups[split][key]={'frames':len(rr),**{f:float(np.mean([r[f] for r in rr if r.get(f) is not None])) for f in fields}}
    weights=ROOT/'outputs/pretrained/torch/hub/checkpoints'
    report={'decoder':str(a.decoder),'decoder_sha256':sha(a.decoder),'probes_sha256':sha(a.probes),'metric_protocol':{'resolution':[224,224],'resize':'bilinear align_corners=False, no crop on full-image score','range':'[-1,1] LPIPS; [0,1] PSNR data_range=1','lpips':'bundled LPIPS alex v0.1; torchvision pretrained AlexNet; per-frame forward then arithmetic mean','weights':{p.name:sha(p) for p in weights.glob('*.pth') if 'alex' in p.name},'aggregation':'equal weight per frame, separated val and known_seed; never train','crop':'fixed GT-selected xyxy, identical positions for every candidate; LPIPS crops resized bilinear 128x128','edge':'mean absolute error of horizontal and vertical first differences in [-1,1]'},'mean':groups,'rows':rows}
    (a.output/'report.json').write_text(json.dumps(report,indent=2));torch.save(saved,a.output/'predictions.pt');print(json.dumps(groups),flush=True)
if __name__=='__main__':main()
