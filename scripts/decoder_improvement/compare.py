#!/usr/bin/env python3
"""Pair reports by immutable frame identity and make original/new crop sheets."""
import argparse,json
from pathlib import Path
import torch
from PIL import Image,ImageDraw
from evaluate import image


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    b,c=[json.loads((d/'report.json').read_text()) for d in [a.baseline,a.candidate]]
    assert b['probes_sha256']==c['probes_sha256'] and b['metric_protocol']==c['metric_protocol']
    identity=lambda r:(r['split'],r.get('dataset'),r.get('episode_index',r.get('seed')),r['frame'])
    bb={identity(r):r for r in b['rows']};cc={identity(r):r for r in c['rows']};assert bb.keys()==cc.keys()
    paired=[]
    for key,br in bb.items():
        cr=cc[key];assert br.get('roi_xyxy_224')==cr.get('roi_xyxy_224')
        row={k:br[k] for k in ['split','task','frame']};row['identity']=list(key);row['delta']={}
        for path in br['metrics'].keys() & cr['metrics'].keys():
            row['delta'][path]={m:cr['metrics'][path][m]-v for m,v in br['metrics'][path].items() if v is not None and cr['metrics'][path].get(m) is not None}
        paired.append(row)
    summary={'baseline':b['decoder'],'candidate':c['decoder'],'probes_sha256':b['probes_sha256'],'baseline_mean':b['mean'],'candidate_mean':c['mean'],'paired_rows':paired,'scope':'Fixed validation episodes and separate known simulator seed; pilot evidence, not all50 score'}
    (a.output/'comparison.json').write_text(json.dumps(summary,indent=2))
    bp={r['id']:r for r in torch.load(a.baseline/'predictions.pt',map_location='cpu',weights_only=False)}
    cp={r['id']:r for r in torch.load(a.candidate/'predictions.pt',map_location='cpu',weights_only=False)}
    for rid in bp:
        bv,cv=bp[rid]['views'],cp[rid]['views'];pathkeys=[k for k in ['raw','reencoded','cached','wm'] if k in bv and k in cv]
        columns=[('GT',bv['GT'])]+[(f'{tag} / {k}',views[k]) for k in pathkeys for tag,views in [('baseline',bv),('candidate',cv)]]
        if 'historical_WM' in bv:columns.append(('historical WM',bv['historical_WM']))
        br=next(r for r in b['rows'] if f"{r.get('dataset','seed')}_{r.get('episode_index',r.get('seed'))}_{r['frame']}"==rid)
        roi=br.get('roi_xyxy_224');sheet=Image.new('RGB',(228*len(columns),492 if roi else 260),'white');draw=ImageDraw.Draw(sheet)
        for j,(label,v) in enumerate(columns):
            draw.text((j*228+2,5),label,fill='black');sheet.paste(image(v),(j*228,25))
            if roi:
                x,y,x1,y1=roi;draw.rectangle((j*228+x,25+y,j*228+x1,25+y1),outline='cyan',width=1)
                sheet.paste(image(v[:,y:y1,x:x1]).resize((224,224),Image.Resampling.NEAREST),(j*228,267))
        sheet.save(a.output/f'{rid}.png')
    print(json.dumps({'baseline':b['mean'],'candidate':c['mean']}),flush=True)
if __name__=='__main__':main()
