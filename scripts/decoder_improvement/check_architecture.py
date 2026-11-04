#!/usr/bin/env python3
"""Check that the new architecture starts as the exact frozen CNN baseline."""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from decoder_improvement.models import load_decoder


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--probes',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4);torch.cuda.set_device(0)
    base=load_decoder(a.baseline,'cuda:0',freeze=True);new=load_decoder(a.baseline,'cuda:0',architecture='multiscale',freeze=True)
    samples=torch.load(a.probes,map_location='cpu',weights_only=False)[:2]
    errors={}
    for key in ['raw','cached']:
        x=torch.stack([s[key] for s in samples]).to('cuda:0')
        with torch.autocast('cuda',dtype=torch.bfloat16):b=base(x)[0];c=new(x)[0]
        errors[key]=(b.float()-c.float()).abs().max().item();assert errors[key]==0,errors
    result=dict(initial_max_abs=errors,base_parameters=sum(p.numel() for p in base.parameters()),multiscale_parameters=sum(p.numel() for p in new.parameters()),new_outputs_zero_initialized=True,input_contract='same 108x2560 merger; no extra image features',changes=['explicit learned 9x12 spatial position','same merger reinjected at five scales through zero-initialized 1x1 projections','zero-initialized token-to-16x16-pixel residual detail head'])
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
