#!/usr/bin/env python3
"""Analytical invariants for image ranges, PSNR, SSIM and LPIPS."""
import argparse,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from compare_rgb.common import pixels,ImageMetrics,atomic_json


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
 black=np.zeros((224,224,3),dtype=np.uint8);white=np.full_like(black,255)
 values=pixels(torch.stack([torch.full((3,224,224),-1.),torch.ones(3,224,224)]))
 assert np.array_equal(values[0],black) and np.array_equal(values[1],white)
 m=ImageMetrics('cpu');same=m.frame(white,white);different=m.frame(white,black)
 assert same['mse_01']==0 and same['psnr']==120 and abs(same['ssim']-1)<1e-6 and same['lpips']==0,same
 assert abs(different['mse_01']-1)<1e-6 and abs(different['psnr'])<1e-6 and different['lpips']>0,different
 atomic_json(dict(status='passed',range_endpoints='[-1,1] map to [0,255] exactly',identical=same,white_vs_black=different,zero_MSE_handling='PSNR capped at120dB to keep JSON finite; no scored context frames'),a.output)
 print('metric invariants passed',flush=True)
if __name__=='__main__':main()
