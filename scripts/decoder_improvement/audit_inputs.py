#!/usr/bin/env python3
"""Audit codec state equality, decoder config, normalization, and data lineage."""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from diagnose import sha

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--probes-manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    manifest=json.loads(a.probes_manifest.read_text());current=Path(manifest['arguments']['codec']);source=Path(manifest['cache_spec']['codec_checkpoints']['symmetric_qwen']);decoder=Path(manifest['arguments']['decoder'])
    aa,bb=[torch.load(x,map_location='cpu',weights_only=False,mmap=True) for x in [current,source]]
    sa,sb=aa['state_dict'],bb['state_dict'];samekeys=sa.keys()==sb.keys()
    different=[k for k in sa if k not in sb or not torch.equal(sa[k],sb[k])]
    d=torch.load(decoder,map_location='cpu',weights_only=False,mmap=True)
    config=d['config'];config=config if isinstance(config,dict) else vars(config)
    def plain(o):
        if isinstance(o,dict):return {k:plain(v) for k,v in o.items()}
        if isinstance(o,(list,tuple)):return [plain(v) for v in o]
        if hasattr(o,'__dict__'):return plain(vars(o))
        return o
    result={'codec_current':str(current),'codec_source':str(source),'file_hashes':[sha(x) for x in [current,source]],'same_state_keys':samekeys,'state_tensor_count':len(sa),'different_tensors':different,'state_identical':samekeys and not different,'decoder':str(decoder),'decoder_config':plain(config),'training_loss':'tokenworld.decoder.train.compute_reconstruction_loss = MSE on [-1,1]; no perceptual or edge training loss','grid_path':'Qwen image_grid_thw 1x18x24 expected -> spatial_merge_size 2 -> 9x12; row-major rearrange; decoder four x2 upsample stages 144x192 -> bilinear224x224','training_input':'unnormalized cached S-VAE z -> frozen codec -> bf16 merger -> learned LayerNorm -> convolutional RGB decoder -> tanh','preprocessing_note':'fixed diagnostics preserve Table I tensor bilinear+antialias -> uint8 PIL path; historical cache export preprocessing must be compared separately','source_paths':[str(ROOT/'tokenworld/data/dataset.py'),str(ROOT/'tokenworld/decoder/train.py'),str(ROOT/'tokenworld/decoder/model.py')]}
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['decoder_config']}),flush=True)
if __name__=='__main__':main()
