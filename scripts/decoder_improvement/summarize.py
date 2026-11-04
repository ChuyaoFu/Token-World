#!/usr/bin/env python3
"""Collect decoder evidence without conflating panels or crop protocols."""
import argparse,csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from decoder_improvement.diagnose import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    reports=['baseline_final','candidate_v1/evaluation','candidate_v2/evaluation','candidate_v2_mse/evaluation','candidate_v3_color/evaluation','architecture_control_cnn/evaluation','architecture_multiscale/evaluation','broad_baseline','broad_v1','broad_v2','broad_architecture_control','broad_multiscale']
    rows=[];metadata={}
    for name in reports:
        path=a.run/name/'report.json'
        if not path.is_file():continue
        d=json.loads(path.read_text());metadata[name]={k:d[k] for k in ['decoder','decoder_sha256','probes_sha256','metric_protocol']}
        for split,paths in d['mean'].items():
            for branch,values in paths.items():
                if branch=='historical_WM':continue
                rows.append({'report':name,'split':split,'branch':branch,**values})
    fields=['report','split','branch','frames','psnr','lpips','mse_01','color_mae_01','edge_l1','roi_lpips_128','roi_mse_01','roi_red_iou','roi_red_area_ratio']
    with (a.run/'metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    result={'scope':'decoder-only; formal balanced100 model comparison deferred','metrics':rows,'reports':metadata,'source_sha256':{p.name:sha(p) for p in (ROOT/'scripts/decoder_improvement').glob('*') if p.is_file()},'notes':['Broad whole-image validation is400 frames/50 tasks; pilot WM validation is16 frames.','Only baseline_final and matching extended-probe evaluations use final GT object ROIs. Broad reports contain separate exploratory known_seed crops; do not mix crop numbers.','Architecture controls use the same explicit training-sample RNG; earlier v2 used the original global RNG.','Color shape experiment was not selected. Baseline deployment weights remain unchanged.']}
    (a.run/'summary.json').write_text(json.dumps(result,indent=2));print('reports',len(metadata),'metric rows',len(rows),flush=True)
if __name__=='__main__':main()
