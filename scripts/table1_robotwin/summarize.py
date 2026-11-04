import json,statistics,sys
from collections import defaultdict
from pathlib import Path
here=Path(sys.argv[1]).resolve()
r=json.loads((here/'report.json').read_text())
metrics={'feature_cosine':('qwen_features','qwen_feature','row_cosine_similarity'),'feature_nmse':('qwen_features','qwen_feature','nmse'),'action_cosine':('action_expert_outputs_50x14','action_expert_output_50x14','row_cosine_similarity'),'action_nmse':('action_expert_outputs_50x14','action_expert_output_50x14','nmse')}
ref=json.loads((Path(__file__).resolve().with_name('archived_table1_values.json')).read_text())
grp=defaultdict(list)
for e in r['episodes']:grp[e['task']].append(e)
ev=r['evaluation']
assert len(grp)==50 and all(sorted(e['outcome'] for e in g)==['failure','success'] for g in grp.values())
assert (ev['evaluated_episodes'],ev['generated_frames'],ev['replay_chunks'])==(100,22152,1524)
assert r['protocol']['cross_segment_cache'] in ('gru','reset')
out={'sample':{k:ev[k] for k in ('evaluated_episodes','generated_frames','replay_chunks')},'cross_segment_cache':r['protocol']['cross_segment_cache'],'per_task':{},'metrics':{}}
out['action_interface']=r['protocol'].get('action_interface','merger_only')
for k,(key,ek,stat) in metrics.items():
 by={t:statistics.mean(e[ek][stat] for e in g) for t,g in grp.items()}
 for t,v in by.items():out['per_task'].setdefault(t,{})[k]=v
 micro=r['fidelity'][key][stat]
 # Rebuild the micro score using per-episode statistics without weighting every episode equally.
 g=r['episodes']
 if stat=='nmse':
  err=sum(e[ek]['mse']*e[ek]['elements'] for e in g);energy=sum(e[ek]['mse']*e[ek]['elements']/e[ek][stat] for e in g);recomputed=err/energy
 else:recomputed=sum(e[ek][stat]*e[ek]['rows'] for e in g)/sum(e[ek]['rows'] for e in g)
 assert abs(recomputed-micro)<1e-9
 original=ref['fidelity'][key][stat]
 out['metrics'][k]=dict(release_micro=micro,release_task_macro=statistics.mean(by.values()))
 if k.startswith('feature_') or out['action_interface']=='merger_only':
  out['metrics'][k].update(paper_source_micro=original,release_micro_minus_source=micro-original)
(here/'comparison.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out['metrics'],indent=2))
