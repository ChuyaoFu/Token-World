"""Run with python tests/check_open_loop_fidelity_regressions.py (CPU only)."""
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F

root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('canonical', root/'tokenworld/evaluation/open_loop.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
a=m.FidelityAccumulator()
x=torch.tensor([[3.,4.],[0.,2.],[0.,2.]])
y=torch.tensor([[3.,0.],[0.,2.],[0.,0.]])
a.update(x[:1],y[:1]); a.update(x[1:],y[1:])
expected={'nmse':20/33,'mse':20/6,'mae':1.0,'row_cosine_similarity':1.6/3,'max_absolute_error':4.0,'rows':3,'elements':6}
for k,v in expected.items():
    assert math.isclose(a.report()[k],v,rel_tol=1e-12,abs_tol=1e-12),(k,a.report()[k],v)
for schema,d in [('robotwin14',14),('muka7',7)]:
    ds=SimpleNamespace(action_schema=schema, action_mean=torch.full((d,),0.9), action_std=torch.full((d,),2.0))
    discrete,continuous=m._cache_action_to_model(torch.full((d,),0.55),ds,torch.device('cpu'))
    assert (discrete==1).all()
    assert torch.allclose(continuous,torch.full_like(continuous,-0.175))

for views,squeeze in [(1,True),(1,False),(2,False)]:
    native=torch.randn(7,views,3,4)*3+7
    mean=torch.full((views,1,4),7.0); std=torch.full((views,1,4),3.0)
    states=torch.arange(7).float().unsqueeze(1).expand(-1,14)
    payload={'latents':{'qwen_raw':native},'states':states,'actions':states}
    record={'dataset':'fixture','task':'test','episode_index':0,'outcome':'failure'}
    ds=SimpleNamespace(records=[record],_load_episode=lambda _:payload,latent_method='qwen_raw',view_indices=list(range(views)),latent_mean=mean,latent_std=std,action_mean=torch.zeros(14),action_std=torch.ones(14),action_schema='robotwin14')
    class Oracle:
        step=4
        def step_with_history(self,h,d,c,p):
            expected=(native[self.step-4:self.step]-mean)/std
            torch.testing.assert_close(h[0],expected)
            assert d.shape==(1,5,2) and c.shape==(1,5,12)
            result=((native[self.step]-mean)/std).unsqueeze(0)
            state=states[self.step].unsqueeze(0)
            self.step+=1
            return (result[:,0] if squeeze else result),state
    class Expert:
        def actions(self,task,outcome,state,features):
            assert outcome=='failure'
            return features.mean((1,2)).unsqueeze(1)
    for reference_mode in ['cached','raw_teacher']:
        ds._load_raw_images=lambda i,r,ids:native[ids]
        m._extract_teacher_qwen=lambda teacher,frame,normalize:frame.unsqueeze(0)
        result,acc,act,_,_=m._run_episode(dataset=ds,record_index=0,world_model=Oracle(),codec=None,teacher=None,action_expert=Expert(),normalize_features=None,device=torch.device('cpu'),max_steps=0,query_every=1,reference_mode=reference_mode,initial_context_frames=4)
        assert result['predicted_frames']==3
        assert acc.report()['nmse']<1e-12
        assert act.report()['nmse']<1e-12
print('PASS: analytical metric values; training gripper threshold; denormalization; 1/2-view recursive history; raw-reference frame alignment; action pairing')

records=[{'task':f'task_{t:02}', 'outcome':o} for t in range(50) for o in ('failure','success') for _ in range(5)]
selected=m.select_episode_indices(records,list(range(500)),100,'balanced_task_outcome',3047)
assert selected==list(range(0,500,5))
assert len({(records[i]['task'],records[i]['outcome']) for i in selected})==100

# A synthetic clock proves policy/GT latency cannot enter step_time.
clock=SimpleNamespace(now=0.0)
original_time=m.time
m.time=SimpleNamespace(perf_counter=lambda:clock.now)
class TimedOracle(Oracle):
    def step_with_history(self,*args):
        clock.now+=0.125
        return super().step_with_history(*args)
class TimedExpert(Expert):
    def actions(self,*args):
        clock.now+=10.0
        return super().actions(*args)
class TimedCodec:
    tokens_per_view=3
    def decode(self,x,lengths):
        clock.now+=0.025
        return x
try:
    for method,expected_seconds in [('qwen_raw',0.125),('symmetric_qwen',0.150)]:
        ds.latent_method=method
        payload['latents'][method]=native
        result,*_=m._run_episode(dataset=ds,record_index=0,world_model=TimedOracle(),codec=TimedCodec(),teacher=None,action_expert=TimedExpert(),normalize_features=None,device=torch.device('cpu'),max_steps=0,query_every=1,reference_mode='cached',initial_context_frames=4)
        assert all(math.isclose(t,expected_seconds,abs_tol=1e-10) for t in result['step_seconds']), result['step_seconds']
        assert result['elapsed_seconds']>60
        report=m.step_time_report(result['step_seconds'],method,warmup_steps=1)
        assert report['measured_steps']==2 and report['warmup_steps_excluded']==1
        assert report['decoder_applied']==(method!='qwen_raw')
finally:
    m.time=original_time
print('PASS: balanced100 coverage; timed prediction/decode excludes policy and reference branches; warmup accounting')

# Single-GT initialization must not feed later ground-truth latents or states.
ds.latent_method='qwen_raw'
class SingleFrameModel:
    def __init__(self):
        self.expected_latent=((native[0]-mean)/std).unsqueeze(0)
        self.expected_state=states[0].unsqueeze(0)
        self.calls=0
        self.outputs=[]
    def step(self,current,discrete,continuous,proprio):
        torch.testing.assert_close(current,self.expected_latent)
        torch.testing.assert_close(proprio,self.expected_state)
        self.expected_latent=current+0.2
        self.expected_state=proprio-0.1
        self.calls+=1
        self.outputs.append(self.expected_latent*std+mean)
        return self.expected_latent,self.expected_state
for reference_mode in ['cached','raw_teacher']:
    wm=SingleFrameModel()
    result,acc,*_=m._run_episode(dataset=ds,record_index=0,world_model=wm,codec=None,teacher=None,action_expert=None,normalize_features=None,device=torch.device('cpu'),max_steps=0,query_every=1,reference_mode=reference_mode,initial_context_frames=1)
    assert wm.calls==6 and result['predicted_frames']==6 and result['initial_context_frames']==1
    expected=m.FidelityAccumulator();expected.update(native[1:],torch.cat(wm.outputs))
    for key in ['nmse','row_cosine_similarity','mae']:
        assert math.isclose(acc.report()[key],expected.report()[key],rel_tol=1e-7)
    assert result['step_time']['initial_context_frames']==1
print('PASS: one GT frame; recursive predicted visual/proprio feedback; six generated frames; reference alignment')

# Chunk output must follow predicted frames without resetting autoregressive state.
wm=SingleFrameModel()
result,*_=m._run_episode(dataset=ds,record_index=0,world_model=wm,codec=None,teacher=None,
    action_expert=None,normalize_features=None,device=torch.device('cpu'),max_steps=0,
    query_every=1,reference_mode='cached',initial_context_frames=1,chunk_size=2)
assert wm.calls==6
assert [(c['first_predicted_frame'],c['last_predicted_frame']) for c in result['chunks']]==[(1,2),(3,4),(5,6)]
for chunk in result['chunks']:
    a,b=chunk['first_predicted_frame'],chunk['last_predicted_frame']
    expected=m.FidelityAccumulator();expected.update(native[a:b+1],torch.cat(wm.outputs[a-1:b]))
    assert math.isclose(chunk['qwen_feature_metrics']['nmse'],expected.report()['nmse'],rel_tol=1e-7)
print('PASS: contiguous chunk boundaries and analytical chunk metrics without GT resets')
