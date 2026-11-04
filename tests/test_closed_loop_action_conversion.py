import unittest
import json,tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from tokenworld.evaluation.closed_loop import policy_to_cache_order,action_to_model_inputs

class ActionConversionTest(unittest.TestCase):
    def setUp(self):
        self.stats={'robot':{'action':{'min':[-2.0]*12+[0.0,0.0],'max':[2.0]*12+[1.0,1.0]}}}
    def test_binary_grippers_do_not_use_joint_inverse_normalization(self):
        q=torch.zeros(1,4,14)
        q[0,:,12]=torch.tensor([0.0,0.2,0.49,0.50])
        q[0,:,13]=torch.tensor([1.0,0.8,0.5,0.49])
        a=policy_to_cache_order(q,self.stats)
        self.assertTrue(torch.equal(a[0,:,6],torch.tensor([0.,0.,0.,1.])))
        self.assertTrue(torch.equal(a[0,:,13],torch.tensor([1.,1.,1.,0.])))
        discrete,_=action_to_model_inputs(a[0],torch.zeros(14),torch.ones(14))
        self.assertTrue(torch.equal(discrete,a[0,:,[6,13]].long()))
    def test_arm_order_and_scale_are_preserved(self):
        q=torch.zeros(1,14);q[0,:6]=-1;q[0,6:12]=1
        a=policy_to_cache_order(q,self.stats)
        self.assertTrue(torch.equal(a[0,:6],torch.full((6,),-2.)))
        self.assertTrue(torch.equal(a[0,7:13],torch.full((6,),2.)))
    def test_explicit_legacy_mode_reproduces_old_gripper_rule(self):
        q=torch.zeros(1,14);q[0,12:]=torch.tensor([.2,.8])
        old=policy_to_cache_order(q,self.stats,gripper_mode='legacy_minmax')
        self.assertTrue(torch.allclose(old[0,[6,13]],torch.tensor([.6,.9])))
    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):policy_to_cache_order(torch.zeros(1,14),self.stats,gripper_mode='unknown')
    def test_old_outputs_cannot_be_silently_reused_with_binary_mode(self):
        from tokenworld.evaluation.closed_loop import run_episode,episode_name
        entry=dict(manifest_index=0,outcome='success',task='test',dataset='robotwin_success',episode_index=0,cache_path='x.pt',num_frames=2)
        payload={'latents':{'symmetric_qwen':torch.zeros(2,1,108,16)},'actions':torch.zeros(2,14),'states':torch.zeros(2,14)}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'episodes'/episode_name(entry);(out/'videos').mkdir(parents=True)
            (out/'videos/closed_loop.mp4').touch();(out/'thumbnail.png').touch();(out/'metadata.json').write_text('{}')
            args=SimpleNamespace(cache=root,max_frames=0,output_root=root,skip_existing=True,gripper_mode='binary',cache_mode='reset')
            kwargs=dict(policy=None,codec=None,world_model=None,rgb_decoder=None,latent_mean=None,latent_std=None,action_mean=None,action_std=None,policy_statistics=None,device='cpu')
            with patch('tokenworld.evaluation.closed_loop.torch_load',return_value=payload):
                with self.assertRaisesRegex(ValueError,'Existing output uses gripper_mode'):
                    run_episode(entry,args,**kwargs)
                args.gripper_mode='legacy_minmax'
                self.assertEqual(run_episode(entry,args,**kwargs),{})
if __name__=='__main__':unittest.main()
