"""Safety properties for automatic feature cleanup and deterministic continuation."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest import mock
import unittest
import torch

SOURCE=Path(__file__).resolve().parents[1]/'scripts/deepstack_pipeline.py'
spec=importlib.util.spec_from_file_location('deepstack_pipeline',SOURCE)
pipeline=importlib.util.module_from_spec(spec);spec.loader.exec_module(pipeline)


class PipelineTest(unittest.TestCase):
    def test_resumed_batches_match_uninterrupted_order(self):
        full=list(pipeline.StepBatches(17,4,0,20,3047))
        resumed=list(pipeline.StepBatches(17,4,7,20,3047))
        self.assertEqual(full[7:],resumed)
        self.assertEqual(len(set(x for batch in full[:4] for x in batch)),16)

    def fixture(self,root):
        out=root/'run';out.mkdir();features=out/'features';features.mkdir()
        run={'run_id':'test-run'};entry={'file':'train_episode.pt'}
        a=SimpleNamespace(output=out,stage='cleanup',rank=0)
        checkpoint={'run_id':'test-run','model':{'weight':torch.tensor([1.])}}
        for name in ['adapter_best.pt','adapter_last.pt']:torch.save(checkpoint,out/name)
        complete={'status':'complete','run_id':'test-run','best_sha256':pipeline.digest(out/'adapter_best.pt'),'last_sha256':pipeline.digest(out/'adapter_last.pt')}
        (out/'training_complete.json').write_text(json.dumps(complete))
        (out/'action_evaluation.json').write_text(json.dumps({'status':'complete','run_id':'test-run','checkpoint_sha256':complete['best_sha256']}))
        (features/'.owner.json').write_text(json.dumps({'run_id':'test-run','output':str(out.resolve())}))
        (features/entry['file']).write_bytes(b'owned cache')
        (out/'export_complete.json').write_text(json.dumps({'files':[{'file':entry['file'],'bytes':11}]}))
        return a,run,[entry]

    def test_cleanup_only_removes_owned_features(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);a,run,entries=self.fixture(root)
            outside=root/'unrelated.pt';outside.write_text('keep')
            with mock.patch.object(pipeline,'context',return_value=(run,entries)):
                pipeline.cleanup(a)
                pipeline.cleanup(a)
            self.assertEqual(outside.read_text(),'keep')
            self.assertTrue((a.output/'adapter_best.pt').exists())
            self.assertTrue((a.output/'adapter_last.pt').exists())
            self.assertFalse((a.output/'features/train_episode.pt').exists())

    def test_cleanup_refuses_incomplete_training(self):
        with tempfile.TemporaryDirectory() as temp:
            a,run,entries=self.fixture(Path(temp))
            path=a.output/'training_complete.json';state=json.loads(path.read_text());state['status']='failed';path.write_text(json.dumps(state))
            with mock.patch.object(pipeline,'context',return_value=(run,entries)),self.assertRaises(ValueError):pipeline.cleanup(a)
            self.assertTrue((a.output/'features/train_episode.pt').exists())

    def test_cleanup_refuses_unexpected_or_symlink_files(self):
        for symlink in [False,True]:
            with self.subTest(symlink=symlink),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);a,run,entries=self.fixture(root)
                outside=root/'outside';outside.write_text('keep')
                extra=a.output/'features/extra'
                if symlink:extra.symlink_to(outside)
                else:extra.write_text('not in manifest')
                with mock.patch.object(pipeline,'context',return_value=(run,entries)),self.assertRaises(ValueError):pipeline.cleanup(a)
                self.assertTrue((a.output/'features/train_episode.pt').exists())
                self.assertEqual(outside.read_text(),'keep')

    def test_cleanup_refuses_wrong_owner_or_modified_checkpoint(self):
        for kind in ['owner','checkpoint']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as temp:
                a,run,entries=self.fixture(Path(temp))
                if kind=='owner':(a.output/'features/.owner.json').write_text('{}')
                else:(a.output/'adapter_best.pt').write_bytes(b'changed')
                with mock.patch.object(pipeline,'context',return_value=(run,entries)),self.assertRaises(ValueError):pipeline.cleanup(a)
                self.assertTrue((a.output/'features/train_episode.pt').exists())


if __name__=='__main__':unittest.main()
