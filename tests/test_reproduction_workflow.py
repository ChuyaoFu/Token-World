"""Workflow tests use a recording interpreter; no devices or model weights needed."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.fixture
def workflow(tmp_path):
    root = tmp_path / 'checkout with spaces'
    scripts = root / 'scripts'
    scripts.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1]
    for name in ['reproduce_world_model.sh', 'train_world_model.sh']:
        shutil.copy2(source / 'scripts' / name, scripts / name)
    (scripts / 'run_python.sh').write_text('#!/usr/bin/env bash\nexec "$PYTHON_BIN" "$@"\n')
    (scripts / '_environment.sh').write_text('load_training_environment() { :; }\n')
    for relative in ['configs/robotwin/dynamics/svae_d16.yaml',
                     'configs/robotwin/splits/seed3047.json',
                     'outputs/svae_d16_3views/best.pt']:
        p = root / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('fixture')
    model, data = root / 'model files', root / 'data files'
    model.mkdir(); data.mkdir()
    log = tmp_path / 'calls.jsonl'
    python = tmp_path / 'recording python'
    python.write_text(f'#!{sys.executable}\n' + '''import json,os,sys
from pathlib import Path
args=sys.argv[1:]
if '--finalize-only' in args: stage='finalize'
elif 'torch.distributed.run' in args: stage='export'
elif any(a.endswith('check_world_model_cache.py') for a in args): stage='check'
else: stage='train'
with open(os.environ['WORKFLOW_LOG'],'a') as f: f.write(json.dumps({'stage':stage,'args':args})+'\\n')
if stage==os.environ.get('FAIL_STAGE'): sys.exit(19)
''')
    python.chmod(0o755)
    env = {**os.environ, 'WORKFLOW_LOG': str(log)}
    command = ['bash', str(scripts / 'reproduce_world_model.sh'), '--python', str(python),
               '--model', str(model), '--data-root', str(data)]
    return root, log, env, command


def invoke(workflow, *args, fail=None):
    root, log, env, command = workflow
    if fail:
        env = {**env, 'FAIL_STAGE': fail}
    result = subprocess.run(command + list(args), cwd=root.parent, env=env,
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return result, calls


@pytest.mark.parametrize('stage,expected', [
    ('all', ['export', 'finalize', 'check', 'train']),
    ('export', ['export', 'finalize', 'check']),
    ('check', ['check']), ('train', ['check', 'train']),
])
def test_stage_order_and_argument_boundaries(workflow, stage, expected):
    root, _, _, _ = workflow
    result, calls = invoke(workflow, '--stage', stage, '--gpus', '4')
    assert result.returncode == 0, result.stdout
    assert [c['stage'] for c in calls] == expected
    for c in calls:
        if c['stage'] == 'export':
            assert '--nproc_per_node=4' in c['args']
            assert str(root / 'model files') in c['args']
        if c['stage'] == 'train':
            assert c['args'][c['args'].index('--num_processes') + 1] == '4'
            assert str(root / 'outputs/world_model_svae_d16_cam_high_100k') in c['args']


@pytest.mark.parametrize('stage,expected_count', [('export', 1), ('finalize', 2), ('check', 3), ('train', 4)])
def test_failure_stops_dependent_stages_and_preserves_exit_status(workflow, stage, expected_count):
    result, calls = invoke(workflow, '--stage', 'all', fail=stage)
    assert result.returncode == 19, result.stdout
    assert len(calls) == expected_count
    assert calls[-1]['stage'] == stage


def test_dry_run_does_not_execute_or_create_outputs(workflow):
    root, log, _, _ = workflow
    result, calls = invoke(workflow, '--dry-run')
    assert result.returncode == 0, result.stdout
    assert not calls and not log.exists()
    assert not (root / 'outputs/cache').exists()
    assert not (root / 'outputs/world_model_svae_d16_cam_high_100k').exists()


def test_existing_training_checkpoint_requires_explicit_resume(workflow):
    root, _, _, _ = workflow
    output = root / 'outputs/world_model_svae_d16_cam_high_100k'
    output.mkdir(parents=True)
    checkpoint = output / 'checkpoint_step_0010000.pt'
    checkpoint.write_text('preserve')
    result, calls = invoke(workflow, '--stage', 'train')
    assert result.returncode == 2 and not calls
    result, calls = invoke(workflow, '--stage', 'train', '--resume', str(checkpoint))
    assert result.returncode == 0, result.stdout
    assert [c['stage'] for c in calls] == ['check', 'train']
    assert calls[-1]['args'][calls[-1]['args'].index('--resume_from') + 1] == str(checkpoint)
    assert checkpoint.read_text() == 'preserve'
