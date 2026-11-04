import sys
import types

import pytest
import torch

from tokenworld.decoder.checkpoint import load_external_decoder
from tokenworld.decoder.model import VQGAN
from tokenworld.utils.checkpoint import load_decoder_from_ckpt


@pytest.mark.parametrize('weight_key', ['state_dict', 'decoder_state_dict'])
@pytest.mark.parametrize('legacy_config', [False, True])
def test_decoder_loads_legacy_config_without_old_package(tmp_path, monkeypatch, weight_key, legacy_config):
    config = dict(in_channel=3, emb_dim=4, channel=8, channel_mult=[1, 1],
                  n_res_block=1, resolution=8, num_groups=4, use_attention=False,
                  grid_size=[2, 2])
    torch.manual_seed(42)
    model = VQGAN(**config).eval()
    inputs = torch.randn(2, 4, 4)
    expected = model.decode(inputs)
    path = tmp_path / 'decoder.pt'
    if legacy_config:
        parent = types.ModuleType('dreamer4')
        parent.__path__ = []
        module = types.ModuleType('dreamer4.train_decoder_simple')
        module.DictConfig = type('DictConfig', (), {'__module__': module.__name__})
        obj = module.DictConfig()
        obj.decoder = module.DictConfig()
        obj.decoder.__dict__.update(config)
        with monkeypatch.context() as patch:
            patch.setitem(sys.modules, 'dreamer4', parent)
            patch.setitem(sys.modules, module.__name__, module)
            torch.save({'config': obj, weight_key: model.state_dict()}, path)
    else:
        torch.save({'config': {'decoder': config}, weight_key: model.state_dict()}, path)
    decoder = load_external_decoder(path, device='cpu')
    assert decoder.state_dict().keys() == model.state_dict().keys()
    torch.testing.assert_close(decoder.decode(inputs), expected)
    frozen = load_decoder_from_ckpt(path)
    assert all(not parameter.requires_grad for parameter in frozen.parameters())
    torch.testing.assert_close(frozen.decode(inputs), expected)
    assert 'dreamer4.train_decoder_simple' not in sys.modules
