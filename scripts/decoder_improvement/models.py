"""Backward-compatible CNN baseline and zero-initialized spatial detail ablation."""
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from tokenworld.decoder.model import VQGAN
from tokenworld.decoder.checkpoint import _convert_to_dict
from tokenworld.utils.checkpoint_compat import load_checkpoint_payload


class MultiScaleDecoder(VQGAN):
    """Reinject the same 9x12 tokens at every scale plus a token-to-patch residual.

    All new output projections start at zero: initial RGB is identical to the
    release CNN. No GT pixels, extra image features, or WM changes are used.
    """
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        dim=kwargs['emb_dim'];channel=kwargs['channel'];mult=list(reversed(kwargs['channel_mult']))
        self.spatial_position=nn.Parameter(torch.zeros(1,dim,*self.grid_size))
        self.conditioners=nn.ModuleList([nn.Conv2d(dim,channel*m,1) for m in mult])
        for layer in self.conditioners:
            nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
        scale=2**(len(mult)-1)
        self.detail_head=nn.Sequential(nn.Conv2d(dim,128,3,padding=1),nn.SiLU(),nn.Conv2d(128,3*scale*scale,3,padding=1),nn.PixelShuffle(scale))
        nn.init.zeros_(self.detail_head[2].weight);nn.init.zeros_(self.detail_head[2].bias)
    def forward(self,x):
        if x.ndim==3:
            assert x.shape[1]==self.grid_size[0]*self.grid_size[1]
            x=self.input_norm(x)
            x=rearrange(x,'b (h w) c -> b c h w',h=self.grid_size[0],w=self.grid_size[1])
        elif x.ndim==4:x=self.input_norm(x.permute(0,2,3,1)).permute(0,3,1,2)
        else:raise ValueError(x.shape)
        x=x.float();source=x+self.spatial_position
        h=self.conv_in(x)+self.conditioners[0](source)
        for block in self.mid_blocks:h=block(h)
        for i,blocks in enumerate(self.up_blocks):
            for layer in blocks:h=layer(h)
            residual=self.conditioners[i+1](source)
            h=h+F.interpolate(residual,size=h.shape[-2:],mode='bilinear',align_corners=False).to(dtype=h.dtype)
        logits=self.conv_out(self.act_out(self.norm_out(h)))
        detail=self.detail_head(source)
        assert detail.shape==logits.shape,(detail.shape,logits.shape)
        out=torch.tanh(logits+detail.to(dtype=logits.dtype))
        if out.shape[-2:]!=self.output_size:out=F.interpolate(out,size=self.output_size,mode='bilinear',align_corners=False)
        return out,torch.zeros(1,device=out.device,dtype=out.dtype)


def load_decoder(path,device='cuda',*,freeze=False,architecture=None):
    saved=load_checkpoint_payload(path)
    actual=saved.get('decoder_architecture','baseline');target=actual if architecture is None else architecture
    config=_convert_to_dict(saved['config']);config=config.get('decoder',config)
    keys=('in_channel','emb_dim','channel','channel_mult','n_res_block','resolution','num_groups','use_attention')
    kwargs={k:config[k] for k in keys};kwargs['grid_size']=config.get('grid_size') or config.get('latent_grid_size') or [9,12];kwargs['input_norm']=config.get('input_norm',True)
    cls={'baseline':VQGAN,'multiscale':MultiScaleDecoder}[target];model=cls(**kwargs)
    state=saved['decoder_state_dict'] if 'decoder_state_dict' in saved else saved['state_dict']
    if target==actual:model.load_state_dict(state,strict=True)
    elif actual=='baseline' and target=='multiscale':
        incompatible=model.load_state_dict(state,strict=False)
        assert not incompatible.unexpected_keys
        assert all(k.startswith(('conditioners.','detail_head.','spatial_position')) for k in incompatible.missing_keys),incompatible
    else:raise ValueError(f'Unsupported architecture conversion {actual} -> {target}')
    model=model.to(device).eval()
    if freeze:model.requires_grad_(False)
    return model
