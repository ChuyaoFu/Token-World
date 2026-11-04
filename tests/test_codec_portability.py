import torch
from tokenworld.codec.common import instantiate_codec
from tokenworld.codec.decoder import load_symmetric_qwen_decoder


def test_portable_codec_and_decoder_only_loader_agree(tmp_path):
    config = dict(method="symmetric_qwen", input_dim=32, native_tokens=6, tokens_per_view=6,
                  token_dim=4, model_dim=32, depth=1, num_heads=2, pixel_output_size=[8, 12],
                  grid_size=[2, 3], latent_grid_size=[2, 3], projection_hidden_dim=16)
    torch.manual_seed(42)
    codec = instantiate_codec(config).eval()
    features = [torch.randn(6, 32), torch.randn(6, 32)]
    _, mean, _ = codec.encode_with_posterior(features, sample_posterior=False)
    assert mean.shape == (2, 6, 4)
    expected = torch.stack(codec.decode(mean, [6, 6]))
    checkpoint = tmp_path / "codec.pt"
    torch.save({"version": 1, "codec_config": config, "state_dict": codec.state_dict()}, checkpoint)
    decoder, _ = load_symmetric_qwen_decoder(checkpoint, "cpu")
    torch.testing.assert_close(decoder(mean), expected)
