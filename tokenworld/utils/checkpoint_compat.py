"""Read trusted research checkpoints whose Python module paths predate the release.

This preserves legacy config-object loading without creating global module aliases.
New checkpoints should store plain configuration dictionaries and state dictionaries.
"""
import io
import pickle
import torch
from tokenworld.utils.config import DictConfig

LEGACY_MODULES = {'dreamer4.dreamer4': 'tokenworld.dynamics.model', 'dreamer4.trainers': 'tokenworld.dynamics.trainer', 'dreamer4.train_dreamer_vlm': 'tokenworld.dynamics.train', 'dreamer4.task_tokenizer': 'tokenworld.dynamics.task_tokenizer', 'dreamer4.vqgan': 'tokenworld.decoder.model', 'dreamer4.train_decoder_robotwin': 'tokenworld.decoder.train', 'dreamer4.symmetric_qwen_codec': 'tokenworld.codec.decoder', 'dreamer4.vqvae': 'tokenworld.codec.vqvae', 'dreamer_datasets.robotwin_svae_dset': 'tokenworld.data.dataset', 'eval.prepare_closed_loop_manifest': 'tokenworld.evaluation.prepare_manifest', 'eval.evaluate_tokenworld_open_loop_fidelity': 'tokenworld.evaluation.open_loop', 'eval.closed_loop_oft': 'tokenworld.evaluation.closed_loop', 'eval.world_model_rollout': 'tokenworld.evaluation.rollout', 'eval.generate_from_ckpt': 'tokenworld.decoder.checkpoint', 'eval.utils.checkpoint_utils': 'tokenworld.utils.checkpoint', 'metrics.image_metrics': 'tokenworld.decoder.metrics.image', 'metrics.__init__': 'tokenworld.decoder.metrics.__init__', 'tokenworld.data.io': 'tokenworld.utils.io', 'dreamer4.train_decoder_simple': 'tokenworld.decoder.train'}


class LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "DictConfig" and (module == "__main__" or module.startswith("dreamer4.") or module in {"tokenworld.decoder.train", "tokenworld.dynamics.train"}):
            return DictConfig
        return super().find_class(LEGACY_MODULES.get(module, module), name)


class CheckpointPickle:
    Unpickler = LegacyUnpickler
    dump = staticmethod(pickle.dump)
    dumps = staticmethod(pickle.dumps)

    @staticmethod
    def load(handle, **kwargs):
        return LegacyUnpickler(handle, **kwargs).load()

    @staticmethod
    def loads(data, **kwargs):
        return LegacyUnpickler(io.BytesIO(data), **kwargs).load()


def load_checkpoint_payload(path, map_location="cpu", **kwargs):
    return torch.load(path, map_location=map_location, weights_only=False,
                      pickle_module=CheckpointPickle, **kwargs)
