# task_tokenizer.py
"""
Task Tokenizer for Dreamer4 VLM World Model.
Uses pretrained PaliGemma embeddings for semantic understanding.
"""
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import sentencepiece
    SENTENCEPIECE_AVAILABLE = True
except ImportError:
    SENTENCEPIECE_AVAILABLE = False
    logging.warning("sentencepiece not available. TaskTokenizer will not work.")


def maybe_download_tokenizer(path: str) -> Path:
    """Download tokenizer model if needed."""
    if path.startswith("gs://"):
        try:
            import gcsfs
            fs = gcsfs.GCSFileSystem(token="anon")
            local_path = Path("/tmp") / path.replace("gs://", "").replace("/", "_")
            if not local_path.exists():
                fs.get(path, str(local_path))
            return local_path
        except ImportError:
            raise RuntimeError("gcsfs is required. Install with: pip install gcsfs")
    else:
        return Path(path)


class TaskTokenizer(nn.Module):
    """
    Task Tokenizer with pretrained PaliGemma embeddings.
    
    Args:
        max_len: Maximum token sequence length (default: 48)
        tokenizer_path: Path to SentencePiece model file
        pretrained_embed_path: Path to pretrained embedding weights (.pth)
        embed_dim: Embedding dimension (2048 for PaliGemma)
        output_dim: Output dimension after projection (default: same as embed_dim)
        pad_token_id: Token ID for padding (default: 0)
        freeze_embeddings: Whether to freeze embeddings (default: True)
    """
    
    def __init__(
        self,
        max_len: int = 48,
        tokenizer_path: str = "gs://big_vision/paligemma_tokenizer.model",
        pretrained_embed_path: Optional[str] = None,
        embed_dim: int = 2048,
        output_dim: Optional[int] = None,
        pad_token_id: int = 0,
        freeze_embeddings: bool = True,
    ):
        super().__init__()
        
        if not SENTENCEPIECE_AVAILABLE:
            raise RuntimeError("sentencepiece is required. Install with: pip install sentencepiece")
        
        self._max_len = max_len
        self._pad_token_id = pad_token_id
        self._embed_dim = embed_dim
        self._output_dim = output_dim or embed_dim
        
        # Load SentencePiece tokenizer
        path = maybe_download_tokenizer(tokenizer_path)
        with open(path, "rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())
        
        vocab_size = self._tokenizer.vocab_size()
        logging.info(f"Loaded TaskTokenizer with vocab_size={vocab_size}")
        
        # Initialize embedding layer
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        
        # Load pretrained weights
        if pretrained_embed_path is not None:
            self._load_pretrained_embeddings(pretrained_embed_path)
        
        # Freeze embeddings
        if freeze_embeddings:
            self.embed.weight.requires_grad = False
            logging.info("Embeddings frozen (not trainable)")
        
        # Projection layer if output_dim differs
        if self._output_dim != embed_dim:
            self.proj = nn.Linear(embed_dim, self._output_dim)
        else:
            self.proj = nn.Identity()
    
    def _load_pretrained_embeddings(self, embed_path: str):
        """Load pretrained embedding weights from .pth file."""
        state = torch.load(embed_path, map_location='cpu')
        
        if isinstance(state, dict):
            if 'weight' in state:
                embed_weight = state['weight']
            elif 'embed.weight' in state:
                embed_weight = state['embed.weight']
            else:
                raise KeyError(f"Could not find embedding weights. Keys: {list(state.keys())}")
        else:
            embed_weight = state
        
        # Convert bfloat16 to float32 if needed
        if embed_weight.dtype == torch.bfloat16:
            embed_weight = embed_weight.float()
        
        # Load weights
        if embed_weight.shape[0] != self.vocab_size:
            min_vocab = min(embed_weight.shape[0], self.vocab_size)
            self.embed.weight.data[:min_vocab] = embed_weight[:min_vocab]
        else:
            self.embed.weight.data = embed_weight
        
        logging.info(f"Loaded pretrained embeddings: {embed_weight.shape}")
    
    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size()
    
    @property
    def max_len(self) -> int:
        return self._max_len
    
    @property
    def embed_dim(self) -> int:
        return self._embed_dim
    
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    def tokenize(self, prompt: str) -> Tuple[np.ndarray, np.ndarray]:
        """Tokenize a text prompt."""
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        tokens = (
            self._tokenizer.encode(cleaned_text, add_bos=True) 
            + self._tokenizer.encode("\n")
        )
        
        tokens_len = len(tokens)
        
        if tokens_len < self._max_len:
            padding = [self._pad_token_id] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
        else:
            tokens = tokens[:self._max_len]
            mask = [True] * self._max_len
        
        return np.asarray(tokens), np.asarray(mask)
    
    def tokenize_batch(self, prompts: str | list[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Tokenize a batch of prompts."""
        if isinstance(prompts, str):
            prompts = [prompts]
        
        batch_tokens, batch_masks = [], []
        for prompt in prompts:
            tokens, mask = self.tokenize(prompt)
            batch_tokens.append(tokens)
            batch_masks.append(mask)
        
        return np.stack(batch_tokens), np.stack(batch_masks)
    
    def get_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Get embeddings for token IDs, with projection."""
        emb = self.embed(token_ids)
        return self.proj(emb)
    
    def forward(
        self,
        prompts: str | list[str],
        device: Optional[torch.device] = None,
        return_embeddings: bool = False,
    ):
        """
        Tokenize and return tensors.
        
        Returns:
            If return_embeddings=False: (token_ids, masks)
            If return_embeddings=True: (token_ids, masks, embeddings)
        """
        tokens, masks = self.tokenize_batch(prompts)
        
        tokens_tensor = torch.from_numpy(tokens).long()
        masks_tensor = torch.from_numpy(masks).bool()
        
        if device is not None:
            tokens_tensor = tokens_tensor.to(device)
            masks_tensor = masks_tensor.to(device)
        
        if return_embeddings:
            self.embed = self.embed.to(tokens_tensor.device)
            self.proj = self.proj.to(tokens_tensor.device)
            embeddings = self.get_embeddings(tokens_tensor)
            return tokens_tensor, masks_tensor, embeddings
        
        return tokens_tensor, masks_tensor
    
    def decode(self, tokens) -> str:
        """Decode token IDs back to text."""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().numpy()
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()
        tokens = [t for t in tokens if t != self._pad_token_id]
        return self._tokenizer.decode(tokens)