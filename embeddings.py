"""
embeddings.py — Sentence embeddings for dense retrieval
========================================================
Runs sentence-transformers/all-MiniLM-L6-v2 through ONNX Runtime, so dense search needs
no PyTorch and works on a plain CPU (CI, an old laptop, an air-gapped box).

Same output as sentence-transformers: mean pooling over tokens, then L2 normalisation,
384 dimensions, input truncated at 256 tokens.

The model files (model.onnx, tokenizer.json) are downloaded once into EMBED_MODEL_DIR.
For an offline machine, copy those two files into that folder yourself.

    python embeddings.py --download
"""

import argparse
import os

import numpy as np
import requests

from config import EMBED_MODEL_DIR, EMBED_MODEL_REPO

EMBED_DIMS = 384
MAX_TOKENS = 256          # sentence-transformers' max_seq_length for this model

_HF_FILES = {
    "model.onnx": f"https://huggingface.co/{EMBED_MODEL_REPO}/resolve/main/onnx/model.onnx",
    "tokenizer.json": f"https://huggingface.co/{EMBED_MODEL_REPO}/resolve/main/tokenizer.json",
}


def download_model(model_dir: str = EMBED_MODEL_DIR) -> str:
    """Fetch the ONNX model and tokenizer if they are not already on disk."""
    os.makedirs(model_dir, exist_ok=True)
    for name, url in _HF_FILES.items():
        path = os.path.join(model_dir, name)
        if os.path.exists(path):
            continue
        print(f"Downloading {url}")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            tmp = path + ".part"
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
            os.replace(tmp, path)
    return model_dir


class MiniLMEmbedder:
    """all-MiniLM-L6-v2 on ONNX Runtime. Imports are lazy so the core install does not need them."""

    dims = EMBED_DIMS
    name = EMBED_MODEL_REPO

    def __init__(self, model_dir: str = EMBED_MODEL_DIR):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_path = os.path.join(model_dir, "model.onnx")
        tok_path = os.path.join(model_dir, "tokenizer.json")
        if not (os.path.exists(model_path) and os.path.exists(tok_path)):
            raise FileNotFoundError(
                f"Embedding model not found in {model_dir}. Run: python embeddings.py --download"
            )
        self._tok = Tokenizer.from_file(tok_path)
        self._tok.enable_truncation(MAX_TOKENS)
        self._tok.enable_padding(pad_id=0, pad_token="[PAD]")
        self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._inputs = {i.name for i in self._session.get_inputs()}

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        out = []
        for start in range(0, len(texts), batch_size):
            enc = self._tok.encode_batch(texts[start:start + batch_size])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask,
                    "token_type_ids": np.zeros_like(ids)}
            hidden = self._session.run(None, {k: v for k, v in feed.items() if k in self._inputs})[0]
            m = mask[..., None].astype(np.float32)
            pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
            out.append(pooled / np.linalg.norm(pooled, axis=1, keepdims=True))
        return np.vstack(out).astype(np.float32)

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0].tolist()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Download or check the embedding model")
    p.add_argument("--download", action="store_true", help="download model.onnx and tokenizer.json")
    p.add_argument("--model-dir", default=EMBED_MODEL_DIR)
    args = p.parse_args()
    if args.download:
        download_model(args.model_dir)
    emb = MiniLMEmbedder(args.model_dir)
    v = emb.embed(["The weather is lovely today.", "It's so sunny outside!", "He drove to the stadium."])
    print("Similarity check (sentence-transformers reference: 0.666 and 0.105):")
    print(f"  {float(v[0] @ v[1]):.3f}  {float(v[0] @ v[2]):.3f}")
