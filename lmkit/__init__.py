"""lmkit — a small toolkit for training language models from scratch.

Bring your own architecture (a torch.nn.Module mapping token ids to logits) and
config; lmkit supplies the pretrain/anneal loop, SFT, tokenizer, sharding, eval,
hub push, and optional experiment tracking. It never imports a concrete model.
"""
__version__ = "0.1.0"
