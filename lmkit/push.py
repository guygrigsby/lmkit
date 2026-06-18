"""Push checkpoints and files to a model hub (HuggingFace), private-first.

Generic: the caller supplies the repo id and the local->remote file map; no repo
names, owners, or tokens are baked in (token comes from the `token` arg or the
standard `HF_TOKEN` env). Creating a repo on the Hub defaults to PUBLIC, so when
`private=True` we create and assert private before the first upload.

    from lmkit.push import push
    push("you/my-model", {"checkpoints/final/model.pt": "model.pt",
                          "model.py": "model.py", "config.py": "config.py"})
"""
from __future__ import annotations

from pathlib import Path


def ensure_repo(api, repo_id: str, *, repo_type: str = "model", private: bool = True):
    api.create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True)
    info = api.repo_info(repo_id, repo_type=repo_type)
    if private and info.private is not True:
        raise RuntimeError(f"{repo_id} is NOT private — aborting push")
    return info


def push(repo_id: str, files: dict, *, repo_type: str = "model",
         private: bool = True, token: str | None = None):
    """Upload a {local_path: repo_path} map to `repo_id`. Returns the repo info."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    ensure_repo(api, repo_id, repo_type=repo_type, private=private)
    for local, remote in files.items():
        if not Path(local).exists():
            raise FileNotFoundError(f"missing artifact: {local}")
        api.upload_file(path_or_fileobj=str(local), path_in_repo=remote,
                        repo_id=repo_id, repo_type=repo_type)
    return api.repo_info(repo_id, repo_type=repo_type)
