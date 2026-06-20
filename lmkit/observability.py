"""Optional experiment tracking.

Logs to an Aim server at env ``AIM_REPO`` (e.g. ``aim://host:53800``) when set,
and no-ops otherwise — training never depends on the tracker being installed or
reachable. The endpoint is supplied by the environment, never baked in.
"""
from __future__ import annotations

import os
import sys
from typing import Optional


def make_run(experiment: str, hparams: Optional[dict] = None,
             name: Optional[str] = None, description: Optional[str] = None):
    """Return an Aim ``Run`` logging to ``$AIM_REPO``, or ``None`` if unset or
    unavailable.

    ``experiment`` groups runs; ``name`` is the human label shown in the runs list
    (default is the run hash, which is unhelpful), and ``description`` is a one-line
    summary. The caller logs with ``run.track(value, name=..., step=...)`` and calls
    ``run.close()`` at the end; guard every use with ``if run:``.
    """
    repo = os.environ.get("AIM_REPO")
    if not repo:
        return None
    try:
        import aim
    except Exception:
        return None
    try:
        run = aim.Run(repo=repo, experiment=experiment)
        if name:
            run.name = name
        if description:
            run.description = description
        if hparams:
            run["hparams"] = dict(hparams)
        return run
    except Exception as e:  # unreachable server, version skew, etc.
        print(f"[lmkit.observability] Aim tracking disabled: {e}", file=sys.stderr)
        return None
