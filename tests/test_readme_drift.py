"""Drift guards: every fact the README restates about the code is machine-checked
here, with the code as the source of truth. Static only (stdlib ast + text), so
this runs without torch or the heavy deps and stays fast enough for CI.

If a check fails, fix the README (or the code) named in the message; do not edit
this test to make it pass unless the fact it guards genuinely changed.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
PROJECT_PREFIXES = ("lmkit", "examples")


def _readme() -> str:
    return README.read_text(encoding="utf-8")


def _python_blocks(text: str) -> list[str]:
    return re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)


def _module_file(mod: str) -> Path | None:
    """Resolve a dotted module to its source file (or package dir)."""
    p = ROOT / mod.replace(".", "/")
    if p.with_suffix(".py").exists():
        return p.with_suffix(".py")
    if (p / "__init__.py").exists():
        return p / "__init__.py"
    if p.is_dir():                      # namespace package (examples has no __init__)
        return p
    return None


def _top_level(pyfile: Path) -> set[str]:
    tree = ast.parse(pyfile.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _name_exists(mod: str, name: str) -> bool:
    if _module_file(f"{mod}.{name}") is not None:   # submodule (e.g. lmkit.pretrain)
        return True
    f = _module_file(mod)
    if f is None or f.is_dir():
        return False
    return name in _top_level(f)


def _classdef(mod: str, name: str) -> ast.ClassDef | None:
    f = _module_file(mod)
    if f is None or f.is_dir():
        return None
    for node in ast.parse(f.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def _class_fields(cls: ast.ClassDef) -> set[str]:
    return {n.target.id for n in cls.body
            if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}


def _imports(blocks: list[str]) -> dict[str, str]:
    """{imported_name: module} for project imports across all README code blocks."""
    imported: dict[str, str] = {}
    for block in blocks:
        for node in ast.walk(ast.parse(block)):
            if isinstance(node, ast.ImportFrom) and node.module \
                    and node.module.split(".")[0] in PROJECT_PREFIXES:
                for alias in node.names:
                    imported[alias.asname or alias.name] = node.module
    return imported


# --- guards ------------------------------------------------------------------

def test_readme_python_imports_resolve():
    """Every `lmkit.*` / `examples.*` symbol imported in a README snippet exists."""
    for mod, name in ((m, n) for n, m in _imports(_python_blocks(_readme())).items()):
        assert _name_exists(mod, name), (
            f"README imports `{name}` from `{mod}`, which no longer exists. "
            f"Fix the import in README.md, or restore the symbol in "
            f"{mod.replace('.', '/')}.py")


def test_readme_dataclass_kwargs_are_fields():
    """Every keyword passed to an imported config class in a README snippet is a
    real field of that dataclass (catches a field rename)."""
    blocks = _python_blocks(_readme())
    imported = _imports(blocks)
    for block in blocks:
        for node in ast.walk(ast.parse(block)):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            cname = node.func.id
            if cname not in imported:
                continue
            cls = _classdef(imported[cname], cname)
            if cls is None:                      # a function, not a class; skip
                continue
            fields = _class_fields(cls)
            for kw in node.keywords:
                assert kw.arg in fields, (
                    f"README calls `{cname}({kw.arg}=...)` but `{cname}` in "
                    f"{imported[cname].replace('.', '/')}.py has no field "
                    f"`{kw.arg}`. Fix the README kwarg or rename the field.")


def test_readme_linked_and_repo_paths_exist():
    """Markdown links and repo-internal (`docs/…`, `examples/…`) paths named in
    the README point at files that exist."""
    text = _readme()
    refs = set(re.findall(r"\]\((?!https?://|#)([^)]+)\)", text))       # [x](path)
    refs |= set(re.findall(r"`((?:docs|examples)/[\w./-]+)`", text))    # `docs/…` `examples/…`
    for ref in refs:
        assert (ROOT / ref).exists(), (
            f"README references `{ref}`, which does not exist. Fix the path in "
            f"README.md or restore the file.")


def test_tracker_env_vars_match_code():
    """The env vars the README tells users to set are the ones the code reads."""
    obs = (ROOT / "lmkit" / "observability.py").read_text(encoding="utf-8")
    readme = _readme()
    for var in ("AIM_REPO", "MLFLOW_TRACKING_URI"):
        assert var in readme, f"README no longer documents `{var}`"
        assert var in obs, (
            f"README documents `{var}` but lmkit/observability.py no longer "
            f"reads it. Update the README or the code.")


def test_pip_extras_exist():
    """Each extra named in the README's `Extras:` line exists in pyproject."""
    opt = re.search(r"\[project\.optional-dependencies\](.*?)(\n\[|\Z)",
                    (ROOT / "pyproject.toml").read_text(encoding="utf-8"),
                    flags=re.DOTALL).group(1)
    keys = set(re.findall(r"^(\w[\w-]*)\s*=", opt, flags=re.MULTILINE))
    for extra in ("hub", "track", "dev"):
        assert extra in keys, (
            f"README's `Extras:` line names `{extra}`, absent from "
            f"pyproject.toml [project.optional-dependencies]. Fix one.")


def test_terminal_statuses_documented():
    """The MLflow terminal statuses the README lists are the ones the code sets."""
    src = ((ROOT / "lmkit" / "pretrain.py").read_text(encoding="utf-8")
           + (ROOT / "lmkit" / "observability.py").read_text(encoding="utf-8"))
    for status in ("KILLED", "FAILED", "FINISHED"):
        assert status in src, (
            f"README says the loop records `{status}`, but no lmkit source sets "
            f"it. Update the README's terminal-status bullet or the code.")


def test_cli_only_dispatches_quickstart():
    """The README claims the `lmkit` binary only runs `quickstart`."""
    cli = (ROOT / "lmkit" / "cli.py").read_text(encoding="utf-8")
    dispatched = set(re.findall(r'cmd == "(\w+)"', cli))
    assert dispatched == {"quickstart"}, (
        f"lmkit/cli.py now dispatches {dispatched or '{}'}, but the README says "
        f"the binary only runs `quickstart`. Update the 'Using the library' "
        f"section (and add a subcommand's own docs) or the CLI.")


if __name__ == "__main__":                         # runnable without pytest
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} drift guards passed")
