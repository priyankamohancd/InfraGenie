"""
Shared, safe path bootstrap for this backend's two cross-repo imports:
`shared.*` (arch2tf-product/shared/) and `arch2terraform.*` (the separate
arch2terraform repo).

Local dev: this file lives at arch2tf-product/backend/app/_pathboot.py,
which is two levels below the "thesis" root (thesis/arch2tf-product/) and
three levels below it (thesis/), where the separate arch2terraform repo
sits as a sibling (thesis/arch2terraform/src/arch2terraform/). Neither
`shared` nor `arch2terraform` is importable without adding both of those
directories to sys.path — see ensure_paths() below.

Docker: the Dockerfile (see its module-level comment) intentionally
flattens all of this — `shared/` and arch2terraform's package contents are
copied directly under /app, and PYTHONPATH=/app already makes both
importable with zero extra sys.path entries. Walking this file's parents
the same number of levels doesn't reach anything meaningful at that depth
inside the container (parents[2]/[3] land on "/" or go out of range
entirely) — ensure_paths() is written to recognize that and no-op rather
than crash, which is what every individual sys.path.insert(0,
Path(__file__).resolve().parents[N]) call across this codebase used to do
before it was centralized here (each was calibrated only for the local-dev
depth and raised IndexError the first time it actually ran inside a
container — see the 2026-07-30 EC2 deploy).

Every module that needs `shared.*` or `arch2terraform.*` should call
ensure_paths() before importing them, instead of doing its own
Path(__file__).resolve().parents[N] arithmetic. Safe to call from multiple
modules — each insert is guarded so it only happens once and only when the
target directory actually exists.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _parent(path: Path, n: int) -> Path | None:
    """Path(...).parents[n], or None if that many levels don't exist."""
    try:
        return path.resolve().parents[n]
    except IndexError:
        return None


def ensure_paths() -> None:
    here = Path(__file__)  # arch2tf-product/backend/app/_pathboot.py (or /app/app/_pathboot.py)

    # thesis/arch2tf-product/ — makes `shared.*` importable. In the
    # container this resolves to "/", which correctly fails the .is_dir()
    # check below (no /shared at the filesystem root), so nothing is
    # inserted — /app/shared is already reachable via PYTHONPATH=/app.
    product_root = _parent(here, 2)
    if product_root is not None and (product_root / "shared").is_dir():
        p = str(product_root)
        if p not in sys.path:
            sys.path.insert(0, p)

    # thesis/arch2terraform/src — makes `arch2terraform.*` importable. In
    # the container this either goes out of range (caught, returns None)
    # or, even if it resolved, wouldn't contain an "arch2terraform/src"
    # subdirectory (the container's copy of arch2terraform lives directly
    # at /app/arch2terraform, no "src" level) — so this is a no-op there,
    # same reasoning as above.
    thesis_root = _parent(here, 3)
    if thesis_root is not None:
        arch2tf_src = thesis_root / "arch2terraform" / "src"
        if arch2tf_src.is_dir():
            p = str(arch2tf_src)
            if p not in sys.path:
                sys.path.insert(0, p)
