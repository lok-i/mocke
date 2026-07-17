#!/usr/bin/env bash
# install.sh — pip install mocke, then satisfy the rsl_rl fork pin ([tool.mocke]
# in pyproject.toml) without clobbering a newer fork already in the env:
#   installed rsl_rl is a git checkout whose HEAD contains the pinned SHA
#   (equal or descendant) -> leave it; anything else (PyPI wheel, older/foreign
#   checkout, not installed) -> pip install --no-deps the pin.
# The pin bypasses the resolver on purpose: mjlab pins rsl-rl-lib==5.4.0 (PyPI),
# the fork reports 5.4.1 — co-resolution is impossible.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# mjlab already in the env -> --no-deps, or the resolver downgrades the rsl_rl
# fork to mjlab's PyPI pin (rsl-rl-lib==5.4.0) on every run. Fresh env -> full
# install (mjlab + its PyPI rsl-rl-lib), pin step below then swaps in the fork.
if python -c "import mjlab" 2>/dev/null; then
    pip install --no-deps -e "$ROOT"
else
    pip install -e "$ROOT"
fi

PIN=$(python -c "
import tomllib
print(tomllib.load(open('$ROOT/pyproject.toml','rb'))['tool']['mocke']['rsl_rl'])")
SHA="${PIN##*@}"

# pip git-snapshot already at the pinned SHA (direct_url.json) -> nothing to do
INSTALLED_SHA=$(python -c "
import importlib.metadata as m, json
print(json.loads(m.distribution('rsl-rl-lib').read_text('direct_url.json') or '{}')
      .get('vcs_info', {}).get('commit_id', ''))" 2>/dev/null || true)
if [ -n "$INSTALLED_SHA" ] && [ "$INSTALLED_SHA" = "$SHA" ]; then
    echo "[install] rsl_rl pip snapshot already at pin ${SHA:0:12} — leaving as-is"
    exit 0
fi

# locate the installed rsl_rl package's checkout root (if any)
LOC=$(python -c "
import rsl_rl, pathlib
print(pathlib.Path(rsl_rl.__file__).parent.parent)" 2>/dev/null || true)

if [ -n "$LOC" ] && git -C "$LOC" rev-parse --is-inside-work-tree &>/dev/null; then
    # make sure the pinned SHA is known locally before the ancestry test
    git -C "$LOC" cat-file -e "$SHA" 2>/dev/null || git -C "$LOC" fetch -q origin "$SHA" 2>/dev/null || true
    if git -C "$LOC" merge-base --is-ancestor "$SHA" HEAD 2>/dev/null; then
        echo "[install] rsl_rl @ $(git -C "$LOC" rev-parse --short HEAD) ($LOC) contains pin ${SHA:0:12} — leaving as-is"
        exit 0
    fi
    echo "[install] rsl_rl checkout at $LOC does not contain pin ${SHA:0:12} — installing pin"
fi

pip install --no-deps "$PIN"
