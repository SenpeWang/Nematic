# Local Mamba3-only Package

This directory is the local `mamba_ssm` package used by the Nematic project.
It was intentionally reduced to a Mamba3-only layout on 2026-04-04 so that
state-space experiments can be done directly on the local project copy.

Current intent:
- The project should use the local `Mamba3` implementation in this folder.
- Unrelated Mamba1 / Mamba2 / model-zoo / distributed utilities were removed.
- Future state-space modifications should be made directly in `modules/mamba3.py`
  and its Mamba3 kernel dependencies.

Project entrypoints currently using this package:
- `networks/block.py` imports `Mamba3` from `mamba_ssm.modules.mamba3`
- `mamba_ssm/__init__.py` exports only `Mamba3`

Kept in this trimmed package:
- `modules/mamba3.py`
- `ops/triton/mamba3/`
- `ops/triton/layernorm_gated.py`
- `ops/triton/angle_cumsum.py`
- `ops/cute/mamba3/`
- `ops/tilelang/mamba3/`
- package `__init__.py` files required for imports

Removed from this trimmed package:
- Mamba1 / Mamba2 module implementations
- model wrappers such as `MambaLMHeadModel`
- distributed helpers
- non-Mamba3 selective scan / SSD files
- stale `__pycache__` and `.pyc` files

Recovery / history:
- Import-layer backup: `/home/wangshengping/04_Nero/code/Nematic/.codex_backups/20260404_mamba3_only`
- Full pre-prune package backup: `/home/wangshengping/04_Nero/code/Nematic/.codex_backups/20260404_mamba3_prune`
