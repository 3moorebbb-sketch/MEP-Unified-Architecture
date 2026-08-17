## chore: buffer/device fixes, preallocated trajectory, add reqs/LICENSE

This Draft PR contains a small set of targeted fixes and additions to improve
device handling, memory behavior, and reproducibility for the core PyTorch
module `mep_thermodynamic_core.py`.

Summary of changes
- Fix: `set_problem_topology` now copies adjacency matrices into the registered
  buffer `self.J` in-place, preserving the buffer's device and dtype.
- Perf: `ThermodynamicIsingNetwork.forward` preallocates an output trajectory
  tensor instead of growing a Python list and stacking, improving performance
  and reducing memory fragmentation.
- Perf: `HierarchicalThermodynamicNetwork.evolve_core_physics` optimized to
  integrate only free nodes (pretransposed J, preallocated x_full) to reduce
  allocations and per-step compute.
- Fix: Device/dtype-aware tensor creation and broadcasting adjustments across
  thermodynamic modules to avoid CPU/GPU mismatch errors.
- Feature: Add `requirements.txt` (pinned torch & numpy), CC0 `LICENSE`.
- Add: Lightweight CLI `run_mep.py` to run the smoke tests with --device/--seed
  flags; `SMOKE_TESTS.md` documents usage.
- Add: Basic unit tests for helper functions in `tests/test_helpers.py`.

Why
- Replacing registered buffers (e.g., `self.J = ...`) can break module
  behavior when moving models across devices or when saving/loading state.
- Preallocating trajectory reduces Python-level memory churn and is friendly to
  GPUs where repeated small allocations are expensive.
- Evolving only free nodes reduces runtime and memory use while keeping end-to-
  end gradients.

Notes
- The patch purposefully avoids changing the numeric dynamics or training
  algorithms. All changes are interface/compatibility/perf improvements.

Checklist (manual)
- [ ] Run `python -m pytest tests` on CPU to validate the helper tests.
- [ ] Run `python run_mep.py --which all --device cpu` to smoke test.
- [ ] If running on GPU, confirm `--device cuda` and watch memory usage.

