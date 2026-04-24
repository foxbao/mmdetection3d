# Repository Guidelines

## Project Structure & Module Organization
- `mmdet3d/`: core library code (models, datasets, engine hooks, evaluation, visualization, and utilities).
- `configs/`: model and dataset configs. Filenames encode model, schedule, and data scope, for example `pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py`.
- `tools/`: runnable entry points such as `train.py`, `test.py`, `dist_train.sh`, `dist_test.sh`, and dataset converters.
- `tests/`: unit and integration tests grouped by feature area (`test_models`, `test_datasets`, `test_engine`, etc.).
- `projects/`: incubating project-specific extensions (for example BEVFusion, DSVT, DETR3D).
- `docs/en` and `docs/zh_cn`: documentation sources. `demo/` contains inference examples.

## Build, Test, and Development Commands
```bash
pip install -v -e .                          # editable install
pip install -r requirements/tests.txt        # test/lint dependencies
python tools/train.py <config> --work-dir work_dirs/<exp>
./tools/dist_train.sh <config> <gpus>
python tools/test.py <config> <checkpoint>
./tools/dist_test.sh <config> <checkpoint> <gpus>
pytest tests/                                # run unit tests
coverage run --branch --source mmdet3d -m pytest tests/
pre-commit run --all-files
```

## Coding Style & Naming Conventions
- Follow Python 4-space indentation and PEP8-oriented formatting via `yapf`.
- Keep imports `isort`-clean and code `flake8`-clean; run `pre-commit` before committing.
- Prefer ~79 character lines (consistent with `isort`/`docformatter` settings).
- Use descriptive `snake_case` for Python symbols and `test_*.py` for test modules.
- Keep config names explicit and pattern-based: `<model>_<batchxgpu>-<schedule>_<dataset>-<task>.py`.

## Testing Guidelines
- Add or update tests in `tests/` for every behavior change or bug fix.
- Run targeted tests during development, for example:
  `pytest tests/test_models/test_detectors/test_voxelnet.py -k <keyword>`.
- Before opening a PR, run `pytest tests/`; for larger changes, also run coverage locally.
- CI runs `pytest` and coverage reporting, so missing tests for touched logic will block review.

## Commit & Pull Request Guidelines
- Match existing commit style prefixes where applicable: `[Fix]`, `[Feature]`, `[Docs]`, `[CI]`, `[WIP]`.
- Keep one focused change per branch and PR.
- Complete the PR template sections: Motivation, Modification, BC-breaking (if any), and Use cases.
- Link related issues, summarize validation performed, and update docs when user-facing behavior changes.
- Ensure `pre-commit` and relevant tests pass before requesting review.

## Data & Artifact Hygiene
- Do not commit generated artifacts or large local data (`work_dirs/`, checkpoints, prediction dumps, dataset copies).
- Keep dataset and output paths configurable in config files; avoid hardcoded machine-specific absolute paths.
