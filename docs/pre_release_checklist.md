# Pre-release checklist

- Test suite passes:

```bash
pytest
```

- Lint and format checks pass:

```bash
ruff check .
ruff format --check .
```

- Demo CLI command runs from a clean checkout:

```bash
python -m neweds.cli examples/demo_timeseries.csv --output-dir outputs/demo
```

- Generated files are not committed:

```bash
git status --short --ignored
```

Check that `__pycache__`, `.pytest_cache`, `.ruff_cache`, `outputs/`, reports and local data files are absent from tracked changes.

- README commands are still correct.
- No large local datasets are staged.
- No private paths, usernames, tokens or absolute machine-specific paths are present.
- License status is explicit in `README.md`.
- `examples/run_demo.sh` has executable mode in Git when publishing from a Unix-like environment.
