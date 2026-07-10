# Contributing to LOFOP

Thanks for your interest in improving LOFOP. This guide covers how to set up a
development environment, the quality gates every change must pass, and how to
propose changes.

## Development setup

LOFOP has an optional native C++ ops library and optional PyTorch/ONNX
dependencies. For full development, install the test extras and build the
native ops:

```bash
python -m pip install -e ".[dev,models,deploy]"
python -c "from lofop.ops import build_native, backend; build_native(); print(backend())"
```

The core, data, and CLI layers are torch-free by design, so most contributions
can be developed and tested without a GPU.

## Quality gates

Every change must pass the same checks CI runs:

```bash
ruff check lofop tests benchmarks examples   # lint
python -m pytest tests/ -q                    # tests
python scripts/check_version_sync.py          # version consistency
```

- **Tests are required.** Every new module or behavior needs unit tests. Use
  the project test base (`from torch.testing._internal` is not used here; see
  existing tests for the plain `unittest`/`pytest` style). Device-generic
  numerics should be tested on CPU.
- **Lint must be clean.** Run `ruff check` (and `ruff check --fix` for
  autofixable issues) before committing.
- **Backward compatibility.** Public APIs (anything exported from a package
  `__init__`) must not break without a clear reason and a CHANGELOG entry. New
  dataclass fields should be defaulted.
- **Docs.** Update `MANUAL.md` and the relevant file under `docs/` when you add
  or change user-facing behavior.

## Coding style

- Type hints on public functions; concise, self-documenting code.
- Match the surrounding style and architectural patterns. Prefer reusing
  existing abstractions (registry, event bus, config) over new ones.
- ASCII only in new code comments.
- Keep the core/data/CLI layers importable without torch; put anything that
  needs PyTorch in the models/training/deploy layers.

## Proposing changes

1. Branch off the development branch and make focused, reviewable commits.
2. Add tests and docs alongside the code.
3. Add a `CHANGELOG.md` entry under "Unreleased".
4. Open a pull request describing the change, the motivation, and the exact
   test commands you ran (see the PR template).

## Reporting bugs and requesting features

Use the issue templates. A good bug report includes the LOFOP version
(`lofop version`), the output of `lofop doctor`, and a minimal reproduction.

By contributing, you agree that your contributions are licensed under the
project's Apache-2.0 license.
