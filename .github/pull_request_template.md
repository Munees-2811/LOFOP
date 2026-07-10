## Summary

What does this change do, and why?

## Changes

- 

## Test plan

Commands you ran and their results:

```
ruff check lofop tests benchmarks examples
python -m pytest tests/ -q
```

## Checklist

- [ ] Tests added or updated for the change
- [ ] `ruff check` is clean
- [ ] `pytest` passes locally
- [ ] Docs updated (`MANUAL.md` / `docs/`) if user-facing
- [ ] `CHANGELOG.md` updated under "Unreleased"
- [ ] Public APIs remain backward compatible (or the break is justified above)
