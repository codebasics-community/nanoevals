## Summary

<!-- One or two sentences: what does this PR do? -->

## Related issue

Closes #<!-- issue number -->

## Changes

<!-- Bullet list of what changed and why. Be specific about files/classes touched. -->

-

## Checklist

- [ ] Targets the `dev` branch (not `main`)
- [ ] `ruff check nanoevals/` passes with no errors
- [ ] `ruff format nanoevals/` applied
- [ ] `pytest tests/ -m "not integration"` passes
- [ ] New behaviour is covered by tests (no LLM calls in unit tests)
- [ ] New evaluators expose `feedback_key` and `required_keys` class variables
- [ ] Docstrings updated for any new/changed public API
- [ ] No new mandatory runtime dependencies added
- [ ] `CONTRIBUTING.md` guidelines followed
