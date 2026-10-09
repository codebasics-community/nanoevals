# Contributing to nanoevals

Thanks for your interest in improving **nanoevals**.  
This document covers everything you need to submit a high-quality contribution.

---

## Table of Contents

- [Ground Rules](#ground-rules)
- [Getting Started](#getting-started)
- [Branching Strategy](#branching-strategy)
- [Code Style](#code-style)
- [Testing](#testing)
- [Commit Conventions](#commit-conventions)
- [Pull Request Process](#pull-request-process)
- [What We Accept](#what-we-accept)
- [What We Don't Accept](#what-we-dont-accept)

---

## Ground Rules

- Be direct and respectful. Read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
- Keep changes **focused**. One logical change per PR.
- **No new mandatory dependencies.** The library must install with zero transitive deps. Optional deps go in `[project.optional-dependencies]`.
- All public behaviour must be covered by tests before a PR can merge.
- **Never push directly to `main` or `dev`.** Branch protection is enforced server-side — pushes will be rejected.  
  PRs are the only path to either branch.

---

## Getting Started

### 1. Fork and clone

```bash
# Fork https://github.com/codebasics-community/nanoevals first, then:
git clone https://github.com/<your-username>/nanoevals.git
cd nanoevals
```

### 2. Set up the environment

[Install uv](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then sync the project (creates `.venv` and installs all deps including the `dev` group):

```bash
uv sync --extra dev
```

You don't need to activate the venv — prefix any command with `uv run` and it will use the project environment automatically.

### 3. Verify the setup

```bash
uv run pytest tests/ -v -m "not integration"
uv run ruff check nanoevals/
```

Both commands should exit cleanly before you start making changes.

---

## Branching Strategy

| Branch        | Purpose                                              |
|---------------|------------------------------------------------------|
| `main`        | Stable, tagged releases only. Never commit directly. |
| `dev`         | Integration branch. PRs target here first.           |
| `feat/<name>` | New evaluators or features (`feat/cosine-similarity`). |
| `fix/<name>`  | Bug fixes (`fix/json-parse-edge-case`).              |
| `docs/<name>` | Documentation only (`docs/update-api-reference`).   |
| `chore/<name>`| Tooling, CI, dependencies (`chore/add-ruff`).        |

**Always branch off `dev`, not `main`.**

```bash
git checkout dev
git pull origin dev
git checkout -b feat/your-feature-name
```

---

## Code Style

We use **[Ruff](https://docs.astral.sh/ruff/)** for linting and formatting.

```bash
# Check
uv run ruff check nanoevals/

# Auto-fix safe issues
uv run ruff check --fix nanoevals/

# Format
uv run ruff format nanoevals/
```

### Style conventions

- **Small, named functions** — if a function needs a comment to explain what it does, split it.
- **Explicit type annotations** at all public boundaries (evaluator factories, base classes, public methods).
- **`async/await` only** — no callbacks or `.then()` chains.
- **Explicit error handling** — catch specific exceptions; never swallow errors silently.
- New evaluators must expose `feedback_key` and `required_keys` class variables (see `BaseEvaluator`).

---

## Testing

```bash
# Unit tests — no API key required
uv run pytest tests/ -v -m "not integration"

# Integration tests — requires GROQ_API_KEY (or your preferred LLM key)
GROQ_API_KEY=sk-... uv run pytest tests/ -v
```

### Test conventions

- **No LLM calls in unit tests.** Mock the LLM callable with a plain function that returns a fixed string.
- Each new evaluator factory or base class must have at least one unit test.
- Tests go in `tests/` and follow the `test_<module>.py` naming convention.
- Mark tests that require a live API key with `@pytest.mark.integration`.

---

## Commit Conventions

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add cosine similarity evaluator
fix: handle non-JSON LLM responses in _parser
docs: update CONTRIBUTING with test conventions
chore: bump ruff to 0.4
test: add unit tests for trajectory_match subset mode
```

Use the imperative mood in the subject line. Keep it under 72 characters.

---

## Pull Request Process

1. Target the **`dev` branch**, never `main`.
2. Fill in the PR template completely — sparse descriptions will be sent back.
3. Ensure the checklist in the template is fully checked before requesting review.
4. One reviewer approval is required to merge into `dev`.
5. Maintainers merge `dev` → `main` during release cuts.

---

## What We Accept

- **New evaluator factories** — e.g. `create_rouge_score`, `create_bleu_score`.
- **New base class variants** — e.g. a streaming-aware evaluator base.
- **New built-in prompts** — under `nanoevals/prompts/`. Must be general-purpose.
- **Bug fixes** — with a regression test.
- **Performance improvements** — for the async runner or parser.
- **Documentation fixes** — typos, clarifications, better examples.
- **pytest plugin improvements** — output formatting, new fixtures.

---

## What We Don't Accept

- **New mandatory runtime dependencies.** If your evaluator needs a library, it must be optional.
- **Opinionated prompt rewrites** — prompts are intentionally minimal; don't add brand-specific framing.
- **LLM-provider-specific integrations** — nanoevals is provider-agnostic by design.
- **Breaking changes to the public API** without a prior discussion in an issue.
- **Large refactors** without prior discussion.

If you're unsure whether your idea fits, open a **Feature Request** issue first and get a thumbs-up before spending time on the implementation.
