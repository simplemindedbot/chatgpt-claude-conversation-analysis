# GitHub Issue Creator — Authoring Guide

This guide explains how to write Markdown so that `github_issue_creator.py` can parse your plans and automatically create well‑labeled GitHub issues. It includes concrete, copy‑pasteable examples for each supported pattern.

The script reads, by default:
- `development_plan.md`
- `data-mining-recommendations.md`

Optionally, it can also extract TODO checkboxes from any other file you include via `--files`.

---

## Quick Usage

```bash
# Preview (no network calls)
python github_issue_creator.py --repo <owner>/<repo> --dry-run

# Create issues (requires a token)
export GITHUB_TOKEN=ghp_your_token
python github_issue_creator.py --repo <owner>/<repo>

# Assign and create milestones per Phase
python github_issue_creator.py --repo <owner>/<repo> \
  --assignee <your-gh-username> --milestones-per-phase
```

Key properties:
- Idempotent by issue title (skips if the same title already exists unless `--force`).
- Creates missing labels automatically (e.g., `Phase 2`, `Priority 1`, `Docs`, `Analysis`).
- Skips completed tasks marked with `✓` in `development_plan.md`.

---

## Formatting Rules by Source Document

### A) development_plan.md

The parser expects Phase sections introduced by a third‑level heading (`###`). Under each Phase, add bullet items beginning with `-`. You can optionally mark item status at the end of the bullet line using:
- `✓` complete (will be skipped)
- `*` in‑progress (will be included)
- `!` failed (will be included)

The parser also looks for a special section `### Documentation Updates Needed` anywhere in the file; bullets there will become `Docs` issues regardless of phase.

#### Minimal Phase Example

```md
### Phase 2 (Stabilization)

- Improve CLI help output *
- Add error handling for missing spaCy model !
- Update Quick Start docs ✓
```

Produces issues:
- [Phase 2] Improve CLI help output (labels: `Phase 2`)
- [Phase 2] Add error handling for missing spaCy model (labels: `Phase 2`)
- The “Update Quick Start docs ✓” line is skipped because it is marked complete.

Heuristic extra labels based on content:
- Contains any of: `doc`, `readme`, `troubleshooting` → adds label `Docs`.
- Contains any of: `test`, `unit test`, `ci` → adds label `Testing`.
- Contains any of: `cli`, `argparse`, `process_runner` → adds label `CLI`.
- Contains any of: `cluster`, `topics`, `knowledge graph`, `code snippet`, `similarity`, `search`, `gap analysis` → adds label `Analysis`.

#### Documentation Updates Needed Example

```md
### Documentation Updates Needed
- Add NLTK VADER install step to README
- Clarify Python version policy ✓
```

Produces issues:
- [Docs] Add NLTK VADER install step to README (labels: `Docs`)
- The “Clarify Python version policy ✓” line is skipped.

Notes:
- Status markers go at the end of the bullet line (after the text), e.g., `- Fix X ✓`.
- Indentation can be 0–4 spaces before `- `; only single‑level bullets are considered for issues.

---

### B) data-mining-recommendations.md

The parser scans `#### Priority N` sections (fourth‑level headings). Inside each Priority block, it looks for numbered items whose titles are bold, then includes the following sub‑bullets in the issue body for context.

#### Priority Block Example

```md
#### Priority 1 - Immediate Value

1. **Personal Knowledge Graph**
   - Use semantic embeddings to create topic clusters
   - Build search utility over conversations

2. **Learning Gap Analysis**
   - Identify topics with high question density but low explanation sentiment

#### Priority 2 - Advanced Analytics

1. **Productivity Timeline**
   - Monthly aggregates and trends
```

Produces issues:
- [Priority 1] Personal Knowledge Graph (labels: `Priority 1`, `Analysis`)
- [Priority 1] Learning Gap Analysis (labels: `Priority 1`, `Analysis`)
- [Priority 2] Productivity Timeline (labels: `Priority 2`, `Analysis`)

Formatting details:
- The numbered item’s title must be bold: `1. **Title**`.
- Any sub‑bullets following the numbered title (until the next numbered title) will be copied into the issue body as scope/context.

---

### C) Generic TODO Checkboxes (any file)

For files other than the two primary docs, you can still generate issues by using GitHub‑style TODO checkboxes. Include the file in `--files` and use unchecked checkboxes `- [ ]`.

```md
- [ ] Add `--dry-run` example to README
- [ ] Document environment variables
```

Produces issues:
- [TODO] Add `--dry-run` example to README (labels: `Todo`)
- [TODO] Document environment variables (labels: `Todo`)

Note: Checked items `- [x]` are ignored by the parser.

---

## Labels and Milestones

- Labels are inferred from Phase/Priority and heuristics; missing labels are created automatically.
- If you pass `--milestones-per-phase`, a milestone named exactly like the `Phase N` label will be created (if missing), and all `Phase N` issues will be attached to it.

---

## Idempotency and De‑duplication

- The script will not create a new issue if another issue with the exact same title already exists (open or closed), unless you pass `--force`.
- Within a single run, duplicate titles are de‑duplicated automatically.

---

## Troubleshooting

- “Token not found”: Set your token via `export GITHUB_TOKEN=...` (or another env name via `--token-env`).
- “requests package not available”: Install `requests` (`pip install requests`) or use `--dry-run`.
- No issues created: Ensure your Markdown formatting matches the patterns in this guide (headings, bold numbered titles, status markers).

---

## Authoring Checklist

- [ ] Phases use `### Phase N ...` headings.
- [ ] Phase tasks are top‑level bullets starting with `- ` and optional status marker at the end (`✓`, `*`, `!`).
- [ ] Documentation‑specific items live under `### Documentation Updates Needed`.
- [ ] Recommendations use `#### Priority N` sections with numbered items formatted `1. **Title**` followed by sub‑bullets.
- [ ] Optional: Unchecked TODO boxes `- [ ] ...` in other files.

With these conventions, `github_issue_creator.py` can turn your roadmap into a structured GitHub issue tracker with consistent labels and optional milestones.