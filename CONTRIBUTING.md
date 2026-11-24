# Contributing

This repository publishes numbers that other people cite, so the rules are
about keeping claims and code in step rather than about style preferences.

## Before you change anything

```bash
make install-dev     # package + dev tools in .venv (created if absent)
make verify          # tests, formatting, lint and the published numbers
```

`make verify` is what CI runs. If it passes locally it will pass there.

## The four rules

1. **Never edit `legacy/`.** It is the frozen pre-Phase-0 record, including the
   single-script pipeline that produced the book chapter. Its numbers are
   history; correcting history destroys the audit trail. Fix forward in `src/`.
2. **A published number has one home.** Changing what the README, the API or the
   dashboard says about a measurement means updating the artefact, the entry in
   `results/CANONICAL.json` and the prose in the *same commit*. `make check-claims`
   enforces it.
3. **Reports are append-only.** Phase reports and audits record the state at the
   time. A correction is a dated revision section at the end, never an edit to
   the original text.
4. **No new availability assumptions.** Anything a model consumes must be legal
   at its issue time under `docs/reference/FEATURE_AVAILABILITY_CONTRACT.md`.
   Future observations, however convenient, are leakage.

## Adding a feature or a model

* Put the rule in `config/`, not in code, when it is a scientific choice
  (threshold, window, baseline).
* Prove it with a test that would fail if the old behaviour came back.
* A new model joins the M0–M8 matrix as a configuration; it does not get its own
  bespoke evaluation. Otherwise the comparison is not like-for-like.

## Style

* Python: `black` (88 columns), `isort` (black profile), `ruff` for errors.
  `make fmt` applies the first two.
* TypeScript: `npm run typecheck`, `npm run lint` in `ui/`.
* Commit messages: an imperative subject under 72 characters, then *why*, then
  what it changes. Reference defect IDs (`D-002`) when relevant.

## Pull requests

- Keep the formatting-only changes in their own commit, so a reviewer can skip
  them.
- Say what you ran. "Verified from a clean clone: 266 tests, 13/13 claims" is
  worth more than a description of intent.
- If a number moved, say which artefact moved it and why.

## Reporting a problem

Use the issue templates. For a scientific objection, quote the artefact you are
looking at — a reviewer cannot act on "this looks wrong" any more than you can.
