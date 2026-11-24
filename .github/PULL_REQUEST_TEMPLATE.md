## What this changes

<!-- One paragraph. If a published number moved, say which artefact moved it. -->

## Why

<!-- The defect, report or question this addresses. Reference D-xxx if relevant. -->

## Checks

- [ ] `make verify` passes locally (tests, formatting, lint, published numbers)
- [ ] If a number in the README, API or dashboard changed: the artefact,
      `results/CANONICAL.json` and the prose moved in the same commit
- [ ] `legacy/` is untouched
- [ ] A test would fail if this change were reverted
- [ ] Formatting-only changes are in their own commit

## Evidence

<!-- Test counts, claim counts, a screenshot, or the command output you relied on. -->
