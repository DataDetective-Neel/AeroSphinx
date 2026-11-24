# Security policy

## What is in scope

This repository contains no credentials, no personal data and no user accounts.
The deployed application is a read-only dashboard over committed artefacts: it
has no write endpoints, no authentication and no database.

| Area | Report to us if... |
|---|---|
| The deployed API | A route does something other than return a committed artefact |
| The web front end | Script injection, or data leaving the page other than to `/api/*` |
| This repository | A secret has been committed, or a dependency has a known vulnerability |
| Data licensing | Committed provider data is redistributed beyond its terms |

## Reporting

Open a private security advisory on GitHub, or email the maintainer listed in
`CITATION.cff`. Please do not open a public issue for a live vulnerability.

We aim to acknowledge within three working days. There is no bug bounty.

## Not a security issue

* A forecast endpoint that abstains, or a dashboard panel that reports an
  artefact as unavailable. That is the intended behaviour: the platform states
  what it cannot support rather than inventing a value.
* A scientific disagreement. Those belong in an issue with the artefact quoted,
  and are genuinely welcome.
