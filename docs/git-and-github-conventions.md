# Git and GitHub conventions

[Project README](../README.md) · [Documentation index](README.md)

## Branches

Use `dev` as the integration branch and the base for normal pull requests.
The repository also has `main`; target it only for an agreed release.

Start new work from an up-to-date `dev`:

```bash
git switch dev
git pull --ff-only origin dev
git switch -c feat/36-experiment-pipeline
```

Use `type/issue-number-short-summary` where an issue exists. Valid types
include `feat`, `fix`, `docs`, `refactor`, `test`, `chore` and `spike`.
Use lowercase summaries and include the issue number in the PR description.

## Commits and pull requests

Use a consistent `type(scope): subject` format, for example:

```text
fix(pipeline): preserve sequence context during validation
```

Write an imperative subject that describes the change, ideally within
72 characters. Add a body when the reason or tradeoff is not clear from the
diff.

Open normal PRs into `dev` and keep unfinished work as drafts. A useful
description states:

- the problem and resulting behaviour;
- linked issues, such as `Closes #36`;
- relevant test commands and results;
- any remaining limitations or follow-up work.

Read the complete diff before requesting review. Run the relevant checks
locally and inspect any GitHub status checks; do not assume automation has
verified the change. Avoid unrelated edits, generated caches and secrets.

The team's review convention is one approval for integration work and two
for releases. Authors should not approve their own PRs and should respond to
actionable review comments.

## Keeping a branch current

For a branch only you are using, coordinate before rebasing if it has already
been reviewed:

```bash
git fetch origin
git rebase origin/dev
git push --force-with-lease
```

If teammates depend on the branch, merge `dev` into it instead:

```bash
git fetch origin
git merge origin/dev
git push
```

Resolve conflicts, rerun relevant checks and review the final diff before
pushing. A force push rewrites remote history; avoid rebasing shared branches
without agreement.

## Merging and releases

Use merge commits to retain individual commits and delete working branches
after merge when they are no longer needed. For an agreed release, open a PR
from `dev` to `main`, document the release and tag the merged commit if the
team is using versioned releases.

A release hotfix should start from the release branch and be brought back
into `dev` after merging, so the next release retains it.
