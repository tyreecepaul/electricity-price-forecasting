# Git and GitHub Conventions

## Branches

Two long-lived branches are maintained:

| Branch | Role | Protected |
| --- | --- | --- |
| `production` | Deployable and demo-ready; receives releases and hotfixes | Yes |
| `dev` | Integration branch and base for normal work | Yes |

Working branches start from an up-to-date `dev` and use:

```text
type/issue-number-short-summary
```

Valid types include `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, and
`spike`. Hotfixes start from `production` and use `hotfix/`.

```bash
git switch dev
git pull origin dev
git switch -c feat/36-experiment-pipeline
```

Use lowercase kebab-case summaries of two to five words. Every working branch
must include its GitHub issue number.

## Commits

Use [Conventional Commits](https://www.conventionalcommits.org/):

```text
type(scope): subject

Optional body explaining why the change was needed.
```

Common types are `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`,
`perf`, `ci`, and `revert`. Useful scopes for this repository include
`pipeline`, `data`, `preprocessing`, `models`, `evaluation`, `docs`, `tests`,
and `repo`.

Subjects must use imperative mood, start lowercase after the colon, contain no
trailing full stop, and remain within 72 characters. Avoid vague subjects such
as `update`, `fixed stuff`, `WIP`, or `final version`.

## Pull Requests

| Pull request | Head | Base |
| --- | --- | --- |
| Normal work | Working branch | `dev` |
| Release | `dev` | `production` |
| Hotfix | `hotfix/...` | `production`, followed by a back-merge to `dev` |

Use the same Conventional Commit format for the PR title. The description
should include:

- summary;
- related issues, such as `Closes #36`;
- changes;
- reproducible test steps;
- evidence when the change is user-facing; and
- a completed checklist.

Keep unfinished work as a draft. Before requesting review, read the complete
diff and run the relevant tests locally because GitHub Actions are disabled for
this repository. Keep each PR focused on one issue where practical.

PRs into `dev` require one approval. Releases and hotfixes into `production`
require two. Authors do not approve their own PRs, and every review comment
should receive a response. Respond to review requests within 24 hours where
possible, or tell the author when that is not possible.

Reviewers should check that the change:

- satisfies its linked issue;
- handles relevant edge cases and failure paths;
- is readable without undocumented context;
- includes behavioural tests where appropriate; and
- does not commit secrets, generated caches, or unrelated files.

Use **Request changes** for blocking problems, **Comment** for suggestions, and
**Approve** only when the change is ready to merge. Reviewers may push fixes to
the branch when appropriate, but are not required to do so. Authors should
reply to each comment, push fixes as new commits, and request review again.

## Keeping a Branch Current

Before review begins, rebase onto `dev`:

```bash
git fetch origin
git rebase origin/dev
git push --force-with-lease
```

After review begins, or when other work depends on the branch, merge instead:

```bash
git fetch origin
git merge origin/dev
```

## Merging and Releases

Use merge commits so individual commits remain visible. Delete working branches
after merge, and never squash a `dev` release into `production`.

For a release:

1. Open a PR from `dev` to `production` titled `release: vX.Y.Z`.
2. Summarise the release and obtain two approvals.
3. Merge with a merge commit.
4. Tag the merge using semantic versioning.

```bash
git switch production
git pull origin production
git tag -a v0.2.0 -m "Release v0.2.0"
git push origin v0.2.0
```

After merging a hotfix into `production`, immediately back-merge it into
`dev` so it is retained by the next release.
