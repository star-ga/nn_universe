# Git history scrub — DEFERRED (requires explicit force-push approval)

**Status:** Audit C7 flagged commit `ac4307f` as exposing `ceo@star.ga`.
This is in git history; cleaning it requires a destructive rewrite + force-push.

## Why this is deferred

- Force-push to `main` rewrites every cloned copy's history. Any open PR or
  collaborator clone breaks.
- The current head is fully anonymised (no STARGA / Nedovodin / star.ga in any
  tracked file as of commit after this note).
- For NeurIPS double-blind, what reviewers see is the submitted PDF, not the
  GitHub repo. The repo URL itself (`github.com/star-ga/...`) leaks identity
  — that's the more fundamental issue, and it can't be fixed by history scrub
  (you'd need to rename the org or fork to a fresh anonymous account).

## When to actually run this

Only if you decide to make the repo public during anonymous review.
NeurIPS reviewers can be told the repo URL post-acceptance; until then the
repo is one identity-leak you cannot scrub by rewriting commits.

## Recipe (when you decide to)

```bash
# Install git-filter-repo if missing
pip install --user git-filter-repo

# 1. Make a backup clone first
git clone --mirror https://github.com/star-ga/nn_universe.git nn_universe-backup.git

# 2. Run filter-repo on a fresh clone
git clone https://github.com/star-ga/nn_universe.git nn_universe-clean
cd nn_universe-clean

# 3. Replace identity strings in every commit message + file
cat > /tmp/replacements.txt <<'REPL'
ceo@star.ga==>anonymous@example.org
star.ga==>example.org
STARGA, Inc.==>Anonymous
STARGA Inc.==>Anonymous
STARGA==>Anonymous
Nikolai Nedovodin==>Anonymous Authors
Nedovodin==>Anonymous
Nikolai==>Anonymous
REPL

git filter-repo --replace-text /tmp/replacements.txt

# 4. Force-push (DESTRUCTIVE)
git remote add origin https://github.com/star-ga/nn_universe.git
git push --force --all
git push --force --tags

# 5. Tell collaborators to re-clone
```

## Decision

Default: keep history as-is. The submitted paper.pdf is fully anonymous,
and the GitHub URL itself already reveals identity (org name `star-ga`).
History scrub doesn't add meaningful protection.
