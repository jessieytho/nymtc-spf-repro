# Publishing this repo to GitHub — step by step

Written for a first-time publisher. Estimated time: 30–60 min the first time.

---

## ⚠️ READ FIRST: double-blind review and anonymity

The paper is going to **double-blind review** (TRB 2027; submission ~Aug 1, 2026). The manuscript cites an
**anonymized** code capsule for review and a **public DOI on acceptance** — on purpose.

**Do not make a public GitHub repo under your own name/handle during review.** A public, name-linked repo
(or your profile, commit emails, prior repos) can deanonymize you, which is grounds for desk rejection.

Use this timeline instead:

| Phase | What to publish | How |
|---|---|---|
| **Now → submission** | a **PRIVATE** GitHub repo (set everything up, push, let CI run) | §2–§5 below, choosing *Private* |
| **For the review PDF** | an **anonymized** read-only capsule | §6 — `anonymous.4open.science` |
| **On acceptance** | flip the repo to **Public** + mint a **DOI** | §7 — GitHub release + Zenodo |

A private repo still runs GitHub Actions (limited free minutes, plenty for this suite), so you get CI now
without exposing anything.

---

## 1. Pre-flight checklist (before the first commit)

- [ ] `LICENSE` present and the code license is what you want (see `LICENSE`; MIT is the included default).
- [ ] **Data rights.** `./data` ships only publicly accessible ITSMR tables and the authors' own
      derived county-year aggregates; the official state DVMT exposure workbook is not redistributed
      (user-supplied & optional). No redistribution permission is needed for what ships. (See `DOCS.md`
      §9 and `data/README.md`.)
- [ ] **No secrets** anywhere (API keys, tokens, personal emails in code/notebooks). The corresponding-author
      email belongs in the paper, not the repo.
- [ ] `__pycache__/` and `*.pyc` are git-ignored (they are — see `.gitignore`).
- [ ] Tests pass locally: `python tests/run_tests.py` → `ALL TESTS PASSED`.
- [ ] Decide whether to commit `outputs/` and `figures/`. Committing them lets reviewers see expected output
      without running anything (recommended for a repro artifact). If you'd rather keep the repo lean, add
      `outputs/` and `figures/` to `.gitignore` — but then a reviewer must run `run_all.py` to see results.

---

## 2. One-time local setup

Install Git (https://git-scm.com/downloads), then set your identity once:
```
git config --global user.name  "Your Name"
git config --global user.email "you@example.com"
```
For double-blind: consider using a GitHub **noreply** email so your real address isn't in commit metadata
(GitHub → Settings → Emails → "Keep my email addresses private"; then use the `...@users.noreply.github.com`
address shown).

## 3. Initialize and make the first commit

From inside the repo folder:
```
git init
git add .
git status            # sanity-check: no __pycache__, no secrets, no stray large files
git commit -m "Initial commit: reproducible SPF artifact (NYMTC From Counts to Rates)"
```

## 4. Create the empty repo on GitHub

**Web UI:** GitHub → New repository → name `nymtc-spf-repro` → **Private** (for now) → do **not** add a
README/license/.gitignore (you already have them) → Create.

**Or with the GitHub CLI** (`gh`, https://cli.github.com):
```
gh repo create nymtc-spf-repro --private --source=. --remote=origin
```

## 5. Connect and push

If you used the web UI, copy the repo URL it shows, then:
```
git branch -M main
git remote add origin https://github.com/<your-username>/nymtc-spf-repro.git
git push -u origin main
```
(First push over HTTPS will prompt for a **Personal Access Token** as the password — GitHub → Settings →
Developer settings → Personal access tokens. Or use SSH / `gh auth login`.)

Then on GitHub open the **Actions** tab and confirm the `ci` workflow runs green (test on 3.11/3.12, docker
build, lint). Green CI on a fresh clone is the strongest reproducibility signal you can show a reviewer.

---

## 6. Anonymized capsule for peer review

Keep the GitHub repo private and create a read-only anonymized mirror for the review PDF:

1. Go to `https://anonymous.4open.science` and follow "Anonymize a repository."
2. Point it at your repo (you grant temporary read access; it strips identifying metadata and serves a
   double-blind-safe view).
3. It returns a URL like `https://anonymous.4open.science/r/nymtc-spf-repro-XXXX`.
4. Put that URL in the manuscript's *Data and Code Availability* statement, replacing the
   `nymtc-spf-repro-XXXX` placeholder already there.

Double-check the anonymized view yourself: no name in README/LICENSE/CITATION, no identifying paths,
no author email. (`LICENSE`/`CITATION.cff` carry your name — either use a neutral placeholder in the
**review** snapshot, or rely on the anonymizer to mask them and verify it did.)

---

## 7. On acceptance: go public + mint a DOI

1. **Flip to public:** GitHub → repo → Settings → General → Danger Zone → Change visibility → Public.
2. **Connect Zenodo:** sign in at https://zenodo.org with GitHub, open the GitHub tab, toggle the repo **On**.
3. **Tag a release:** GitHub → Releases → Draft a new release → tag e.g. `v1.0.0` → Publish. Zenodo
   automatically archives that release and issues a **DOI** (a versioned DOI plus a concept DOI for "all versions").
4. **Update the paper:** replace the `10.5281/zenodo.XXXXXXX` placeholder with the concept DOI; restore your
   identity in `LICENSE`/`CITATION.cff` if you had masked it; add the camera-ready author block.
5. Optionally add the DOI badge Zenodo gives you to the top of `README.md`.

---

## 8. Handy everyday Git commands
```
git status                       # what changed
git add <files> && git commit -m "message"
git push                         # send commits to GitHub
git pull                         # get remote changes
git log --oneline --graph        # history
git checkout -b <branch>         # start a branch for an experiment
```

## 9. Common first-timer snags
- **Push rejected / auth failed** → use a Personal Access Token (not your password) over HTTPS, or set up SSH / `gh auth login`.
- **Accidentally committed `__pycache__` or a big file** → `git rm -r --cached <path>`, add it to `.gitignore`, commit again. For a secret committed by mistake, rotate the secret and scrub history (`git filter-repo`) — don't just delete in a new commit.
- **CI red on lint only** → lint is informational; check the `test` and `docker` jobs are green.
- **"large files" warning** → the `.xlsx` inputs are ~0.1–0.5 MB each (fine). Anything >50 MB triggers a warning, >100 MB is blocked — none here, but keep new data files small or use Git LFS.
