# Anonymous Submission Verification

This snapshot is prepared for double-anonymous review.

## Applied safeguards

- The snapshot was created without the source repository's Git history or
  remote configuration.
- The snapshot's sole commit uses the neutral identity
  `Anonymous Authors <anonymous@invalid.example>`.
- Project branding, Python imports, command-line entry points, environment
  variables, documentation, and GUI labels use the neutral name
  `inkvoidmotif`.
- Contributor names, usernames, email addresses, affiliations, source-repository
  coordinates, and absolute workstation paths were removed or converted to
  repository-relative paths.
- Author, affiliation, copyright-holder, and final licensing details are
  withheld in the review snapshot and replaced by a review-only notice.
- Generated package metadata, interpreter caches, editor/OS metadata, and local
  runtime outputs are excluded.
- Provider credentials are represented only by documented placeholders; real
  `.env` files are ignored.
- The GUI screenshot is generated from the anonymous build.

## Verification before upload

From `inkvoidmotif/`, run:

```bash
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
mypy src/inkvoidmotif
bandit -q -r src/inkvoidmotif scripts app.py
python -m unittest discover -s tests -v
python -m build
```

Inspect the staged file list and confirm that no `.env`, run directories,
build caches, or local paths are present before creating the submission
archive.

## Double-anonymous distribution

Submit this snapshot only through the venue's anonymous supplementary-material
channel or another venue-approved anonymous host. Do not push it to the source
repository, a personal account, an institutional account, or any service whose
URL or owner profile identifies the authors.

The original project has previously existed in a public source repository.
Removing explicit identifiers reduces accidental disclosure, but cannot prevent
deanonymization through code or data similarity. Follow the venue's policy for
previously public artifacts and disclose that circumstance to the program chairs
if required.
