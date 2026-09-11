# Shipping inkvoidmotif

This checklist is for maintainers preparing a package for collaborators. The
installable project lives at `inkvoidmotif/` in the full research
repository.

## 1. Audit the source tree

From the package directory:

```bash
python -m pip install -e '.[dev]'
ruff check src tests app.py
ruff format --check src tests app.py
mypy src
bandit -q -r src
python -m unittest discover -s tests -v
pip-audit
```

The tests use local provider fakes and do not consume model quota.

Before building, confirm that the package contains no credentials or
machine-specific paths. In particular, never distribute:

- `.env`
- `.venv/`
- `runs/`
- build caches or `*.egg-info/`
- research datasets, extracted paintings, or generated motif banks

The repository's `.gitignore` excludes the local package artifacts above.

## 2. Build standard distributions

```bash
python -m build
```

This creates:

```text
dist/inkvoidmotif-0.1.0-py3-none-any.whl
dist/inkvoidmotif-0.1.0.tar.gz
```

The source archive is governed by `MANIFEST.in` and includes the README,
quickstart, example environment file, launcher, source, tests, and anonymous-review
license notice.

## 3. Inspect the archives

```bash
python -m zipfile -l dist/inkvoidmotif-0.1.0-py3-none-any.whl
tar -tzf dist/inkvoidmotif-0.1.0.tar.gz
```

Confirm that `.env`, `runs/`, caches, generated images, and credentials are not
present.

## 4. Test the wheel in a clean environment

```bash
python3 -m venv /tmp/inkvoidmotif-release-test
source /tmp/inkvoidmotif-release-test/bin/activate
python -m pip install --upgrade pip
python -m pip install dist/inkvoidmotif-0.1.0-py3-none-any.whl
python -m pip check
inkvoidmotif --help
inkvoidmotif-studio --help
```

Build the Gradio interface without making a provider call:

```bash
python -c 'from inkvoidmotif.gradio_app import build_demo; build_demo(); print("GUI build: OK")'
```

## 5. Share

For Python users, share the wheel. Share the source archive when recipients
need the source, tests, and documentation. During double-anonymous review, use
only a venue-approved anonymous upload channel; do not publish from a personal
or institutional account. Do not manually archive the entire research repository:
it contains large datasets and generated assets that are not part of the software
package.

Recipients should begin with `QUICKSTART.md`.
