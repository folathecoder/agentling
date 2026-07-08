# Releasing agentling

The pipeline is small: bump the version, validate, dry-run on TestPyPI, publish
to PyPI, then tag and cut a GitHub release. Versions on PyPI and TestPyPI are
**permanent and non-reusable**, which is why the dry run matters.

## 1. Prepare

- [ ] On `main`, working tree clean (release from `main` after the PR merges).
- [ ] Bump `version` in `pyproject.toml`.
- [ ] Add a `## [x.y.z]` section to `CHANGELOG.md`.
- [ ] The gate is green:

  ```bash
  uv run pytest
  uv run ruff check src tests examples
  uv run ruff format --check src tests examples
  uv run mypy src tests examples
  ```

## 2. Build and write the release notes

```bash
rm -rf dist && uv build
uv run --with twine twine check dist/*
```

Write the version's summary once and reuse it for the tag and the GitHub
release. Keep it in step with the `CHANGELOG.md` entry:

```bash
cat > /tmp/vX.Y.Z-notes.md <<'EOF'
vX.Y.Z — <one-line theme>

<short paragraph>

Highlights:
- ...

Full changelog: CHANGELOG.md
EOF
```

## 3. Handle the token securely

Never put a token on the command line or in a file (shell and git history are
forever). Read it into an env var at a silent prompt; uv reads
`UV_PUBLISH_TOKEN`:

```bash
read -rs UV_PUBLISH_TOKEN && export UV_PUBLISH_TOKEN
# Verify without revealing it (length + harmless prefix only):
echo "length: ${#UV_PUBLISH_TOKEN} | prefix: ${UV_PUBLISH_TOKEN:0:5}"   # want ~150-200 and 'pypi-'
```

TestPyPI and PyPI use **separate** tokens. `unset UV_PUBLISH_TOKEN` before
switching between them, and again when done. The first-ever upload of a new
project needs an account-scoped token; use a project-scoped token afterward.

## 4. Dry run on TestPyPI

```bash
uv publish --publish-url https://test.pypi.org/legacy/
```

Install it back in a throwaway environment. agentling has real dependencies, so
`--extra-index-url` is required (TestPyPI does not host them):

```bash
uv venv /tmp/testpypi
uv pip install --python /tmp/testpypi \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  agentling
/tmp/testpypi/bin/python -c "import agentling; print(agentling.__version__)"
unset UV_PUBLISH_TOKEN && rm -rf /tmp/testpypi
```

## 5. Publish to PyPI

```bash
read -rs UV_PUBLISH_TOKEN && export UV_PUBLISH_TOKEN   # PyPI token this time
uv publish                                             # default target is real PyPI
unset UV_PUBLISH_TOKEN
```

Confirm a clean-environment install:

```bash
uv venv /tmp/pypi
uv pip install --python /tmp/pypi agentling
/tmp/pypi/bin/python -c "import agentling; print(agentling.__version__)"
rm -rf /tmp/pypi
```

## 6. Tag and release

Use an **annotated** tag so the tag itself carries the summary (`git show
vX.Y.Z` prints it), and reuse the same notes file for the GitHub release:

```bash
git tag -a vX.Y.Z -F /tmp/vX.Y.Z-notes.md
git push origin vX.Y.Z                                 # tags are not pushed by a normal push
gh release create vX.Y.Z --title "vX.Y.Z" --notes-file /tmp/vX.Y.Z-notes.md
```

## Checklist

- [ ] Version bumped in `pyproject.toml`
- [ ] `CHANGELOG.md` updated
- [ ] Gate green; `uv build` + `twine check` pass
- [ ] TestPyPI dry run installs and imports
- [ ] Published to PyPI; installs from a clean environment
- [ ] Annotated tag pushed; GitHub release created
- [ ] `UV_PUBLISH_TOKEN` unset
