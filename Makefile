# FirstOps SDK — dev + release tasks.
#
# Release:  make release VERSION=0.2.1
#   -> bumps pyproject version, runs tests, commits, tags v0.2.1, pushes the
#      tag, and the GitHub `publish.yml` workflow builds + publishes to PyPI.
#
# PyPI versions are immutable: every release needs a NEW version. This target
# keeps the pyproject version and the git tag in lockstep so they can't drift.

.PHONY: help install test build clean release

PYTHON ?= python3
VENV   ?= .venv
PY      = $(VENV)/bin/python

help:
	@echo "make install                 create venv + install package and dev deps"
	@echo "make test                    run the test suite"
	@echo "make build                   build wheel + sdist into dist/"
	@echo "make clean                   remove build artifacts"
	@echo "make release VERSION=X.Y.Z   bump version, test, tag, and publish to PyPI"

$(VENV):
	$(PYTHON) -m venv $(VENV)

install: $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]" build

test:
	$(PY) -m pytest -q

build: clean
	$(PY) -m build

clean:
	rm -rf dist build src/*.egg-info *.egg-info

release:
	@if [ -z "$(VERSION)" ]; then \
		echo "Usage: make release VERSION=X.Y.Z"; exit 1; fi
	@echo "$(VERSION)" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+([abrc][0-9]+|\.post[0-9]+|\.dev[0-9]+)?$$' \
		|| { echo "VERSION must look like 1.2.3 (got '$(VERSION)')"; exit 1; }
	@if [ -n "$$(git status --porcelain)" ]; then \
		echo "Working tree not clean — commit or stash your changes first."; \
		echo "(A release commit should contain ONLY the version bump.)"; exit 1; fi
	@if git rev-parse "v$(VERSION)" >/dev/null 2>&1; then \
		echo "Tag v$(VERSION) already exists. Pick a new version (PyPI is immutable)."; exit 1; fi
	@echo ">> bumping pyproject.toml -> $(VERSION)"
	@$(PY) -c "import re,pathlib; p=pathlib.Path('pyproject.toml'); s=p.read_text(); s,n=re.subn(r'(?m)^version = \".*\"','version = \"$(VERSION)\"',s,count=1); assert n==1,'version line not found'; p.write_text(s)"
	@echo ">> running tests"
	@$(PY) -m pytest -q
	@echo ">> committing + tagging v$(VERSION)"
	@git add pyproject.toml
	@git commit -m "Release v$(VERSION)"
	@git tag -a "v$(VERSION)" -m "firstops v$(VERSION)"
	@echo ">> pushing branch + tag (triggers the PyPI publish workflow)"
	@git push
	@git push origin "v$(VERSION)"
	@echo ">> released v$(VERSION). Watch the publish run:"
	@echo "   gh run watch \$$(gh run list --workflow=publish.yml -L1 --json databaseId -q '.[0].databaseId') --exit-status"
