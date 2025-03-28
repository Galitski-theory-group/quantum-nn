# Best Practices

## Github

Feature-based development is summarized in the /docs/ folder. Semantic versioning and conventional commits are alluded to in CONTRIBUTING.md.

## Python Development

It is fine to prototype and play around in jupyter notebooks. But once the code is "finished", it should be bundled into a python package and minimally a .py script.

It is recommended to use venv, lightweight python virtual machines which isolate your development environment from your machine and therefore ensure minimal dependency conflict and portability of usage.

The Makefile shows how one can create such an environment. One can also just use that makefile to create the env using.

```
python$(PYTHON_VERSION) -m venv .test_repo
```

A linter system is also bundled up with this prototype repository. A linter checks for syntax, and also for style. We enforce the "black" style guide for readibility. Using the make command will install the pre-commit "hooks".
