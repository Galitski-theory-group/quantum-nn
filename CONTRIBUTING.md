# Contributing

This project combines [test-driven development](https://tdd.mooc.fi/) with the [Git feature branch workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow).  Please submit your changes for review as a [GitHub pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests).  Changes must include functional and integration tests.

## Development Environment

This project requires Python 3.12, [pre-commit](https://pre-commit.com/), [OpenTofu 1.6.0](https://opentofu.org/docs/intro/install), and [TFLint](https://github.com/terraform-linters/tflint).  Run `pre-commit install --install-hooks` to set up the Git pre-commit hook scripts.  Additional Python tooling such as [the AWS CLI](https://aws.amazon.com/cli/) or [pytest](https://pytest.org/) may be installed via the `dev` and `test` lists of optional dependencies.

## Code Style

The following code styles are in use:

- [isort](https://pycqa.github.io/isort/) and [Python Black](https://black.readthedocs.io/)

- [the Home Assistant YAML style guide](https://developers.home-assistant.io/docs/documenting/yaml-style-guide/)

- [HashiCorp Terraform Style Conventions](https://developer.hashicorp.com/terraform/language/syntax/style)

## Commit Messages

This project implements [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html) using [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/).  Please use English in commit messages.  The first line of the commit message should be at most 100 characters, while the rest of the commit message should be wrapped at column 70.  A commit's description should be a verb phrase in the imperative present tense, with the starting verb in lower case and no ending punctuation.

Valid commit types are:

- **build**—changes to the build system or external dependencies

- **ci**—changes to the CI configuration files and scripts

- **docs**—documentation-only changes

- **feat**—a new feature

- **fix**—a bug fix

- **perf**—a code change that improves performance

- **refactor**—a code change that neither fixes a bug nor adds a feature

- **style**—a code change that only affects formatting (e.g., whitespace, semi-colons)

- **test**—new tests or corrections to existing tests

A commit's scope should be the second-level Python module name sans the package prefix or any suffixes, with the following exceptions:

- **infra**—declarative infrastructure-as-code, e.g., [main.tf](main.tf)

- **packaging**—package layout or other metadata, e.g., the arrangement of [src/](src/), alterations to [pyproject.toml](pyproject.toml)

- the [dunder](https://wiki.python.org/moin/DunderAlias) module name without the leading or trailing double underscores, e.g., **init** instead of **\_\_init\_\_**, **main** instead of **\_\_main\_\_**

- no scope—for **refactor** or **test** changes covering multiple scopes or for **build**, **ci**, or **docs** changes not specific to one scope
