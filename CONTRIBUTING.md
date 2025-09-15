# Contributing

Thank you for taking the time to contribute to Elysia!

> And if you like the project, but just don’t have time to contribute, that’s fine. There are other easy ways to support the project and show your appreciation, which we would also be very happy about:
> - Star the project
> - Tweet/post on social media about it
> - Tell your friends about it!

## Table of Contents

- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)
- [Guidelines for Pull Requests](#guidelines-for-pull-requests)
    - [Branch Structure](#branch-structure)
    - [Naming Scheme](#naming-scheme)

## Reporting Bugs

**Before Submitting a Bug Report**

A good bug report shouldn’t leave others needing to chase you up for more information. Therefore, we ask you to investigate carefully, collect information and describe the issue in detail in your report. Please complete the following steps in advance to help us fix any potential bug as fast as possible.

- Make sure that you are using the latest version.
- Determine if your bug is really a bug and not an error on your side e.g. using incompatible environment components/versions (Make sure that you have read [the documentation](https://weaviate.github.io/elysia/)).
- To see if other users have experienced (and potentially already solved) the same issue you are having, check if there is not already a bug report existing for your bug or error in the bug tracker.
- Collect information about the bug:
    - Stack trace (Traceback) if applicable, or terminal output
    - OS, Platform and Version (Windows, Linux, macOS, x86, ARM)
    - Python version
    - Possibly your input and the output
    - Can you reliably reproduce the issue? And can you also reproduce it with older versions?
- For Elysia specifically, which revolves around using LLMs for tasks. Make sure that your issue is not an 'LLM issue' - is it an LLM making a mistake, calling the wrong tool? Does the problem persist when trying different LLMs? Can you try with larger LLMs that are less likely to make mistakes?

## Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues.

**Before Submitting an Enhancement**

- Make sure that you are using the latest version.
- Read the documentation carefully and find out if the functionality is already covered, maybe by an individual configuration.
- Perform a search to see if the enhancement has already been suggested. If it has, add a comment to the existing issue instead of opening a new one.
- Find out whether your idea fits with the scope and aims of the project. It’s up to you to make a strong case to convince the project’s developers of the merits of this feature. Keep in mind that we want features that will be useful to the majority of our users and not just a small subset. If you’re just targeting a minority of users, consider writing an add-on/plugin library.

For your suggestion:

- Use a clear and descriptive title for the issue to identify the suggestion.
- Provide a step-by-step description of the suggested enhancement in as many details as possible.
- Describe the current behavior and explain which behavior you expected to see instead and why. At this point you can also tell which alternatives do not work for you.
- You may want to include screenshots or screen recordings which help you demonstrate the steps or point out the part which the suggestion is related to.
- Explain why this enhancement would be useful to most Elysia users. 

## Guidelines for Pull Requests

Elysia uses [black](https://github.com/psf/black) for formatting Python code.

### Branch Structure

The `main` and `dev` branches will be used in active development of Elysia and will be contributing towards the next release. Pull requests that involve new features should be made to the `main` branch. Smaller improvements such as bugfixes should be made to a current release branch, prefixed with `release/`

When a new version of Elysia is released, the release tag will be created from `main` and a new branch called `release/vY.Z.x` will be created for bug fixes or chores. The `release/vY.Z.x` branch serves as the snapshot of the earlier version for fixes.


> **Example**
>
> Say Elysia `v0.2.0` is released, and you want to contribute a bug fix to this version. You should fork the `release/v0.2.x` branch to make your changes from there, and submit a PR to that branch. This will improve the `v0.2.x` release and new versions such as `v0.2.1` will include your changes, if approved. But if you wanted to submit a PR that includes a new feature or is not related to a bug fix, then you should make a PR to `main`.

### Naming Scheme

Your contribution should be organised as follows:

- Your branch should be prefixed by `<contribution_type>/<description_of_contribution>`, such as `bugfix/incorrect_variable_name`. Some examples of `<contribution_type>/`s are `bugfix/` (for small fixes), `hotfix/` (for urgent fixes), `feature/` (for new features), `docs/` (for documentation updates) and `chore/` (for smaller non-fix changes).
- Any `feature/` pull requests must be onto the `main` branch.
- Any active development to the _next_ version should be onto `main` branch.
- Any other PRs should be made to the current `release/` branch.

---

Examples:

| Contribution type           | Branch to PR into                         | Prefix example               |
| --------------------------- | ----------------------------------------- | ---------------------------- |
| Bug fix for current release | `release/vY.Z.x`                          | `bugfix/fix-null-error`      |
| Urgent fix                  | `release/vY.Z.x` or `main` (case-by-case) | `hotfix/patch-api-leak`      |
| New feature                 | `main`                                    | `feature/add-export-command` |
| Docs                        | Current branch relevant to doc version    | `docs/improve-install-guide` |

---

## Testing

To run the tests, you need to install the dev extra of Elysia, via
```bash
pip install "elysia-ai[dev]"
```
or from the source
```bash
pip install ".[dev]"
```

Due to the nature of Elysia's development being heavily involved with LLMs and Weaviate collections, tests are split into two categories:

- `no_reqs`, which have no requirements and deal with the base functionality of Elysia
- `requires_env`, which require several API keys in your environment

The `requires_env` directory of tests require access to a Weaviate cluster as well as an OpenAI API key and an OpenRouter API key. There is no guarantee that running the tests in `requires_env` will be cheap - there will be a lot of LLM calls in these tests. It is not required for you to pass all of the tests in `requires_env`. Any serious contributions can be tested by the Elysia team fully without requirements for you to pay for running the tests.

But you should be expected that if your contributions contain changes to the codebase, at least all of the tests in the `no_reqs` directory pass successfully.

Any questions please send an email to me at danny@weaviate.io!