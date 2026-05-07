## Graphcore

This is a reusable framework for writing LLM-powered applications that iterate using
various client side tools to help the LLM complete a task. It is used by Certora's Concordance
and AI Composer (among others). It is also currently deeply undocumented.

## Development setup

### Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- AWS credentials with access to the Certora CodeArtifact domain

### Developer auth

Bootstrap the `certora-cloud` CLI and write a CodeArtifact token to `~/.netrc`:

```bash
uv tool install 'git+ssh://git@github.com/Certora/certora-cloud-cli.git'
certora-cloud login
certora-cloud codeartifacts login   # writes ~/.netrc, valid ~12h
```

After that, `make login` is the everyday refresh command.

### Install

```bash
make login   # writes ~/.netrc token, valid 12 h
make install
```

### Updating a dependency

When a dependency releases a new version: `make update-deps` then commit the updated `uv.lock`.
