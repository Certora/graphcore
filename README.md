## Graphcore

This is a reusable framework for writing LLM-powered applications that iterate using
various client side tools to help the LLM complete a task. It is used by Certora's Concordance
and AI Composer (among others). It is also currently deeply undocumented.

## Development setup

### Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/)

### Install

```bash
make install
```

### Updating a dependency

When a dependency releases a new version: `make update-deps` then commit the updated `uv.lock`.
