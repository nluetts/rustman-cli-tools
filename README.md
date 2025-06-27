# Raman CLI Tools

## Build

To include git commit sha in binary, do:

```bash
PROJECT_VERSION=$(git rev-parse --short HEAD) cargo build --release
```
