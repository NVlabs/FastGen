# Contributing to FastGen

Thank you for your interest in contributing to FastGen!

## Issue Tracking

* All enhancement, bugfix, or change requests should begin with the creation of an issue.
* The issue should be reviewed and approved prior to code review.

## Development Setup

All development should be done using the provided Docker image to ensure a consistent environment.

```bash
# Build the Docker image
docker build -t fastgen:dev .

# Run interactively with GPU support (as current user, not root)
docker run --gpus all -it --rm \
    --user $(id -u):$(id -g) \
    -e HOME=/workspace \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    -v $(pwd):/workspace \
    fastgen:dev bash

# Inside the container, install linters
make install
```

## Code Style and Formatting

FastGen uses **Ruff** for linting and code formatting. Follow existing conventions when adding new code.

```bash
make format  # Auto-format code
make lint    # Check compliance
```

**Note:** The `fastgen/third_party/` directory is excluded from checks.

## Continuous Integration

The CI pipeline runs automatically on every push to `main` and every pull request targeting `main`:

| Stage | Commands |
|-------|----------|
| Lint | `make lint`, `make mypy` |
| Test | `make pytest` |
| Install | `make install-fastgen` |

**Note:** Before submitting a pull request, ensure all checks pass locally.

The test suite also checks that the built wheel includes configs and runtime
data and works outside the checkout. Run only this check with
`python -m pytest tests/test_packaging.py`.

## Pull Request Process

1. Fork the repository (for external contributors) or create a feature branch.
2. Make your changes following the code style guidelines.
3. Run all checks locally (see [CI section](#continuous-integration)).
4. Commit your changes with sign-off (PRs with unsigned commits will not be accepted):
   ```bash
   git commit -s -m "Add feature X"
   ```
5. Push and create a Pull Request.

**Note:** Keep PRs focused on a single concern and reference related issues (e.g., "Fixes #123").

## Developer Certificate of Origin

```
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129

    Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
```

```
    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or

    (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or

    (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.

    (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
```

## License

Contributions to this project will only be accepted under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file in the root of this repository for the full text.
