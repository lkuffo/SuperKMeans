# Contributing

We are actively developing Super K-Means and accepting contributions! Any kind of PR is welcome. 

These are our current priorities:

**Improvements**:
- Change `.Train()`, `.Assign()` API to work with `std::vector` or `std::span` rather than pointers.
- A proper benchmarking framework for development.
- Support `uint64_t` for the `assignments`. Right now, we are limited to ~4 billion vectors.
- GitHub CI tests
- Regression tests on CI

**Features**:
- Hierarchical K-Means 
- Support for different datatypes: 64-bit `double`, 16-bit `half`, 8-bit `uint8` (experimental).

## Getting Started

1. **Fork the repository** on GitHub and create a feature branch:
```bash
git checkout -b my-feature
```

2. **Make your changes.**
3. **Run the test suite** locally before submitting your PR.
4. **Open a Pull Request (PR)** against the `main` branch.

> [!IMPORTANT]
> Let us know in advance if you plan implementing a big feature!

## Testing

All PRs must pass the full test suite in CI. Before submitting a PR, you should run tests locally:

```bash
ctest .
```

Tests are also prone to bugs. If that is the case, please open an Issue.

## Submitting a PR

* Open your PR against the **`main` branch**.
* Make sure your branch is **rebased on top of `main`** before submission.
* Verify that **CI passes**.
* Keep PRs focused — small, logical changes are easier to review and merge.

## Coding Style
* Function, Class, and Struct names: `PascalCase`
* Variable names: `snake_case`
* Constants and magic variables: `UPPER_SNAKE_CASE`
* Avoid `new` and `delete`
* There is a `.clang-format` in the project. Make sure to adhere to it.

## Communication

* Use GitHub Issues for bug reports and feature requests.
