# Contributing to nanoGPT2

Thank you for your interest in contributing to nanoGPT2! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check the [issues list](https://github.com/d-negatu/nanoGPT2/issues) as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* **Use a clear and descriptive title** for the issue
* **Describe the exact steps which reproduce the problem** with as much detail as possible
* **Provide specific examples to demonstrate the steps**
* **Describe the behavior you observed after following the steps** and point out what exactly is the problem with that behavior
* **Explain which behavior you expected to see instead** and why
* **Include screenshots and/or animated GIFs** if possible
* **Include your environment details** (OS, Python version, etc.)

### Suggesting Enhancements

Enhancement suggestions are tracked as [GitHub issues](https://github.com/d-negatu/nanoGPT2/issues). When creating an enhancement suggestion, please include:

* **Use a clear and descriptive title** for the issue
* **Provide a step-by-step description of the suggested enhancement** with as many details as possible
* **Provide specific examples to demonstrate the steps**
* **Describe the current behavior** and **explain the expected behavior**
* **Explain why this enhancement would be useful** to nanoGPT2 users

### Pull Requests

* Include appropriate test cases
* Update the README.md with details of changes if applicable
* Follow the existing code style
* Ensure your PR passes all CI checks (linting, type checking, tests)
* Include a clear description of the changes in your PR

## Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/nanoGPT2.git
   cd nanoGPT2
   ```
3. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install flake8 mypy pytest pytest-cov
   ```
4. Create a new branch for your changes:
   ```bash
   git checkout -b your-feature-name
   ```

## Development Workflow

1. **Run linting checks:**
   ```bash
   flake8 .
   ```

2. **Run type checking:**
   ```bash
   mypy .
   ```

3. **Run tests:**
   ```bash
   pytest
   ```

4. **Run tests with coverage:**
   ```bash
   pytest --cov=. --cov-report=html
   ```

## Commit Messages

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line

## Style Guide

* Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code
* Use meaningful variable and function names
* Add docstrings to functions and classes
* Keep functions and methods focused and concise

## Additional Notes

Thank you for contributing to nanoGPT2! Your efforts help make this project better for everyone.
