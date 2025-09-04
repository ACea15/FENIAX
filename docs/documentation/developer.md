# Tips


# Tests

- Testing framework using [Pytest](https://docs.pytest.org/en/stable/). Tests marked as *slow*, *private* or *legacy* won't run by default. See [Configuration](../../tests/conftest.py) for details.

- CI/CD via Github actions, so every push to the master branch checks all tests are run.


# Pending improvements

- Extend type-hinting to the entire codebase.

- Add entry point to run FENIAX from command line to .yml file. 

- TODOs: Search for TODOs within the codebase. Either integrated in your IDE, or just run `grep -r --include "*.py" "# TODO:" .` from the terminal

- Setup a proper branching model (as more than one developer contributes to the code).
