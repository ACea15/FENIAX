# Tests

- Testing framework using [Pytest](https://docs.pytest.org/en/stable/). Tests marked as *slow*, *private* or *legacy* won't run by default. See the `tests` folder for details.

- CI/CD via Github actions, so every push to the master branch checks all tests are run.

# Examples

- The examples folder is a git [subtree](https://www.geeksforgeeks.org/git/git-subtree/) of the [FENIAXexamples](https://github.com/ACea15/FENIAXexamples) repo. The workflow followed is to install FENIAX anywhere in the system and the FENIAXexamples separated where one wants to run cases so that the code is not clutter with output files. 

- It is encouraged to keep adding new examples to FENIAXexamples, then updating the /examples in FENIAX: 
`git subtree add --prefix examples/ https://github.com/ACea15/FENIAXexamples --squash`

- Once cases are added, new tests should be included as a validation. Note the tests rely on data in the /examples folder. 

# Pending improvements

- Extend type-hinting to the entire codebase.

- Add entry point to run FENIAX from command line to .yml file. 

- TODOs: Search for TODOs within the codebase. Either integrated in your IDE, or just run `grep -r --include "*.py" "# TODO:" .` from the terminal

- Documentation is not fully completed and it should be continuously extended. 

- Setup a proper git branching model (as more than one developer contributes to the code).

- Add code coverage and improve percentage via utility tests.
