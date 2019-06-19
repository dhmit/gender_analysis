import doctest
import importlib
import pkgutil

import gender_analysis

def print_test_header(test_name):
    print('\n\n')
    print('================================================================================')
    print(f'Testing {test_name}')
    print('================================================================================')


# Recurse through all packages and return a list of all absolute submodule names
def all_absolute_submodule_names(package, names):
    subpackages = pkgutil.walk_packages(package.__path__)
    for module_info in subpackages:
        absolute_module_name = package.__name__ + '.' + module_info.name

        if 'corpus_gen' in absolute_module_name:
            # TODO(ra) skipping corpus_gen bc of Gutenberg installation issues...
            # we'll deal with this later...
            continue

        module = importlib.import_module(absolute_module_name)
        if module_info.ispkg:
            all_absolute_submodule_names(module, names)
        else:
            names.append(absolute_module_name)

if __name__ == '__main__':
    submodule_names = []
    all_absolute_submodule_names(gender_analysis, submodule_names)

    failures = 0
    tests = 0

    for module_name in submodule_names:
        print_test_header(module_name)
        module = importlib.import_module(module_name)
        new_failures, new_tests = doctest.testmod(module)
        failures += new_failures
        tests += new_tests

    if failures:
        exit(1)
    else:
        exit(0)

