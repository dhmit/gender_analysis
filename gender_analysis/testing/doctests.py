import doctest
import importlib
import pkgutil

import gender_analysis

def print_test_header(test_name):
    print('\n\n')
    print('================================================================================')
    print(f'Testing {test_name}')
    print('================================================================================')


# Recurse through all packages and return a list of all submodules
def get_all_submodules(package, module_list):
    subpackages = pkgutil.walk_packages(package.__path__)
    for module_info in subpackages:
        if ('gutenberg_loader' in module_info.name) or ('dependency_parsing' in module_info.name):
            # TODO(ra) - skipping corpus_gen bc of Gutenberg installation issues
            # TODO(ra) - skipping dependency_parsing because it requires the user to have
            #            downloaded a jar file that we're not hosting
            # We should handle both of these eventually, but for now let's leave it
            continue

        absolute_module_name = package.__name__ + '.' + module_info.name
        module = importlib.import_module(absolute_module_name)

        if module_info.ispkg:
            get_all_submodules(module, module_list)
        else:
            module_list.append(module)

if __name__ == '__main__':
    modules = []
    get_all_submodules(gender_analysis, modules)

    failures = 0
    tests = 0

    for module in modules:
        print_test_header(module.__name__)
        new_failures, new_tests = doctest.testmod(module)
        failures += new_failures
        tests += new_tests
        if not new_failures:
            print("All tests passed! Hurrah!")

    if failures:
        exit(1)
    else:
        exit(0)

