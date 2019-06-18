import gender_analysis
import doctest

def print_test_header(test_name):
    print('\n')
    print('================================================================================')
    print(f'Testing {test_name}')
    print('================================================================================')


if __name__ == '__main__':
    failures = 0
    tests = 0

    print_test_header('common')
    new_failures, new_tests = doctest.testmod(gender_analysis.common)
    failures += new_failures
    tests += new_tests

    print_test_header('corpus')
    new_failures, new_tests = doctest.testmod(gender_analysis.corpus)
    failures += new_failures
    tests += new_tests

    print_test_header('novel')
    new_failures, new_tests = doctest.testmod(gender_analysis.novel)
    failures += new_failures
    tests += new_tests

