import gender_analysis
import doctest
import sys

if __name__ == '__main__':
    test_obj_input = sys.argv[1]
    test_obj = eval(test_obj_input)
    print(f'Testing {test_obj}')
    doctest.run_docstring_examples(test_obj, globals(), verbose=False)

