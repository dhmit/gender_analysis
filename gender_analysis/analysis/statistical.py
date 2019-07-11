import numpy as np
from scipy import stats


def get_p_and_ttest_value(list_a,list_b):

    """
    Takes in two lists and returns t-test and p value
    can be used to establish correlation between author gender and word usage
    also used for null hypothesis testing

    :param list_a:
    :param list_b:
    :return: (ttest value , p value)
    """

    ttest_p_value = stats.ttest_ind(list_a, list_b, equal_var=False)
    return ttest_p_value


def ind_ttest(array1, array2, pvalue_target=0.05):
    """
    Independent t-test for two independent variables
    :param array1: array-like, data for one category. e.g. he/she distance in novels authored by
    women
    :param array2: array-like, data for second category. e.g. he/she distance in novels authored
    by men
    :param pvalue_target: largest p-value for which we consider the test statistically significant
    :return: True if the difference in the means of the two arrays are significant, False otherwise
    >>> a1 = np.array([1, 2, 3, 4, 5])
    >>> a2 = np.array([1, 2, 3, 4, 5])
    >>> ind_ttest(a1, a2)
    False
    >>> a3 = np.array([3, 4, 8, 6, 2])
    >>> a4 = np.array([14, 8, 17, 9, 16])
    >>> ind_ttest(a3, a4)
    True
    """

    # don't assume that the two variables have equal standard deviation
    pvalue = stats.ttest_ind(array1, array2, equal_var=False)[1]

    return pvalue < pvalue_target


def linear_regression(array1, array2, pvalue_target=0.05):
    """
    Perform a linear regression on two continuous variables that may or may not be correlated.
    Returns True if the correlation is significant, and false otherwise.

    :param array1: array-like, one set of continuous data to be compared to array2. e.g. list of publication years in a certain order of novels
    :param array2: array-like, second set of continuous data to be compared to array1, should be the same size as array1. e.g. he/she distance in the same order of novels as array1
    :param pvalue_target: largest p-value for which we consider the correlation statistically significant
    :return: True if the correlation is significant, False otherwise

    >>> a1 = np.array([1, 2, 3, 4, 5])
    >>> a2 = np.array([1, 2, 3, 4, 5])
    >>> linear_regression(a1, a2)
    True
    >>> a3 = np.array([3, 4, 8, 6, 2])
    >>> a4 = np.array([14, 8, 17, 9, 16])
    >>> linear_regression(a3, a4)
    False

    """

    pvalue = stats.linregress(array1, array2)[3]
    return pvalue < pvalue_target


def pearson_correlation(array1, array2, pvalue_target=0.05):
    """
    Pearson correlation test of two continuous variables for correlation

    :param array1: array-like, one set of continuous data to be compared to array2
    :param array2: array-like, second set of continuous data to be compared to array1, should be the same size as array1
    :param pvalue_target: largest p-value for which we consider the correlation statistically significant
    :return: True if the correlation is significant, False otherwise.

    >>> a1 = np.array([1, 2, 3, 4, 5])
    >>> a2 = np.array([1, 2, 3, 4, 5])
    >>> pearson_correlation(a1, a2)
    True
    >>> a3 = np.array([3, 4, 8, 6, 2])
    >>> a4 = np.array([14, 8, 17, 9, 16])
    >>> pearson_correlation(a3, a4)
    False

    """

    pvalue = stats.pearsonr(array1, array2)[1]

    return pvalue < pvalue_target
