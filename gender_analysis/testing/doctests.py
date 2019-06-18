import gender_analysis

if __name__ == '__main__':
    import doctest
    doctest.testmod(gender_analysis.common, verbose=True)
    doctest.testmod(gender_analysis.corpus, verbose=True)
    doctest.testmod(gender_analysis.novel, verbose=True)

