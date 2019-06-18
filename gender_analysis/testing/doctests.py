import gender_analysis

if __name__ == '__main__':
    import doctest
    print('Testing common...')
    doctest.testmod(gender_analysis.common)
    print('Testing corpus...')
    doctest.testmod(gender_analysis.corpus)
    print('Testing novel...')
    doctest.testmod(gender_analysis.novel)

