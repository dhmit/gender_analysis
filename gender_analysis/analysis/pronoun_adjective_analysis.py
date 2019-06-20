from gender_analysis.analysis.analysis import find_male_adj, find_female_adj
from gender_analysis import common


def run_adj_analysis(corpus):
    """
    Takes in a corpus of novels. Return a dictionary with each novel mapped to 2 dictionaries: adjectives/#occurences
    associated with male pronouns, and adjectives/# occurrences associated with female pronouns

    :param corpus:
    :return:dictionary where each key is a novel and the value is 2 dictionaries:
    - Adjectives and number of occurences associated with male pronouns
    - Adjectives and number of occurrences associated with female pronouns

    """
    results = {}
    for novel in corpus:
        print(novel.title, novel.author)
        novel_male_results = find_male_adj(novel)
        novel_female_results = find_female_adj(novel)
        if (novel_male_results != "lower window bound less than 5"
                and novel_female_results != "lower window bound less than 5"):
            novel.text = ""
            novel._word_counts_counter = None
            results[novel] = {'male': novel_male_results, 'female': novel_female_results}

    return results


def store_raw_results(results, corpus):
    try:
        common.load_pickle("pronoun_adj_raw_analysis_" + corpus.corpus_name)
        x = input("results already stored. overwrite previous analysis? (y/n)")
        if x == 'y':
            common.store_pickle(results, "pronoun_adj_raw_analysis_" + corpus.corpus_name)
        else:
            pass
    except IOError:
        common.store_pickle(results, "pronoun_adj_raw_analysis_" + corpus.corpus_name)


def merge(novel_adj_dict, full_adj_dict):
    """
    Merges adjective occurrence results from a single novel with results for each novel.
    :param novel_adj_dict: dictionary of adjectives/#occurrences for one novel
    :param full_adj_dict: dictionary of adjectives/#occurrences for multiple novels
    :return: full_results dictionary with novel_results merged in

        >>> novel_adj_dict = {'hello': 5, 'hi': 7, 'hola': 9, 'bonjour': 2}
        >>> full_adj_dict = {'hello': 15, 'bienvenue': 3, 'hi': 23}
        >>> merge(novel_adj_dict, full_adj_dict)
        {'hello': 20, 'bienvenue': 3, 'hi': 30, 'hola': 9, 'bonjour': 2}
    """
    for adj in list(novel_adj_dict.keys()):
        adj_count = novel_adj_dict[adj]
        if adj in list(full_adj_dict.keys()):
            full_adj_dict[adj] += adj_count
        else:
            full_adj_dict[adj] = adj_count
    return full_adj_dict


def merge_raw_results(full_results):
    """
    Merges all of the male adjectives across novels into a single dictionary, and merges all of the female adjectives
    across novels into a single dictionary.
    :param full_results: full corpus results from run_adj_analysis
    :return: dictionary
    """
    merged_results = {'male': {}, 'female': {}}
    for novel in list(full_results.keys()):
        print(novel.title, novel.author)
        for gender in list(full_results[novel].keys()):
            merged_results[gender] = merge(full_results[novel][gender], merged_results[gender])

    return merged_results


def get_overlapping_adjectives_raw_results(merged_results):
    """
    Looks through the male adjectives and female adjectives across the corpus and extracts adjective
    that overlap across both and their occurrences. FORMAT - {'adjective': [male, female]}
    :param merged_results:
    :return:
    """
    overlap_results = {}
    male_adj = list(merged_results['male'].keys())
    female_adj = list(merged_results['female'].keys())

    for a in male_adj:
        if a in female_adj:
            overlap_results[a] = [merged_results['male'][a], merged_results['female'][a]]

    return overlap_results


def results_by_author_gender(full_results):
    """
       takes in the full dictionary of results, returns a dictionary that maps 'male' (male author) and 'female'
       (female author) to a dictionary of adjectives and # occurrences across novels written by an author of
       that gender.
       :param full_results: dictionary from result of run_adj_analysis
       :return: dictionary with two keys: 'male' (male author) or 'female' (female author). Each key maps
       a dictionary of adjectives/occurrences.
       """
    data = {'male_author': {'male': {}, 'female': {}}, "female_author": {'male': {}, 'female': {}}}

    for novel in list(full_results.keys()):
        # print("author gender analysis:", novel.title, novel.author)
        # TODO: check if novel has author_gender attribute
        if novel.author_gender == "male":
            data['male_author']['male'] = merge(full_results[novel]['male'], data['male_author']['male'])
            data['male_author']['female'] = merge(full_results[novel]['female'], data['male_author']['female'])
        else:
            data['female_author']['male'] = merge(full_results[novel]['male'], data['female_author']['male'])
            data['female_author']['female'] = merge(full_results[novel]['female'], data['female_author']['female'])
    return data


def results_by_date(full_results):
    """
    takes in a dictionary of results returns a dictionary that maps different time periods to a dictionary of
    adjectives/number of occurrences across novels written in that time_period

    :param full_results: dictionary from result of run_adj_analysis
    :return: dictionary
    """
    data = {}

    # TODO: remove hardcoded dates

    date_to_1810 = {'male': {}, 'female': {}}
    date_1810_to_1819 = {'male': {}, 'female': {}}
    date_1820_to_1829 = {'male': {}, 'female': {}}
    date_1830_to_1839 = {'male': {}, 'female': {}}
    date_1840_to_1849 = {'male': {}, 'female': {}}
    date_1850_to_1859 = {'male': {}, 'female': {}}
    date_1860_to_1869 = {'male': {}, 'female': {}}
    date_1870_to_1879 = {'male': {}, 'female': {}}
    date_1880_to_1889 = {'male': {}, 'female': {}}
    date_1890_to_1899 = {'male': {}, 'female': {}}
    date_1900_on = {'male': {}, 'female': {}}

    for k in list(full_results.keys()):
        # TODO: check if k has date attribute
        if k.date < 1810:
            date_to_1810['male'] = merge(full_results[k]['male'], date_to_1810['male'])
            date_to_1810['female'] = merge(full_results[k]['female'], date_to_1810['female'])
        elif k.date < 1820:
            date_1810_to_1819['male'] = merge(full_results[k]['male'], date_1810_to_1819['male'])
            date_1810_to_1819['female'] = merge(full_results[k]['female'], date_1810_to_1819['female'])
        elif k.date < 1830:
            date_1820_to_1829['male'] = merge(full_results[k]['male'], date_1820_to_1829['male'])
            date_1820_to_1829['female'] = merge(full_results[k]['female'], date_1820_to_1829['female'])
        elif k.date < 1840:
            date_1830_to_1839['male'] = merge(full_results[k]['male'], date_1830_to_1839['male'])
            date_1830_to_1839['female'] = merge(full_results[k]['female'], date_1830_to_1839['female'])
        elif k.date < 1850:
            date_1840_to_1849['male'] = merge(full_results[k]['male'], date_1840_to_1849['male'])
            date_1840_to_1849['female'] = merge(full_results[k]['female'], date_1840_to_1849['female'])
        elif k.date < 1860:
            date_1850_to_1859['male'] = merge(full_results[k]['male'], date_1850_to_1859['male'])
            date_1850_to_1859['female'] = merge(full_results[k]['female'], date_1850_to_1859['female'])
        elif k.date < 1870:
            date_1860_to_1869['male'] = merge(full_results[k]['male'], date_1860_to_1869['male'])
            date_1860_to_1869['female'] = merge(full_results[k]['female'], date_1860_to_1869['female'])
        elif k.date < 1880:
            date_1870_to_1879['male'] = merge(full_results[k]['male'], date_1870_to_1879['male'])
            date_1870_to_1879['female'] = merge(full_results[k]['female'], date_1870_to_1879['female'])
        elif k.date < 1890:
            date_1880_to_1889['male'] = merge(full_results[k]['male'], date_1880_to_1889['male'])
            date_1880_to_1889['female'] = merge(full_results[k]['female'], date_1880_to_1889['female'])
        elif k.date < 1900:
            date_1890_to_1899['male'] = merge(full_results[k]['male'], date_1890_to_1899['male'])
            date_1890_to_1899['female'] = merge(full_results[k]['female'], date_1890_to_1899['female'])
        else:
            date_1900_on['male'] = merge(full_results[k]['male'], date_1900_on['male'])
            date_1900_on['female'] = merge(full_results[k]['female'], date_1900_on['female'])

    data['date_to_1810'] = date_to_1810
    data['date_1810_to_1819'] = date_1810_to_1819
    data['date_1820_to_1829'] = date_1820_to_1829
    data['date_1830_to_1839'] = date_1830_to_1839
    data['date_1840_to_1849'] = date_1840_to_1849
    data['date_1850_to_1859'] = date_1850_to_1859
    data['date_1860_to_1869'] = date_1860_to_1869
    data['date_1870_to_1879'] = date_1870_to_1879
    data['date_1880_to_1889'] = date_1880_to_1889
    data['date_1890_to_1899'] = date_1890_to_1899
    data['date_1900_on'] = date_1900_on

    return data


def results_by_location(full_results):
    """
    :param full_results: dictionary from result of run_adj_analysis
    :return:
    """
    data = {}

    # TODO: remove hardcoded locations

    location_UK = {'male': {}, 'female': {}}
    location_US = {'male': {}, 'female': {}}
    location_other = {'male': {}, 'female': {}}

    for k in list(full_results.keys()):
        # TODO: check if k has country_publication attribute
        if k.country_publication == 'United Kingdom' or k.country_publication == "England":
            location_UK['male'] = merge(full_results[k]['male'], location_UK['male'])
            location_UK['female'] = merge(full_results[k]['female'], location_UK['female'])
        elif k.country_publication == 'United States':
            location_US['male'] = merge(full_results[k]['male'], location_US['male'])
            location_US['female'] = merge(full_results[k]['female'], location_US['female'])
        else:
            location_other['male'] = merge(full_results[k]['male'], location_other['male'])
            location_other['female'] = merge(full_results[k]['female'], location_other['female'])

    data['location_UK'] = location_UK
    data['location_US'] = location_US
    data['location_other'] = location_other

    return data


def get_top_adj(full_results, num):
    """
    Takes dictionary of results from run_adj_analysis and number of top results to return.
    Returns the top num adjectives associated with male pronouns and female pronouns
    :param full_results: dictionary from result of run_adj_analysis
    :param num: number of top results to return per gender
    :return: tuple of lists of top adjectives associated with male pronouns and female pronouns,
    respectively
    """
    male_adj = []
    female_adj = []

    for adj, val in full_results.items():
        male_adj.append((val[0]-val[1], adj))
        female_adj.append((val[1]-val[0], adj))

    male_top = sorted(male_adj, reverse=True)[0:num]
    female_top = sorted(female_adj, reverse=True)[0:num]

    return male_top, female_top
