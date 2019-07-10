from statistics import median, mean

import matplotlib.pyplot as plt
import pandas as pnds
from scipy import stats
import numpy as np
import seaborn as sns
sns.set()

from gender_analysis import common


def instance_dist(document, word):
    """
    Takes in a particular word, returns a list of distances between each instance of that word in
    the document.
    >>> from gender_analysis import document
    >>> from pathlib import Path
    >>> from gender_analysis import common
    >>> document_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter', 'date': '1966',
    ...                   'filename': 'test_text_3.txt', 'filepath': Path(common.BASE_PATH, 'testing', 'corpora', 'document_test_files', 'test_text_3.txt')}
    >>> scarlett = document.Document(document_metadata)
    >>> instance_dist(scarlett, "her")
    [6, 5, 6, 7, 7]

    :param: document: Document to analyze
    :param: word: str
    :return: list of distances between consecutive instances of word

    """
    return words_instance_dist(document, [word])


def words_instance_dist(document, words):
    """
        Takes in a document and list of words (e.g. gendered pronouns), returns a list of distances
        between each instance of one of the words in that document
        >>> from gender_analysis import document
        >>> from pathlib import Path
        >>> from gender_analysis import common
        >>> document_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter', 'date': '1966',
        ...                   'filename': 'test_text_4.txt', 'filepath': Path(common.BASE_PATH, 'testing', 'corpora', 'document_test_files', 'test_text_4.txt')}
        >>> scarlett = document.Document(document_metadata)
        >>> words_instance_dist(scarlett, ["his", "him", "he", "himself"])
        [6, 5, 6, 6, 7]

        :param: document: Document
        :param: words: list of strings
        :return: list of distances between instances of any word in words
    """
    text = document.get_tokenized_text()
    output = []
    count = 0
    start = False

    for token in text:
        token = token.lower()
        if not start:
            if token in words:
                start = True
        else:
            count += 1
            if token in words:
                output.append(count)
                count = 0
    return output


def male_instance_dist(document):
    """
        Takes in a document, returns a list of distances between each instance of a female pronoun
        in that document.
       >>> from gender_analysis import document
       >>> from pathlib import Path
       >>> from gender_analysis import common
       >>> document_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter', 'date': '1966',
       ...                   'filename': 'test_text_5.txt', 'filepath': Path(common.BASE_PATH, 'testing', 'corpora', 'document_test_files', 'test_text_5.txt')}
       >>> scarlett = document.Document(document_metadata)
       >>> male_instance_dist(scarlett)
       [6, 5, 6, 6, 7]

       :param: document
       :return: list of distances between instances of gendered word
    """
    return words_instance_dist(document, ["his", "him", "he", "himself"])


def female_instance_dist(document):
    """
        Takes in a document, returns a list of distances between each instance of a female pronoun
        in that document.
       >>> from gender_analysis import document
       >>> from pathlib import Path
       >>> from gender_analysis import common
       >>> document_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter', 'date': '1966',
       ...                   'filename': 'test_text_6.txt', 'filepath': Path(common.BASE_PATH, 'testing', 'corpora', 'document_test_files', 'test_text_6.txt')}
       >>> scarlett = document.Document(document_metadata)
       >>> female_instance_dist(scarlett)
       [6, 5, 6, 6, 7]

       :param: document
       :return: list of distances between instances of gendered word
    """
    return words_instance_dist(document, ["her", "hers", "she", "herself"])


def run_distance_analysis(corpus):
    """
    Takes in a corpus of documents. Return a dictionary with each document mapped to an array of 3 lists:
     - median, mean, min, and max distances between male pronoun instances
     - median, mean, min, and max distances between female pronoun instances
     - for each of the above stats, the difference between male and female values. (male stat -
     female stat for all stats)
        POSITIVE DIFFERENCE VALUES mean there is a LARGER DISTANCE BETWEEN MALE PRONOUNS and
        therefore HIGHER FEMALE FREQUENCY.
    dict order: [male, female]

    :param corpus:
    :return:dictionary where the key is a document and the value is the results of distance analysis
    """
    results = {}

    for novel in corpus:
        # print(novel.title, novel.author)
        male_results = male_instance_dist(novel)
        female_results = female_instance_dist(novel)

        male_stats = get_stats(male_results)
        female_stats = get_stats(female_results)

        diffs = {}
        for stat in range(0, 4):
            stat_diff = list(male_stats.values())[stat] - list(female_stats.values())[stat]
            diffs[list(male_stats.keys())[stat]] = stat_diff

        novel.text = ""
        novel._word_counts_counter = None
        results[novel] = {'male': male_stats, 'female': female_stats, 'difference': diffs}

    return results


def store_raw_results(results, pickle_filepath='instance_distance_raw_analysis.pgz'):
    try:
        common.load_pickle(pickle_filepath)
        x = input("results already stored. overwrite previous analysis? (y/n)")
        if x == 'y':
            common.store_pickle(results, pickle_filepath)
        else:
            pass
    except IOError:
        common.store_pickle(results, pickle_filepath)


def get_stats(distance_results):
    """
    list order: median, mean, min, max
    :param distance_results:
    :return: dictionary of stats
    """
    if len(distance_results) == 0:
        return {'median': 0, 'mean': 0, 'min': 0, 'max': 0}
    else:
        return {'median': median(distance_results), 'mean': mean(distance_results), 'min': min(
            distance_results), 'max': max(distance_results)}


def results_by_author_gender(results, metric):
    """
    takes in a dictionary of results and a specified metric from run_distance_analysis, returns a
    dictionary:
     - key = 'male' or 'female' (indicating male or female author)
     - value  = list of lists. Each list has 3 elements: median/mean/max/min male pronoun distance,
     female pronoun distance, and the difference (whether it is median, mean, min, or max depends on
     the specified metric)
     order = [male distance, female distance, difference]
    :param results dictionary
    :param metric ('median', 'mean', 'min', 'max')
    :return: dictionary
    """
    data = {'male': [], "female": []}
    metric_indexes = {"median": 0, "mean": 2, "min": 3, "max": 4}
    if metric not in metric_indexes:
        raise ValueError(f"{metric} is not valid metric name. Valid names: 'median', 'mean', 'min', 'max'")

    stat = metric_indexes[metric]

    for document in list(results.keys()):
        author_gender = getattr(document, 'author_gender', None)
        if author_gender == "male":
            data['male'].append([results[document]['male'][metric], results[document]['female'][metric],
                                 results[document]['difference'][metric]])
        elif author_gender == 'female':
            data['female'].append([results[document]['male'][metric], results[document]['female'][metric],
                                   results[document]['difference'][metric]])
    return data


def results_by_date(results, metric, time_frame, bin_size):
    """
    takes in a dictionary of results and a specified metric from run_distance_analysis, returns a
    dictionary:
     - key = date range
     - value  = list of lists. Each list has 3 elements: median/mean/max/min male pronoun distance,
     female pronoun distance, and the difference (whether it is median, mean, min, or max depends on
     the specified metric)
     order = [male distance, female distance, difference]
    :param results dictionary
    :param metric ('median', 'mean', 'min', 'max')
    :param time_frame: tuple (int start year, int end year) for the range of dates to return
    frequencies
    :param bin_size: int for the number of years represented in each list of frequencies
    :return: dictionary
    """

    metric_indexes = {"median": 0, "mean": 2, "min": 3, "max": 4}
    if metric not in metric_indexes:
        raise ValueError(f"{metric} is not valid metric name. Valid names: 'median', 'mean', 'min', 'max'")

    stat = metric_indexes[metric]

    data = {}
    for bin_start_year in range(time_frame[0], time_frame[1], bin_size):
        data[bin_start_year] = []

    for k in results.keys():
        date = getattr(k, 'date', None)
        if date is None:
            continue
        bin_year = ((date - time_frame[0]) // bin_size) * bin_size + time_frame[0]
        data[bin_year].append([results[k]['male'][metric], results[k]['female'][metric],
                               results[k]['difference'][metric]])

    return data


def results_by_location(results, metric):
    """
    takes in a dictionary of results and a specified metric from run_distance_analysis, returns a
    dictionary:
     - key = location
     - value  = list of lists. Each list has 3 elements: median/mean/max/min male pronoun distance,
      female pronoun distance, and the difference (whether it is median, mean, min,
      or max depends on the specified metric)
      order = [male distance, female distance, difference]
    :param results dictionary
    :param metric ('median', 'mean', 'min', 'max')
    :return: dictionary
    """
    data = {}
    metric_indexes = {"median": 0, "mean": 2, "min": 3, "max": 4}
    if metric not in metric_indexes:
        raise ValueError(f"{metric} is not valid metric name. Valid names: 'median', 'mean', 'min', 'max'")

    stat = metric_indexes[metric]

    for k in results.keys():
        location = getattr(k, 'country_publication', None)
        if location is None:
            continue

        if location not in data:
            data[location] = []
        data[location].append([results[k]['male'][metric], results[k]['female'][metric],
                               results[k]['difference'][metric]])

    return data


def get_highest_distances(results, num):
    """
    Takes results from instance_distance_analysis.run_distance_analysis and a number of top
    results to return.
    Returns 3 lists.
        - Documents with the largest median male instance distance
        - Documents with the largest median female instance distance
        - Documents with the largest difference between median male & median female instance distances
    each list contains tuples, where each tuple has a document and the median male/female/difference
    instance distance
    :param results: dictionary of results from run_distance_analysis
    :param num: number of top distances to get
    :return: 3 lists of tuples.
    """

    male_medians = []
    female_medians = []
    difference_medians = []

    for document in list(results.keys()):
        male_medians.append((results[document]['male']['median'], document))
        female_medians.append((results[document]['female']['median'], document))
        difference_medians.append((results[document]['difference']['median'], document))

    male_top = sorted(male_medians, reverse=True)[0:num]
    female_top = sorted(female_medians, reverse=True)[0:num]
    diff_top = sorted(difference_medians)[0:num]

    return male_top, female_top, diff_top


def get_p_vals(location_median_results, author_gender_median_results, date_median_results):
    """
    Takes results from results_by_location(results, 'median'), results_by_author_gender,
    results_by_date
    ANOVA test for independence of:
        - male vs female authors' median distance between female instances
        - UK vs. US vs. other country authors' median distance between female instances
        - Date ranges authors' median distance between female instances
    :param location_median_results: result of results_by_location(results, 'median')
    :param author_gender_median_results: result of results_by_author_gender(results, 'median)
    :param date_median_results: result of results_by_date(results, 'median')
    :return: data-frame with 3 p-values, one for each category comparison
    """

    r1 = location_median_results
    r2 = author_gender_median_results
    r3 = date_median_results

    names = ["location", "male_vs_female_authors", "date"]
    # median_distance_between_female_pronouns_pvals = []

    location_medians = []
    author_gender_medians = []
    date_medians = []

    med = [location_medians, author_gender_medians, date_medians]
    res = [r1, r2, r3]

    for r in range(0, 3):
        for key in list(res[r].keys()):
            medians = []
            for el in list(res[r][key]):
                medians.append(el[1])
            med[r].append(medians)
    _, location_pval = stats.f_oneway(location_medians[0], location_medians[1])
    _, author_gender_pval = stats.f_oneway(author_gender_medians[0], author_gender_medians[1])
    _, date_pval = stats.f_oneway(*date_medians)
    median_distance_between_female_pronouns_pvals = [location_pval, author_gender_pval, date_pval]

    return pnds.DataFrame({"names": names, "pvals": median_distance_between_female_pronouns_pvals})


def box_plots(inst_data, my_pal, title, x="N/A"):
    """
    Takes in a frequency dictionaries and exports its values as a bar-and-whisker graph
    :param inst_data
    :param my_pal: str, palette to be used
    :param title: str, filename of exported graph
    :param x: name of x-vars
    :return:
    """
    plt.clf()
    groups = []
    val = []
    for k, v in inst_data.items():
        temp1 = []
        for el in v:
            if el[1] <= 60:
                temp1.append(el[1])
        temp2 = [k.replace("_", " ").capitalize()]*len(temp1)
        val.extend(temp1)
        groups.extend(temp2)
    df = pnds.DataFrame({x: groups, 'Median Female Instance Distance': val})
    df = df[[x, 'Median Female Instance Distance']]
    sns.boxplot(x=df[x], y=df['Median Female Instance Distance'],
                palette=my_pal).set_title(title)
    plt.xticks(rotation=90)
    # plt.show()

    filepng = "visualizations/" + title + ".png"
    filepdf = "visualizations/" + title + ".pdf"
    plt.savefig(filepng, bbox_inches='tight')
    plt.savefig(filepdf, bbox_inches='tight')


def process_medians(helst, shelst, authlst):
    """
    >>> medians_he = [12, 130, 0, 12, 314, 18, 15, 12, 123]
    >>> medians_she = [123, 52, 12, 345, 0,  13, 214, 12, 23]
    >>> books = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
    >>> process_medians(helst=medians_he, shelst=medians_she, authlst=books)
    {'he': [0, 2.5, 0, 1.3846153846153846, 0, 1.0, 5.3478260869565215], 'she': [10.25, 0, 28.75, 0, 14.266666666666667, 0, 0], 'book': ['a', 'b', 'd', 'f', 'g', 'h', 'i']}

    :param helst:
    :param shelst:
    :param authlst:
    :return: a dictionary sorted as so {
                                        "he":[ratio of he to she if >= 1, else 0], "she":[ratio of she to he if > 1, else 0] "book":[lst of book authors]
                                       }
    """
    d = {"he": [], "she": [], "book": []}
    for num in range(len(helst)):
        if helst[num] > 0 and shelst[num] > 0:
            res = helst[num] - shelst[num]
            if res >= 0:
                d["he"].append(helst[num] / shelst[num])
                d["she"].append(0)
                d["book"].append(authlst[num])
            else:
                d["he"].append(0)
                d["she"].append(shelst[num] / helst[num])
                d["book"].append(authlst[num])
        else:
            if helst == 0:
                print("ERR: no MALE values: " + authlst[num])
            if shelst == 0:
                print("ERR: no FEMALE values: " + authlst[num])
    return d


def bubble_sort_across_lists(dictionary):
    """
    >>> d = {'he': [0, 2.5, 0, 1.3846153846153846, 0, 1.0, 5.3478260869565215],
    ...     'she': [10.25, 0, 28.75, 0, 14.266666666666667, 0, 0],
    ...     'book': ['a', 'b', 'd', 'f', 'g', 'h', 'i']}
    >>> bubble_sort_across_lists(d)
    {'he': [5.3478260869565215, 2.5, 1.3846153846153846, 1.0, 0, 0, 0], 'she': [0, 0, 0, 0, 10.25, 14.266666666666667, 28.75], 'book': ['i', 'b', 'f', 'h', 'a', 'g', 'd']}

    :param dictionary: containing 3 different list values.
    Note: dictionary keys MUST contain arguments 'he', 'she', and 'book'
    :return dictionary sorted across all three lists in a specific method:
    1) Descending order of 'he' values
    2) Ascending order of 'she' values
    3) Corresponding values of 'book' values
    """
    lst1 = dictionary['he']
    lst2 = dictionary['she']
    lst3 = dictionary['book']
    r = range(len(lst1) - 1)
    p = True

    # sort by lst1 descending
    for j in r:
        for i in r:
            if lst1[i] < lst1[i + 1]:
                # manipulating lst 1
                temp1 = lst1[i]
                lst1[i] = lst1[i + 1]
                lst1[i + 1] = temp1
                # manipulating lst 2
                temp2 = lst2[i]
                lst2[i] = lst2[i + 1]
                lst2[i + 1] = temp2
                # manipulating lst of authors
                temp3 = lst3[i]
                lst3[i] = lst3[i + 1]
                lst3[i + 1] = temp3
                p = False
        if p:
            break
        else:
            p = True

    # sort by lst2 ascending
    for j in r:
        for i in r:
            if lst2[i] > lst2[i + 1]:
                # manipulating lst 1
                temp1 = lst1[i]
                lst1[i] = lst1[i + 1]
                lst1[i + 1] = temp1
                # manipulating lst 2
                temp2 = lst2[i]
                lst2[i] = lst2[i + 1]
                lst2[i + 1] = temp2
                # manipulating lst of authors
                temp3 = lst3[i]
                lst3[i] = lst3[i + 1]
                lst3[i + 1] = temp3
                p = False
        if p:
            break
        else:
            p = True
    d = {}
    d['he'] = lst1
    d['she'] = lst2
    d['book'] = lst3
    return d


def instance_stats(book, medians1, medians2, title):
    """
    :param book:
    :param medians1:
    :param medians2:
    :param title: str, desired name of file
    :return: None, file written to visualizations folder depicting the ratio of two values given
    as a bar graph
    """
    print(book, medians1, medians2)
    fig, ax = plt.subplots()
    plt.ylim(0, 50)

    index = np.arange(len(book))
    bar_width = .7
    opacity = 0.4

    medians_she = tuple(medians2)
    medians_he = tuple(medians1)
    book = tuple(book)

    rects1 = ax.bar(index, medians_he, bar_width, alpha=opacity, color='b', label='Male to Female')
    rects2 = ax.bar(index, medians_she, bar_width, alpha=opacity, color='r', label='Female to Male')

    ax.set_xlabel('Book')
    ax.set_ylabel('Ratio of Median Values')
    ax.set_title(
        'MtF or FtM Ratio of Median Distance of Gendered Instances by Author')
    ax.set_xticks(index)
    plt.xticks(fontsize=8, rotation=90)
    ax.set_xticklabels(book)
    ax.set_yscale("symlog")

    ax.legend()

    fig.tight_layout()
    filepng = "visualizations/" + title + ".png"
    filepdf = "visualizations/" + title + ".pdf"
    plt.savefig(filepng, bbox_inches='tight')
    plt.savefig(filepdf, bbox_inches='tight')


def run_dist_inst(corpus):
    """
    Runs a program that uses the instance distance analysis on all documents existing in a given
    corpus, and outputs the data as graphs
    :param corpus:
    :return:
    """
    documents = corpus.documents
    c = len(documents)
    # loops = c//10 + 1
    loops = c//10 if c % 10 == 0 else c//10+1

    num = 0

    while num < loops:
        medians_he = []
        medians_she = []
        books = []
        for document in documents[num * 10: min(c, num * 10 + 9)]:
            result_he = instance_dist(document, "he")
            result_she = instance_dist(document, "she")
            try:
                medians_he.append(median(result_he))
            except:
                medians_he.append(0)
            try:
                medians_she.append(median(result_she))
            except:
                medians_she.append(0)
            title = hasattr(document, 'title')
            author = hasattr(document, 'author')
            if title and author:
                books.append(document.title[0:20] + "\n" + document.author)
            elif title:
                books.append(document.title[0:20])
            else:
                books.append(document.filename[0:20])
        d = process_medians(helst=medians_he, shelst=medians_she, authlst=books)
        d = bubble_sort_across_lists(d)
        instance_stats(d["book"], d["he"], d["she"], "inst_dist" + str(num))
        num += 1
