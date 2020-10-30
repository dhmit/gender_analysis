from statistics import median, mean

import matplotlib.pyplot as plt
import numpy as np
import pandas as pnds
import seaborn as sns

from gender_analysis import common


def instance_dist(document, word):
    """
    Takes in a particular word, returns a list of distances between each instance of that word in
    the document.

    :param document: Document object to analyze
    :param word: str, word to search text for distance between occurrences
    :return: list of distances between consecutive instances of word

    >>> from gender_analysis import document
    >>> from pathlib import Path
    >>> from gender_analysis import common
    >>> document_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter', 'date': '1966',
    ...                   'filename': 'test_text_3.txt', 'filepath': Path(common.TEST_DATA_PATH, 'document_test_files', 'test_text_3.txt')}
    >>> scarlett = document.Document(document_metadata)
    >>> instance_dist(scarlett, "her")
    [6, 5, 6, 7, 7]

    """
    return words_instance_dist(document, [word])


def words_instance_dist(document, words):
    """
    Takes in a document and list of words (e.g. gendered pronouns), returns a list of distances
    between each instance of one of the words in that document

    :param document: Document object
    :param words: list of strings; words to search for in the document
    :return: list of distances between instances of any word in the supplied list

    >>> from gender_analysis import document
    >>> from pathlib import Path
    >>> from gender_analysis import common
    >>> document_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter', 'date': '1966',
    ...                   'filename': 'test_text_4.txt', 'filepath': Path(common.TEST_DATA_PATH, 'document_test_files', 'test_text_4.txt')}
    >>> scarlett = document.Document(document_metadata)
    >>> words_instance_dist(scarlett, ["his", "him", "he", "himself"])
    [6, 5, 6, 6, 7]

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
    Takes in a document, returns a list of distances between each instance of a male pronoun
    in that document.

    :param: Document object
    :return: list of distances between instances of gendered words

    >>> from gender_analysis import document
    >>> from pathlib import Path
    >>> from gender_analysis import common
    >>> document_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter', 'date': '1966',
    ...                   'filename': 'test_text_5.txt', 'filepath': Path(common.TEST_DATA_PATH, 'document_test_files', 'test_text_5.txt')}
    >>> scarlett = document.Document(document_metadata)
    >>> male_instance_dist(scarlett)
    [6, 5, 6, 6, 7]

    """
    return words_instance_dist(document, common.HE_SERIES)


def female_instance_dist(document):
    """
    Takes in a document, returns a list of distances between each instance of a female pronoun
    in that document.

    :param: Document object
    :return: list of distances between instances of gendered word

    >>> from gender_analysis import document
    >>> from pathlib import Path
    >>> from gender_analysis import common
    >>> document_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter', 'date': '1966',
    ...                   'filename': 'test_text_6.txt', 'filepath': Path(common.TEST_DATA_PATH, 'document_test_files', 'test_text_6.txt')}
    >>> scarlett = document.Document(document_metadata)
    >>> female_instance_dist(scarlett)
    [6, 5, 6, 6, 7]

    """
    return words_instance_dist(document, common.SHE_SERIES)


def run_distance_analysis(corpus, genders=None):
    """
    Takes in a corpus of documents. Return a dictionary with each document mapped in the following
    form:
    ```
    {
        <Document object>: {
            <Gender object>: {
                'median': float,
                'mean: float,
                'min': float,
                'max': float
            }
        }
    }
    ```

    :param corpus: Corpus object
    :param genders: A collection of genders to perform analyses on. If `None`, defaults to an \
        analysis on `common.MALE` and `common.FEMALE`
    :return: Nested dictionary, first mapping documents and next mapping genders to their stats.

    """
    results = {}

    if genders is None:
        genders = {common.MALE, common.FEMALE}

    for document in corpus:
        document_stats = dict()

        for gender in genders:
            gender_results = words_instance_dist(document, gender.identifiers)
            document_stats[gender] = _get_stats(gender_results)

        results[document] = document_stats

    return results


def _get_stats(distance_results):
    """
    Returns a dictionary of data from a given set of instance distance data, with keys of 'median',
    'mean', 'min', and 'max'.

    :param distance_results: Instance Distance data as a list
    :return: dictionary of stats

    """
    if len(distance_results) == 0:
        return {'median': 0, 'mean': 0, 'min': 0, 'max': 0}
    else:
        return {'median': median(distance_results), 'mean': mean(distance_results), 'min': min(
            distance_results), 'max': max(distance_results)}


def _get_document_gender_metrics(document_results, metric):
    """
    This is a helper function for `results_by` functions.

    Pulls out the given metric from a document's min/max/etc. dictionary and maps each gender
    as a key.

    :param document_results: A dictionary mapping different fields to a dictionary of min/max/etc.
    :param metric: The metric to extract from results
    :return: A dictionary mapping each gender from `document_results` to the value of the metric for
        that field
    """

    metric_list = dict()
    for field in document_results:
        metric_list[field] = document_results[field][metric]

    return metric_list


def results_by_author_gender(results, metric):
    """
    Takes in a dictionary of results and a specified metric from **run_distance_analysis**, returns
    a dictionary in the following form:

    ```
    {
        'author_gender': {
            <Document Object>: {
                <Gender Object>: metric_result
            }
        }
    }
    ```

    :param results: dictionary from **run_distance_analysis**
    :param metric: one of ('median', 'mean', 'min', 'max')
    :return: dictionary

    """
    data = dict()
    if metric not in {'median', 'mean', 'min', 'max'}:
        raise ValueError(f"{metric} is not valid metric name. Valid names: 'median', 'mean', "
                         f"'min', 'max'")

    for document in results:

        # Skip the document if it doesn't have a defined author gender
        author_gender = getattr(document, 'author_gender', None)
        if author_gender is None:
            continue

        if author_gender not in data:
            data[author_gender] = dict()

        data[author_gender][document] = _get_document_gender_metrics(results[document], metric)

    return data


def results_by_year(results, metric, time_frame, bin_size):
    """
    Takes in a dictionary of results and a specified metric from run_distance_analysis, returns a
    dictionary in the following form:

    ```
    {
        year: {
            <Document Object>: {
                <Gender Object>: metric_result
            }
        }
    }
    ```

    :param results: dictionary
    :param metric: ('median', 'mean', 'min', 'max')
    :param time_frame: tuple (int start year, int end year) for the range of dates to return
        frequencies
    :param bin_size: int for the number of years represented in each document dictionary
    :return: nested dictionary, mapping integer years->`Document`s->`Gender`s->float results

    """

    if metric not in {'median', 'mean', 'min', 'max'}:
        raise ValueError(f"{metric} is not valid metric name. Valid names: 'median', 'mean', "
                         f"'min', 'max'")

    data = {}
    for bin_start_year in range(time_frame[0], time_frame[1], bin_size):
        data[bin_start_year] = dict()

    for document in results.keys():

        date = getattr(document, 'date', None)
        if date is None:
            continue

        bin_year = ((date - time_frame[0]) // bin_size) * bin_size + time_frame[0]
        data[bin_year][document] = _get_document_gender_metrics(results[document], metric)

    return data


def results_by_location(results, metric):
    """
    Takes in a dictionary of results and a specified metric from **run_distance_analysis**, returns
    a dictionary of the following form:

    ```
    {
        'location': {
            <Document Object>: {
                <Gender Object>: metric_result
            }
        }
    }
    ```

    :param results: dictionary
    :param metric: ('median', 'mean', 'min', 'max')
    :return: dictionary

    """
    data = {}
    metric_indexes = {"median": 0, "mean": 2, "min": 3, "max": 4}
    if metric not in metric_indexes:
        raise ValueError(f"{metric} is not valid metric name. Valid names: 'median', 'mean', "
                         f"'min', 'max'")

    for document in results.keys():

        location = getattr(document, 'country_publication', None)
        if location is None:
            continue

        if location not in data:
            data[location] = dict()

        data[location][document] = _get_document_gender_metrics(results[document], metric)

    return data


def get_highest_distances(results, num):
    """
    Finds the documents with the largest median distances between each gender that was analyzed.

    Returns a dictionary mapping genders to a list of tuples of the form (<Document>, median),
    where `Document`s with higher medians are listed first.

    :param results: dictionary of results from `run_distance_analysis`
    :param num: number of top distances to return
    :return: Dictionary of lists of tuples.

    """

    medians = dict()

    # Get all of the medians for the documents
    for document in results:
        for gender in results['document']:
            if gender not in medians:
                medians[gender] = list()

            medians[gender].append((results[document][gender]['median'], document))

    # Pull out the top medians
    top_medians = dict()
    for gender in medians:
        top_medians[gender] = sorted(medians[gender], reverse=True)[0:num]

    return top_medians


def box_plots(inst_data, palette, title, x="N/A"):
    """
    Takes in a frequency dictionaries and exports its values as a bar-and-whisker graph

    :param inst_data: Dictionary containing instance distance data (from one of the other functions in the module)
    :param palette: str, seaborn palette to be used
    :param title: str, filename of exported graph
    :param x: name of x-vars
    :return: None

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
    common.load_graph_settings()
    sns.boxplot(x=df[x], y=df['Median Female Instance Distance'],
                palette=palette).set_title(title)
    plt.xticks(rotation=90)
    # plt.show()

    filepng = "visualizations/" + title + ".png"
    filepdf = "visualizations/" + title + ".pdf"
    plt.savefig(filepng, bbox_inches='tight')
    plt.savefig(filepdf, bbox_inches='tight')


def process_medians(gen_1_dist, gen_2_dist, authlst):
    """
    :param gen_1_dist: List of gender_1 instance distances
    :param gen_2_dist: List of gender_2 instance distances
    :param authlst:
    :return: a dictionary sorted as so {
        "he":[ratio of he to she if >= 1, else 0], "she":[ratio of she to he if > 1, else 0] "book":[lst of book authors]
        }

    >>> medians_he = [12, 130, 0, 12, 314, 18, 15, 12, 123]
    >>> medians_she = [123, 52, 12, 345, 0,  13, 214, 12, 23]
    >>> books = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
    >>> process_medians(gen_1_dist=medians_he, gen_2_dist=medians_she, authlst=books)
    {'gen_1': [0, 2.5, 0, 1.3846153846153846, 0, 1.0, 5.3478260869565215], 'gen_2': [10.25, 0, 28.75, 0, 14.266666666666667, 0, 0], 'book': ['a', 'b', 'd', 'f', 'g', 'h', 'i']}

    """
    d = {"gen_1": [], "gen_2": [], "book": []}
    for num in range(len(gen_1_dist)):
        if gen_1_dist[num] > 0 and gen_2_dist[num] > 0:
            res = gen_1_dist[num] - gen_2_dist[num]
            if res >= 0:
                d["gen_1"].append(gen_1_dist[num] / gen_2_dist[num])
                d["gen_2"].append(0)
                d["book"].append(authlst[num])
            else:
                d["gen_1"].append(0)
                d["gen_2"].append(gen_2_dist[num] / gen_1_dist[num])
                d["book"].append(authlst[num])
        else:
            if gen_1_dist == 0:
                print("ERR: no gen_1 values: " + authlst[num])
            if gen_2_dist == 0:
                print("ERR: no gen_2 values: " + authlst[num])
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

    fig, ax = plt.subplots()
    plt.ylim(0, 50)

    index = np.arange(len(book))
    bar_width = .7
    opacity = 0.4

    medians_gen_2 = tuple(medians2)
    medians_gen_1 = tuple(medians1)
    book = tuple(book)

    ax.bar(index, medians_gen_1, bar_width, alpha=opacity, color='b', label='Gender 1 to Gender 2')
    ax.bar(index, medians_gen_2, bar_width, alpha=opacity, color='r', label='Gender 2 to Gender 1')

    ax.set_xlabel('Book')
    ax.set_ylabel('Ratio of Median Values')
    ax.set_title(
        'Ratio of Median Distance of Gendered Instances by Author')
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


def run_dist_inst(corpus, genders):
    """
    Runs a program that uses the instance distance analysis on all documents existing in a given
    corpus, and outputs the data as graphs

    :param corpus: Corpus object
    :param genders: Collection of `Gender`s to perform the analysis on
    :return: None

    """
    documents = corpus.documents
    c = len(documents)
    loops = c//10 if c % 10 == 0 else c//10+1

    num = 0

    while num < loops:
        medians_he = []
        medians_she = []
        books = []
        for document in documents[num * 10: min(c, num * 10 + 9)]:
            result_he = male_instance_dist(document)
            result_she = female_instance_dist(document)
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
        d = process_medians(gen_1_dist=medians_he, gen_2_dist=medians_she, authlst=books)
        instance_stats(d["book"], d["gen_1"], d["gen_2"], "inst_dist" + str(num))
        num += 1
