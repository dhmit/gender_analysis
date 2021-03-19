from statistics import median, mean

from gender_analysis import common


def instance_dist(document, word):
    """
    Takes in a particular word, returns a list of distances between each instance of that word in
    the document.

    :param document: Document object to analyze
    :param word: str, word to search text for distance between occurrences
    :return: list of distances between consecutive instances of word

    >>> from corpus_analysis import document
    >>> from pathlib import Path
    >>> from gender_analysis import common
    >>> document_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter',
    ... 'date': '1966', 'filename': 'test_text_3.txt',
    ... 'filepath': Path(common.TEST_DATA_PATH, 'document_test_files', 'test_text_3.txt')}
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

    >>> from corpus_analysis import document
    >>> from pathlib import Path
    >>> from gender_analysis import common
    >>> document_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter',
    ... 'date': '1966', 'filename': 'test_text_4.txt',
    ... 'filepath': Path(common.TEST_DATA_PATH, 'document_test_files', 'test_text_4.txt')}
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

    >>> from corpus_analysis import document
    >>> from pathlib import Path
    >>> from gender_analysis import common
    >>> document_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter',
    ... 'date': '1966', 'filename': 'test_text_5.txt',
    ... 'filepath': Path(common.TEST_DATA_PATH, 'document_test_files', 'test_text_5.txt')}
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

    >>> from corpus_analysis import document
    >>> from pathlib import Path
    >>> from gender_analysis import common
    >>> document_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter',
    ... 'date': '1966', 'filename': 'test_text_6.txt',
    ... 'filepath': Path(common.TEST_DATA_PATH, 'document_test_files', 'test_text_6.txt')}
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
