from statistics import median, mean

import matplotlib.pyplot as plt
import pandas as pnds
from scipy import stats
import seaborn as sns
sns.set()

from gender_analysis import common
from gender_analysis.analysis.analysis import male_instance_dist, female_instance_dist


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
    try:
        stat = metric_indexes[metric]
    except KeyError:
        raise ValueError("Not valid metric name. Valid names: 'median', 'mean', 'min', 'max'")
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
    try:
        stat = metric_indexes[metric]
    except KeyError:
        print("Not valid metric name. Valid names: 'median', 'mean', 'min', 'max'")

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
    try:
        stat = metric_indexes[metric]
    except:
        print("Not valid metric name. Valid names: 'median', 'mean', 'min', 'max'")

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
