from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter

from gender_analysis.common import MissingMetadataError
from gender_analysis.common import create_path_object_and_directories

DEFAULT_VISUALIZATION_OUTPUT_DIR = Path(os.getcwd()).joinpath(
    Path("gender_analysis_visualizations"))


def plot_pubyears(corpus, output_dir=DEFAULT_VISUALIZATION_OUTPUT_DIR, filename=None):
    """
    Creates a histogram displaying the frequency of books that were published within a 20 year
    period.

    *NOTE:* Requires that corpus contains a 'date' metadata field.

    :param corpus: Corpus object
    :param output_dir: path for where to save the file
    :param filename: Name of file to save plot as; will not write a file if None
    :return: None

    """

    if 'date' not in corpus.metadata_fields:
        raise MissingMetadataError(['date'])

    pub_years = []
    for doc in corpus.documents:
        if doc.date is None:
            continue
        pub_years.append(doc.date)

    if corpus.name:
        corpus_name = corpus.name
    else:
        corpus_name = 'corpus'

    sns.set_style('ticks')
    sns.color_palette('colorblind')
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    plt.figure(figsize=(10, 6))
    bins = [num for num in range(min(pub_years), max(pub_years) + 4, 5)]
    plt.hist(pub_years, bins, histtype='bar', rwidth=.8, color='c')
    plt.xlabel('Year', size=15, weight='bold', color='k')
    plt.ylabel('Frequency', size=15, weight='bold', color='k')
    plt.title('Publication Year Concentration for ' + corpus_name.title(), size=18, weight='bold',
              color='k')
    plt.yticks(size=15, color='k')
    plt.xticks([i for i in range(min(pub_years), max(pub_years) + 9, 10)], size=15, color='k')
    for label in ax1.xaxis.get_ticklabels():
        label.set_rotation(60)
    plt.subplots_adjust(left=.1, bottom=.18, right=.95, top=.9)

    if filename:
        plt.savefig(create_path_object_and_directories(output_dir,
                                                       filename.replace(' ', '_') + '.png'))
    else:
        plt.savefig(create_path_object_and_directories(output_dir, 'date_of_pub_for_'
                    + corpus_name.replace(' ', '_') + '.png'))


def plot_pubcountries(corpus, output_dir=DEFAULT_VISUALIZATION_OUTPUT_DIR, filename=None):
    """
    Creates a bar graph displaying the frequency of books that were published in each country.

    *NOTE:* Requires that corpus contains a 'country_publication' metadata field.

    :param corpus: Corpus object
    :param output_dir: Path of where to save the plot
    :param filename: Name of file to save plot as; will not write a file if None
    :return: None

    """
    if 'country_publication' not in corpus.metadata_fields:
        raise MissingMetadataError(['country_publication'])

    pub_country = []
    for doc in corpus.documents:
        if doc.country_publication is None:
            continue
        pub_country.append(doc.country_publication)

    if corpus.name:
        corpus_name = corpus.name
    else:
        corpus_name = 'corpus'

    sns.set_style('ticks')
    sns.color_palette('colorblind')
    plt.figure(figsize=(10, 6))
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    country_counter = {}
    totalbooks = 0
    for country in pub_country:
        country_counter[country] = country_counter.setdefault(country, 0) + 1
        totalbooks += 1
    country_counter2 = {'Other': 0}
    for country in country_counter:
        if country == '':
            pass
        elif country_counter[country] > (.001 * totalbooks):
            # must be higher than .1% of the total books to have its own country name,
            # otherwise it is classified under others
            country_counter2[country] = country_counter[country]
        else:
            country_counter2['Other'] += country_counter[country]
    country_counter2 = sorted(country_counter2.items(), key=lambda kv: -kv[1])
    x = [country[0] for country in country_counter2]
    y = [country[1] for country in country_counter2]
    for label in ax1.xaxis.get_ticklabels():
        label.set_rotation(15)
    plt.bar(x, y, color='c')
    plt.xlabel('Countries', size=15, weight='bold', color='k')
    plt.ylabel('Frequency', size=15, weight='bold', color='k')
    plt.title('Country of Publication for ' + corpus_name.title(), size=18, color='k',
              weight='bold')
    plt.xticks(color='k', size=15)
    plt.yticks(color='k', size=15)
    plt.subplots_adjust(left=.1, bottom=.18, right=.95, top=.9)

    if filename:
        plt.savefig(create_path_object_and_directories(output_dir,
                                                       filename.replace(' ', '_') + '.png'))
    else:
        plt.savefig(create_path_object_and_directories(output_dir,
                                                       'country_of_pub_for_' +
                                                       corpus_name.replace(' ', '_') + '.png'))


def plot_gender_breakdown(corpus, output_dir=DEFAULT_VISUALIZATION_OUTPUT_DIR, filename=None):
    """
    Creates a pie chart displaying the composition of male and female writers in the data.

    *NOTE:* Requires that corpus contains a 'author_gender' metadata field.

    :param corpus: Corpus object
    :param output_dir: Path of where to save the plot
    :param filename: Name of file to save plot as; will not write a file if None
    :return: None

    """
    if 'author_gender' not in corpus.metadata_fields:
        raise MissingMetadataError(['author_gender'])

    pub_gender = []
    for doc in corpus.documents:
        if doc.author_gender is None:
            continue
        pub_gender.append(doc.author_gender)

    if corpus.name:
        corpus_name = corpus.name
    else:
        corpus_name = 'corpus'

    sns.set_color_codes('colorblind')
    gendercount = {}
    for i in pub_gender:
        if i == 'both' or i == 'unknown' or i == 'Both' or i == 'Unknown':
            gendercount['Unknown'] = gendercount.setdefault('Unknown', 0) + 1
        else:
            gendercount[i] = gendercount.setdefault(i, 0) + 1
    total = 0
    for i in gendercount:
        total += gendercount[i]
    slices = [gendercount[i] / total for i in gendercount]
    genders = [i for i in gendercount]
    labelgenders = []
    for i in range(len(genders)):
        labelgenders.append(
            (genders[i] + ': ' + str(int(round(slices[i], 2) * 100)) + '%').title())
    colors = ['c', 'b', 'g']
    plt.figure(figsize=(10, 6))
    plt.pie(slices, colors=colors, labels=labelgenders, textprops={'fontsize': 15})
    plt.title('Gender Breakdown for ' + corpus_name.title(), size=18, color='k', weight='bold')
    plt.legend()
    plt.subplots_adjust(left=.1, bottom=.1, right=.9, top=.9)

    if filename:
        plt.savefig(create_path_object_and_directories(output_dir,
                                                       filename.replace(' ', '_') + '.png'))
    else:
        plt.savefig(create_path_object_and_directories(output_dir, 'gender_breakdown_for_' +
                    corpus_name.replace(' ', '_') + '.png'))


def plot_metadata_pie(corpus, output_dir=DEFAULT_VISUALIZATION_OUTPUT_DIR, filename=None):
    """
    Creates a pie chart indicating the fraction of metadata that is filled in the corpus.

    *NOTE:* Requires that corpus contains 'author_gender' and 'country_publication' metadata fields.

    :param corpus: Corpus object
    :param output_dir: Path of where to save the plot
    :param filename: Name of file to save plot as; will not write a file if None
    :return: None

    """

    if ('author_gender' not in corpus.metadata_fields
            or 'country_publication' not in corpus.metadata_fields):
        raise MissingMetadataError(['author_gender', 'country_publication'])

    if corpus.name:
        name = corpus.name
    else:
        name = 'corpus'

    counter = Counter({'Both Country and Gender': 0, 'Author Gender Only': 0,
                       'Country Only': 0, 'Neither': 0})
    num_documents = len(corpus)
    for doc in corpus.documents:
        if doc.author_gender and doc.author_gender != 'unknown' and doc.country_publication:
            counter['Both Country and Gender'] += 1
        elif doc.author_gender and doc.author_gender != 'unknown':
            counter['Author Gender Only'] += 1
        elif doc.country_publication:
            counter['Country Only'] += 1
        else:
            counter['Neither'] += 1
    labels = []
    for label, number in counter.items():
        labels.append(label + " " + str(int(round(number / num_documents, 2) * 100)) + r"%")
    sns.set_color_codes('colorblind')
    colors = ['c', 'b', 'g', 'w']
    plt.figure(figsize=(10, 6))
    plt.pie(counter.values(), colors=colors, labels=labels, textprops={'fontsize': 13})
    plt.title('Percentage Acquired Metadata for ' + name.title(), size=18, color='k',
              weight='bold')
    plt.legend()
    plt.subplots_adjust(left=.1, bottom=.1, right=.9, top=.9)

    if filename:
        plt.savefig(create_path_object_and_directories(output_dir,
                                                       filename.replace(' ', '_') + '.png'))
    else:
        plt.savefig(create_path_object_and_directories(output_dir,
                                                       'percentage_acquired_metadata_for_' +
                                                       name.replace(' ', '_') + '.png'))


def create_corpus_summary_visualizations(corpus, output_dir=DEFAULT_VISUALIZATION_OUTPUT_DIR):
    """
    Creates graphs and summarizes gender breakdowns, publishing years, countries of origin, and
    overall metadata completion of a given corpus.


    :param corpus: Corpus object
    :param output_dir: Path of where to put the files
    :return: None

    """
    plot_gender_breakdown(corpus, output_dir)
    plot_pubyears(corpus, output_dir)
    plot_pubcountries(corpus, output_dir)
    plot_metadata_pie(corpus, output_dir)
