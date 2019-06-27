import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


def plt_pubyears(pub_years, corpus_name):
    """
    Creates a histogram displaying the frequency of books that were published within a 20 year 
    period
    :param pub_years: list of all years to be included in the plot
    :param corpus_name: str
    RETURNS a pyplot histogram
    """
    sns.set_style('ticks')
    sns.color_palette('colorblind')
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    plt.figure(figsize=(10, 6))
    bins = [num for num in range(min(pub_years), max(pub_years)+4, 5)]
    plt.hist(pub_years, bins, histtype='bar', rwidth=.8, color='c')
    plt.xlabel('Year', size=15, weight='bold', color='k')
    plt.ylabel('Frequency', size=15, weight='bold', color='k')
    plt.title('Publication Year Concentration for '+corpus_name.title(), size=18, weight='bold',
              color='k')
    plt.yticks(size=15, color='k')
    plt.xticks([i for i in range(min(pub_years), max(pub_years)+9, 10)], size=15, color='k')
    for label in ax1.xaxis.get_ticklabels():
        label.set_rotation(60)
    plt.subplots_adjust(left=.1, bottom=.18, right=.95, top=.9)
    plt.savefig('date_of_pub_for_'+corpus_name.replace(' ', '_')+'.png')


def plt_pubcountries(pub_country, corpus_name):
    """
    Creates a bar graph displaying the frequency of books that were published in each country
    :param pub_country: list of all countries to be included in the plot
    :param corpus: Corpus
    RETURNS a pyplot bargraph
    """
    sns.set_style('ticks')
    sns.color_palette('colorblind')
    plt.figure(figsize=(10, 6))
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    country_counter = {}
    totalbooks = 0
    for country in pub_country:
        country_counter[country] = country_counter.setdefault(country, 0)+1
        totalbooks += 1
    country_counter2 = {'Other': 0}
    for country in country_counter:
        if country == '':
            pass
        elif country_counter[country] > (.001*totalbooks):
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
    plt.title('Country of Publication for '+corpus_name.title(), size=18, color='k',
              weight='bold')
    plt.xticks(color='k', size=15)
    plt.yticks(color='k', size=15)
    plt.subplots_adjust(left=.1, bottom=.18, right=.95, top=.9)
    plt.savefig('country_of_pub_for_'+corpus_name.replace(' ', '_')+'.png')


def plt_gender_breakdown(pub_gender, corpus_name):
    """
    Creates a pie chart displaying the composition of male and female writers in the data
    :param pub_gender: list of all author genders to be included in the plot
    :param corpus: Corpus
    RETURNS a pie chart
    """
    sns.set_color_codes('colorblind')
    gendercount = {}
    for i in pub_gender:
        if i == 'both' or i == 'unknown' or i == 'Both' or i == 'Unknown':
            gendercount['Unknown'] = gendercount.setdefault('Unknown', 0)+1
        else:
            gendercount[i] = gendercount.setdefault(i, 0)+1
    total = 0
    for i in gendercount:
        total += gendercount[i]
    slices = [gendercount[i]/total for i in gendercount]
    genders = [i for i in gendercount]
    labelgenders = []
    for i in range(len(genders)):
        labelgenders.append((genders[i]+': ' + str(int(round(slices[i], 2)*100))+'%').title())
    colors = ['c', 'b', 'g']
    plt.figure(figsize=(10, 6))
    plt.pie(slices, colors=colors, labels=labelgenders, textprops={'fontsize': 15})
    plt.title('Gender Breakdown for '+corpus_name.title(), size=18, color='k', weight='bold')
    plt.legend()
    plt.subplots_adjust(left=.1, bottom=.1, right=.9, top=.9)
    plt.savefig('gender_breakdown_for_'+corpus_name.replace(' ', '_')+'.png')


def plt_metadata_pie(corpus, corpus_name=None):
    """
    Creates pie chart indicating fraction of metadata that is filled in corpus
    :param corpus: Corpus
    :param corpus_name: str, plot title and filename use corpus_name if give
    """
    if corpus.name is not None:
        name = corpus.name
    elif corpus_name is not None:
        name = corpus_name
    else:
        name = 'corpus'

    counter = Counter({'Both Country and Gender': 0, 'Author Gender Only': 0,
                       'Country Only': 0, 'Neither': 0})
    num_documents = len(corpus)
    for doc in corpus.documents:
        if doc.author_gender != 'unknown' and doc.country_publication:
            counter['Both Country and Gender'] += 1
        elif doc.author_gender != 'unknown':
            counter['Author Gender Only'] += 1
        elif doc.country_publication:
            counter['Country Only'] += 1
        else:
            counter['Neither'] += 1
    labels = []
    for label, number in counter.items():
        labels.append(label + " " + str(int(round(number/num_documents, 2)*100)) + r"%")
    sns.set_color_codes('colorblind')
    colors = ['c', 'b', 'g', 'w']
    plt.figure(figsize=(10, 6))
    plt.pie(counter.values(), colors=colors, labels=labels, textprops={'fontsize': 13})
    plt.title('Percentage Acquired Metadata for ' + name.title(), size=18, color='k',
              weight='bold')
    plt.legend()
    plt.subplots_adjust(left=.1, bottom=.1, right=.9, top=.9)
    plt.savefig('percentage_acquired_metadata_for_' + name.replace(' ', '_') + '.png')


def create_corpus_summary_visualizations(corpus):
    """
    Runs through all plt functions given a corpus name
    :param corpus: Corpus
    """
    pubyears = [doc.date for doc in corpus.documents]
    pubgender = [doc.author_gender for doc in corpus.documents]
    pubcountry = [doc.country_publication for doc in corpus.documents]
    if corpus.name is not None:
        corpus_name = corpus.name
    else:
        corpus_name = 'corpus'

    plt_gender_breakdown(pubgender, corpus_name)
    plt_pubyears(pubyears, corpus_name)
    plt_pubcountries(pubcountry, corpus_name)
    plt_metadata_pie(corpus)


if __name__ == '__main__':
    '''
    from gender_analysis.corpus import Corpus
    from gender_analysis.common import BASE_PATH
    path = BASE_PATH / 'corpora' / 'sample_novels' / 'texts'
    csv_path = BASE_PATH / 'corpora' / 'sample_novels' / 'sample_novels.csv'
    sample = Corpus(path, csv_path=csv_path)
    create_corpus_summary_visualizations(sample)
    '''
