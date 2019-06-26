import matplotlib.pyplot as plt
import seaborn as sns

from gender_analysis.common import load_graph_settings
from gender_analysis.analysis.dunning import male_vs_female_authors_analysis_dunning, dunning_result_to_dict

load_graph_settings(False)


def score_plot_to_show(results):
    results_dict = dict(results)
    words = []
    dunning_score = []

    for term, data in results_dict.items():
        words.append(term)
        dunning_score.append(data['dunning'])

    opacity = 0.4

    colors = ['r' if entry >= 0 else 'b' for entry in dunning_score]
    ax = sns.barplot(dunning_score, words, palette=colors, alpha=opacity)
    sns.despine(ax=ax, bottom=True, left=True)
    plt.show()


def freq_plot_to_show(results):
    results_dict = dict(results)
    words = []
    female_rel_freq = []
    male_rel_freq = []

    for term, data in results_dict.items():
        words.append(term)
        female_rel_freq.append(data['freq_corp1']/data['freq_total'])
        male_rel_freq.append(-1*data['freq_corp2']/data['freq_total'])

    opacity = 0.4

    colors = ['b']
    ax = sns.barplot(male_rel_freq, words, palette=colors, alpha=opacity)
    sns.despine(ax=ax, bottom=True, left=True)
    plt.show()


if __name__ == '__main__':
    '''
    from gender_analysis.corpus import Corpus
    from gender_analysis.common import BASE_PATH
    filepath = BASE_PATH / 'corpora' / 'sample_novels' / 'texts'
    csv_path = BASE_PATH / 'corpora' / 'sample_novels' / 'sample_novels.csv'
    sample = Corpus(filepath, csv_path=csv_path)

    analysis_results_unsorted = dunning_result_to_dict(male_vs_female_authors_analysis_dunning(
        sample), part_of_speech_to_include="verbs")
    analysis_results_sorted = sorted(analysis_results_unsorted.items(), key=lambda x: x[1][
        'dunning'], reverse=True)
    score_plot_to_show(analysis_results_sorted)
    '''
