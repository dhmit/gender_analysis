# Pronoun Adjective Analysis

def run_analysis(corpus_name):
    print("loading corpus", corpus_name)
    corpus = Corpus(corpus_name)
    novels = corpus.novels

    print("running analysis")
    results = run_adj_analysis(novels)

    print("storing results")
    store_raw_results(results, corpus_name)

    print("loading results")
    r = common.load_pickle("pronoun_adj_raw_analysis_"+corpus_name)
    print("merging and getting final results")
    m = merge_raw_results(r)
    print("getting final results")
    final = get_overlapping_adjectives_raw_results(m)
    print("storing final results")
    common.store_pickle(final, "pronoun_adj_final_results_"+corpus_name)

    #Comment out pprint for large databases where it's not practical to print out results
    #pprint(final)
    r = common.load_pickle("pronoun_adj_raw_analysis_" + corpus_name)
    print("getting results by location")

    r2 = results_by_location(r)
    print("storing 1")
    common.store_pickle(r2, "pronoun_adj_by_location")
    print("getting results by author gender")
    r3 = results_by_author_gender(r)
    print("storing 2")
    common.store_pickle(r3, "pronoun_adj_by_author_gender")
    print("getting results by date")
    r4 = results_by_date(r)
    print("storing 3")
    common.store_pickle(r4, "pronoun_adj_by_date")
    print("DONE")

    r = common.load_pickle("pronoun_adj_raw_analysis_" + corpus_name)
    print("merging and getting final results")
    m = merge_raw_results(r)
    common.store_pickle(m, "pronoun_adj_merged_results_" + corpus_name)

    male_top, female_top = get_top_adj(corpus_name, 30)
    pprint("MALE TOP")
    pprint(male_top)
    pprint("FEMALE TOP")
    pprint(female_top)

# Instance Distance Analysis

def run_analysis(corpus_name):
    """
    Run instance distance analyses on a particular corpus and saves results as pickle files.
    Comment out sections of code or analyses that have already been run or are unnecessary.
    :param corpus_name:
    :return:
    """

    print('loading corpus')
    corpus = Corpus(corpus_name)
    novels = corpus.novels

    print('running analysis')
    results = run_distance_analysis(novels)

    print('storing results')
    store_raw_results(results, corpus_name)

    r = common.load_pickle("instance_distance_raw_analysis_"+corpus_name)
    r2 = results_by_location(r, "mean")
    r3 = results_by_author_gender(r, "mean")
    r4 = results_by_date(r, "median")
    r5 = results_by_location(r, "median")
    r6 = results_by_author_gender(r, "median")
    r7 = results_by_date(r, "median")

    common.store_pickle(r2, "mean_instance_distances_by_location_"+corpus_name)
    common.store_pickle(r3, "mean_instance_distances_by_author_gender_"+corpus_name)
    common.store_pickle(r4, "mean_instance_distances_by_date_"+corpus_name)

    common.store_pickle(r5, "median_instance_distances_by_location_"+corpus_name)
    common.store_pickle(r6, "median_instance_distances_by_author_gender_"+corpus_name)
    common.store_pickle(r7, "median_instance_distances_by_date_"+corpus_name)

    pvals = get_p_vals("gutenberg")
    common.store_pickle(pvals, "instance_distance_comparison_pvals")

    male_top_twenty, female_top_twenty, diff_top_twenty = get_highest_distances("gutenberg", 20)
    top_twenties = {'male_pronoun_top_twenty': male_top_twenty, 'female_pronoun_top_twenty': female_top_twenty,
                    "difference_top_twenty": diff_top_twenty}
    common.store_pickle(top_twenties, "instance_distance_top_twenties")

    inst_data = common.load_pickle("median_instance_distances_by_author_gender_" + corpus_name)
    box_plots(inst_data, "Blues", "Median Female Instance Distance by Author Gender", x="Author Gender")

    inst_data = common.load_pickle("median_instance_distances_by_location_"  + corpus_name)
    box_plots(inst_data, "Blues", "Median Female Instance Distance by Location", x="Location")

    inst_data = common.load_pickle("median_instance_distances_by_date_" + corpus_name)
    box_plots(inst_data, "Blues", "Median Female Instance Distance by Date", x="Date")


# Gender Pronoun Frequency Analysis
def run_all_analyses(corpus):
    '''
    Runs analyses for:
        Female and Male pronoun frequency for:
            author gender, publication date, publication, publication location
        Female and Male Subject Object frequency Comparison for:
            author gender, publication date, publication, publication location
    Prints results nicely
    :return: None
    '''
    all_data = books_pronoun_freq(corpus)

    gender = freq_by_author_gender(all_data)
    date = freq_by_date(all_data)
    location = freq_by_location(all_data)

    print('Male/Female pronoun comparison: ')
    print('By author gender: ')
    print(get_mean(gender))
    print('\n By date: ')
    print(get_mean(date))
    print('\n By location: ')
    print(get_mean(location))

    sub_v_ob = subject_vs_object_pronoun_freqs(corpus)

    female_gender_sub_v_ob = freq_by_author_gender(sub_v_ob[1])
    female_date_sub_v_ob = freq_by_date(sub_v_ob[1])
    female_loc_sub_v_ob = freq_by_location(sub_v_ob[1])

    male_gender_sub_v_ob = freq_by_author_gender(sub_v_ob[0])
    male_date_sub_v_ob = freq_by_date(sub_v_ob[0])
    male_loc_sub_v_ob = freq_by_location(sub_v_ob[0])

    male_tot = dict_to_list(sub_v_ob[0])
    female_tot = dict_to_list(sub_v_ob[1])

    print('Subject/Object comparisons: ')
    print('Male vs Female in the subject: ')
    print('Male: ')
    pprint.pprint(np.mean(male_tot))
    print('Female: ')
    pprint.pprint(np.mean(female_tot))
    print('\n Female pronouns: ')
    print('By author gender: ')
    pprint.pprint(get_mean(female_gender_sub_v_ob))
    print('By date: ')
    pprint.pprint(get_mean(female_date_sub_v_ob))
    print('By location: ')
    pprint.pprint(get_mean(female_loc_sub_v_ob))
    print('\n Male pronouns: ')
    print('By author gender: ')
    pprint.pprint(get_mean(male_gender_sub_v_ob))
    print('By date:')
    pprint.pprint(get_mean(male_date_sub_v_ob))
    print('By location: ')
    pprint.pprint(get_mean(male_loc_sub_v_ob))

    sub_comp_gender = subject_pronouns_gender_comparison(corpus, 'female')
    sub_comp_gender_list = dict_to_list(sub_comp_gender)

    print('Overall comparative female freq:')
    pprint.pprint(np.mean(sub_comp_gender_list))
    print('By author gender:')
    pprint.pprint(get_mean(freq_by_author_gender(sub_comp_gender)))
    print('By date: ')
    pprint.pprint(get_mean(freq_by_date(sub_comp_gender)))
    print('By location: ')
    pprint.pprint(get_mean(freq_by_location(sub_comp_gender)))


# ??? found in instance_distance_analysis.py
def run_pronoun_freq(corpus):
    """
    Runs a program that uses the instance distance analysis on all novels existing in a given
    corpus, and outputs the data as graphs
    :return:
    """

    all_data = books_pronoun_freq(corpus)

    gender = freq_by_author_gender(all_data)
    box_gender_pronoun_freq(gender, my_pal={"Male Author": "b", "Female Author": "r"},
                            title="she_freq_by_author_gender_sample", x="Author Gender")
    # date = freq_by_date(all_data)
    # box_gender_pronoun_freq(date, my_pal="Greens", title="she_freq_by_date_sample", x="Years")
    # location = freq_by_location(all_data)
    # box_gender_pronoun_freq(location, my_pal="Blues", title="she_freq_by_location_sample",
    #                         x="Location")

    sub_v_ob = subject_vs_object_pronoun_freqs(corpus)

    female_gender_sub_v_ob = get_mean(freq_by_author_gender(sub_v_ob[1]))
    male_gender_sub_v_ob = get_mean(freq_by_author_gender(sub_v_ob[0]))
    bar_sub_obj_freq(female_gender_sub_v_ob, male_gender_sub_v_ob, "obj_sub_by_auth_gender_sample",
                     "Author Gender")
    '''
    female_date_sub_v_ob = get_mean(freq_by_date(sub_v_ob[1]))
    male_date_sub_v_ob = get_mean(freq_by_date(sub_v_ob[0]))
    bar_sub_obj_freq(female_date_sub_v_ob, male_date_sub_v_ob, "obj_sub_by_year_sample",
                     "Years")

    female_loc_sub_v_ob = get_mean(freq_by_location(sub_v_ob[1]))
    male_loc_sub_v_ob = get_mean(freq_by_location(sub_v_ob[0]))
    bar_sub_obj_freq(female_loc_sub_v_ob, male_loc_sub_v_ob, "obk_sub_by_location_sample",
                     "Location")                

    '''
