from pathlib import Path
import csv
import os
import urllib

from clint import textui
import requests
from nltk.parse.stanford import StanfordDependencyParser
from nltk.tokenize import sent_tokenize, word_tokenize

from gender_analysis import common
from gender_analysis.corpus import Corpus
from gender_analysis.document import Document


def get_parser_download_if_not_present():
    """
    The jar files are too big to commit directly, so download them
    :param path_to_jar: local path to stanford-parser.jar
    :param path_to_models_jar: local path to stanford-parser-3.9.1-models.jar
    >>> parser = get_parser("assets/stanford-parser.jar","assets/stanford-parser-3.9.1-models.jar")
    >>> parser == None
    False
    """

    parser_dir = common.BASE_PATH / 'stanford_parser'
    if not os.path.exists(parser_dir):
        os.mkdir(parser_dir)

    parser_filename = 'stanford-parser.jar'
    models_filename = 'stanford-parser-3.9.1-models.jar'
    path_to_jar = parser_dir / parser_filename
    path_to_models_jar = parser_dir / models_filename

    if (not os.path.isfile(path_to_jar) or
        not os.path.isfile(path_to_models_jar)):

        user_key = input(f'This function requires us to download the Stanford Dependency Parser.\n'
                         + 'This is a 612 MB download, which may take 10-20 minutes to download on an average 10 MBit/s connection.\n'
                         + 'Press y then enter to download and install this package, or n then enter to cancel and exit.\n')

        while user_key.strip() not in ['y', 'n']:
            user_key = input(f'Press y then enter to download and install this package, or n then enter to cancel and exit.\n')

        if user_key == 'n':
            print('Exiting.')
            exit()
        elif user_key == 'y':
            print('Downloading...')
            parser_url = 'https://nlp.stanford.edu/software/stanford-parser-full-2018-10-17.zip'
            zip_path  = parser_dir / 'parser.zip'
            r = requests.get(parser_url, stream=True)

            # doing this chunk by chunk so we can make a progress bar
            with open(zip_path, 'wb') as f:
                total_length = int(r.headers.get('content-length'))
                for chunk in textui.progress.bar(r.iter_content(chunk_size=1024),
                                                 expected_size=(total_length/1024) + 1):
                    if chunk:
                        f.write(chunk)
                        f.flush()

    parser = StanfordDependencyParser(path_to_jar, path_to_models_jar)
    return parser


def pickle(document, parser, pickle_filepath='dep_tree.pgz'):
    """
    This function returns a pickled tree
    :param document: Document we are interested in
    :param parser: Stanford parser object
    :param pickle_filepath: filepath to store pickled dependency tree
    :return: tree in pickle format
    """

    try:
        tree = common.load_pickle(pickle_filepath)
    except (IOError, FileNotFoundError):
        sentences = sent_tokenize(document.text.lower().replace("\n", " "))
        he_she_sentences = []
        for sentence in sentences:
            add_sentence = False
            words = [word for word in word_tokenize(sentence)]
            for word in words:
                if word == "he" or word == "she" or word == "him" or word == "her":
                    add_sentence = True
            if add_sentence:
                he_she_sentences.append(sentence)
        sentences = he_she_sentences
        result = parser.raw_parse_sents(sentences)
        # dependency triples of the form ((head word, head tag), rel, (dep word, dep tag))
        # link defining dependencies: https://nlp.stanford.edu/software/dependencies_manual.pdf
        tree = list(result)
        tree_list = []
        i = 0
        for sentence in tree:
            tree_list.append([])
            triples = list(next(sentence).triples())
            for triple in triples:
                tree_list[i].append(triple)
            i += 1
        tree = tree_list
        common.store_pickle(tree, pickle_filepath)
    return tree


def parse_document(document, parser):
    """
    This function parses all sentences in the document
    :param document: document object we want to analyze
    :param parser: Stanford dependency parser
    :return: list containing the following:
    - Title of document
    - Count of male pronoun subject occurrences
    - Count of male pronoun object occurrences
    - Count of female pronoun subject occurrences
    - Count of female pronoun object occurrences
    - List of adjectives describing male pronouns as one space-separated string
    - List of verbs associated with male pronouns as one space-separated string
    - List of adjectives describing female pronouns as one space-separated string
    - List of verbs associated with female pronouns as one space-separated string

    >>> parser = get_parser("assets/stanford-parser.jar","assets/stanford-parser-3.9.1-models.jar")
    >>> documents = Corpus('sample_novels').documents
    >>> document_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter', 'date': '1900',
    ...                   'filename': None, 'text': "He told her"}
    >>> toy_novel = Document(document_metadata)
    >>> parse_novel(toy_novel, parser)
    ('Scarlet Letter', 1, 0, 0, 1, [], ['told'], [], [])

    """
    parser = get_parser_download_if_not_present()
    tree = pickle(document, parser)

    male_subj_count = male_obj_count = female_subj_count = female_obj_count = 0
    female_adjectives = []
    male_adjectives = []
    female_verbs = []
    male_verbs = []

    for sentence in tree:
        for triple in sentence:
            if triple[1] == "nsubj" and triple[2][0] == "he":
                male_subj_count += 1
            if triple[1] == "dobj" and triple[2][0] == "him":
                male_obj_count += 1
            if triple[1] == "nsubj" and triple[2][0] == "she":
                female_subj_count += 1
            if triple[1] == "dobj" and triple[2][0] == "her":
                female_obj_count += 1
            if triple[1] == "nsubj" and triple[0][1] == "JJ":
                if triple[2][0] == "she":
                    female_adjectives.append(triple[0][0])
                elif triple[2][0] == "he":
                    male_adjectives.append(triple[0][0])
            if triple[1] == "nsubj" and (triple[0][1] == "VBD" or triple[0][1] == "VB" or
                                         triple[0][1] == "VBP" or triple[0][1] == "VBZ"):
                if triple[2][0] == "she":
                    female_verbs.append(triple[0][0])
                elif triple[2][0] == "he":
                    male_verbs.append(triple[0][0])

    return [document.title, male_subj_count, male_obj_count, female_subj_count, female_obj_count,
            " ".join(male_adjectives), " ".join(male_verbs), " ".join(female_adjectives),
            " ".join(female_verbs)]


# def test_analysis():
#     """
#     This function contains all analysis code to be run (previously in main function)
#     - First generates a Stanford NLP parser
#     - Iterates over sample documents corpus and parses each document (performs analysis: gender pronoun
#     count, list of adjectives, list of verbs)
#     - Writes output to dependency_analysis_results.csv
#     """
#
#     parser = get_parser("assets/stanford-parser.jar", "assets/stanford-parser-3.9.1-models.jar")
#     documents = Corpus('sample_novels').documents
#     for document in documents:
#         try:
#             row = parse_document(document, parser)
#             print(row)
#             with open('dependency_analysis_results.csv', mode='w') as results_file:
#                 writer = csv.writer(results_file, delimiter=',', quotechar='"',
#                                              quoting=csv.QUOTE_MINIMAL)
#                 writer.writerow(row)
#         except OSError:
#             continue
