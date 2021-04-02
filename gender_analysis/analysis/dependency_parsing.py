import os
import sys
import zipfile

from clint import textui
import requests
from nltk.parse.stanford import StanfordDependencyParser
from nltk.tokenize import sent_tokenize, word_tokenize

from gender_analysis import common


def _get_parser_download_if_not_present():
    # pylint: disable=too-many-locals
    """
    Initializes and returns the NLTK wrapper for the Stanford Dependency Parser.

    Prompts the user to download the jar files for the parser if they're not already
    downloaded.

    """

    parser_dir = common.BASE_PATH / 'stanford_parser'
    if not os.path.exists(parser_dir):
        os.mkdir(parser_dir)

    parser_filename = 'stanford-parser.jar'
    models_filename = 'stanford-parser-3.9.2-models.jar'
    path_to_jar = parser_dir / parser_filename
    path_to_models_jar = parser_dir / models_filename

    if not os.path.isfile(path_to_jar) or not os.path.isfile(path_to_models_jar):
        # The required jar files don't exist,
        # so we prompt the user

        user_key = input('This function requires us to download the Stanford Dependency Parser.\n'
                         + 'This is a 612 MB download, which may take 10-20 minutes to download on'
                         + 'an average 10 MBit/s connection.\n'
                         + 'This only happens the first time you run this function.\n'
                         + 'Press y then enter to download and install this package,'
                         + 'or n then enter to cancel and exit.\n')

        while user_key.strip() not in ['y', 'n']:
            user_key = input('Press y then enter to download and install this package,'
                             + 'or n then enter to cancel and exit.\n')

        if user_key == 'n':
            print('Exiting.')
            sys.exit()

        elif user_key == 'y':
            # Download the Jar files
            print('Downloading... (Press CTRL+C to cancel at any time)')
            parser_url = 'https://nlp.stanford.edu/software/stanford-parser-full-2018-10-17.zip'
            zip_path = parser_dir / 'parser.zip'

            req = requests.get(parser_url, stream=True)

            # doing this chunk by chunk so we can make a progress bar
            with open(zip_path, 'wb') as file:
                total_length = int(req.headers.get('content-length'))
                for chunk in textui.progress.bar(req.iter_content(chunk_size=1024),
                                                 expected_size=(total_length / 1024) + 1):
                    if chunk:
                        file.write(chunk)
                        file.flush()

            print('Unpacking files...')

            # unzip and move things to the right place
            zip_base_dir = 'stanford-parser-full-2018-10-17'
            parser_zip_path = zip_base_dir + '/' + parser_filename
            models_zip_path = zip_base_dir + '/' + models_filename

            with zipfile.ZipFile(zip_path) as zipped:
                zipped.extract(parser_zip_path, parser_dir)
                zipped.extract(models_zip_path, parser_dir)

            jar_unzipped_path = parser_dir / zip_base_dir / parser_filename
            models_jar_unzipped_path = parser_dir / zip_base_dir / models_filename
            os.rename(jar_unzipped_path, path_to_jar)
            os.rename(models_jar_unzipped_path, path_to_models_jar)

            # tidy up
            os.rmdir(parser_dir / zip_base_dir)
            os.remove(zip_path)
            print('Done!')

    parser = StanfordDependencyParser(str(path_to_jar), str(path_to_models_jar))
    return parser


def generate_dependency_tree(document, genders=None, pickle_filepath=None):
    # pylint: disable=too-many-locals
    """
    This function returns the dependency tree for a given document. This can optionally be reduced
    such that it will only analyze sentences that involve specified genders' subject/object
    pronouns.

    :param document: Document we are interested in
    :param genders: a collection of genders that will be used to filter out sentences that do not \
        involve the provided genders. If set to `None`, all sentences are parsed (default).
    :param pickle_filepath: filepath to store pickled dependency tree, will not write a file if None
    :return: dependency tree, represented as a nested list

    """

    parser = _get_parser_download_if_not_present()
    sentences = sent_tokenize(document.text.lower().replace("\n", " "))

    # filter out sentences that are not relevant
    if genders is not None:
        filtered_sentences = list()

        # Find all of the words to filter around
        pronoun_filter = set()
        for gender in genders:
            pronoun_filter = pronoun_filter | gender.obj | gender.subj

        for sentence in sentences:
            add_sentence = True

            words = list(word_tokenize(sentence))
            for word in words:
                if word in pronoun_filter:
                    add_sentence = True
            if add_sentence:
                filtered_sentences.append(sentence)
        sentences = filtered_sentences

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

    if pickle_filepath is not None:
        common.store_pickle(tree, pickle_filepath)

    return tree


def get_pronoun_usages(tree, gender):
    """
    Returns a dictionary relating the occurrences of a given gender's pronouns as the
    subject and object of a sentence.

    :param tree: dependency tree for a document, output of **generate_dependency_tree**
    :param gender: `Gender` to check
    :return: Dictionary counting the times male pronouns are used as the subject and object,
        formatted as {'subject': <int>, 'object': <int>}

    """

    obj_count = 0
    subj_count = 0

    for sentence in tree:
        for triple in sentence:
            if triple[1] == "nsubj" and triple[2][0] in gender.subj:
                subj_count += 1
            if triple[1] == "dobj" and triple[2][0] in gender.obj:
                obj_count += 1

    return {'subject': subj_count, 'object': obj_count}


def get_descriptive_adjectives(tree, gender):
    """
    Returns a list of adjectives
    describing pronouns for the given gender in the given dependency tree.

    :param tree: dependency tree for a document, output of **generate_dependency_tree**
    :param gender: `Gender` to search for usages of
    :return: List of adjectives as strings

    """

    adjectives = []

    for sentence in tree:
        for triple in sentence:
            if triple[1] == "nsubj" and triple[0][1] == "JJ":
                if triple[2][0] in gender.identifiers:
                    adjectives.append(triple[0][0])

    return adjectives


def get_descriptive_verbs(tree, gender):
    """
    Returns a list of verbs describing pronouns of the given gender in the given dependency tree.

    :param tree: dependency tree for a document, output of **generate_dependency_tree**
    :param gender: `Gender` to search for usages of
    :return: List of verbs as strings

    """

    verbs = []

    for sentence in tree:
        for triple in sentence:
            if triple[1] == "nsubj" and (triple[0][1] == "VBD" or triple[0][1] == "VB"
                                         or triple[0][1] == "VBP" or triple[0][1] == "VBZ"):
                if triple[2][0] in gender.identifiers:
                    verbs.append(triple[0][0])

    return verbs
