from nltk.tokenize import sent_tokenize, word_tokenize

from corpus_analysis import common
from gender_analysis.common import _get_parser_download_if_not_present


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
