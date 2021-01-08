from more_itertools import windowed
import nltk

from gender_analysis.common import ADJ_TAGS, ADV_TAGS, PROPER_NOUN_TAGS, VERB_TAGS
from gender_analysis import common
pos_dict = {'adj': ADJ_TAGS, 'adv': ADV_TAGS,
            'proper_noun': PROPER_NOUN_TAGS, "verb": VERB_TAGS}

# I've migrated functions from gender_pos to here and modify them for character, but maybe should
# add new functions in gender_pos instead

# first pass on character adj based on gender pos
def find_char_pos(pos_to_find, char_to_find, word_window=5):
    # pylint: disable=too-many-locals
    """
    Takes in a document, a valid part-of-speech, and a Gender to look for,
    and returns a dictionary of words that appear within a window of 5 words around each identifier.

    :param document: Document
    :param pos_to_find: A valid part of speech tag from pos_dict: ['adj','adv','proper_noun','verb']
    :param word_window: number of words to search for in either direction of a gender instance
    :param char_to_find: Character
    :return: dict of words that appear around pronouns mapped to the number of occurrences

    >>> from gender_analysis import document
    >>> from pathlib import Path
    >>> from gender_analysis import common
    >>> document_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter', 'date': \
    '1966', 'filename': 'test_text_7.txt', 'filepath': Path(common.TEST_DATA_PATH, \
    'document_test_files', 'test_text_7.txt')}
    >>> scarlett = document.Document(document_metadata)
    >>> find_gender_pos(scarlett, 'adj', common.MALE, genders_to_exclude=[common.FEMALE])
    {'handsome': 3, 'sad': 1}

    """
    output = {}
    text = char_to_find.document.get_tokenized_text()

    if pos_to_find in pos_dict.keys():
        pos_tags = pos_dict[pos_to_find]
    else:
        return "Invalid part of speech"
    # rewrite based on character class
    identifiers_to_find = [char_to_find.name] + [char_to_find.nicknames] # identifiers: names and nicknames

    for words in windowed(text, 2 * word_window + 1):
        if not words[word_window].lower() in identifiers_to_find:
            continue
        if bool(set(words)):
            continue

        words = list(words)
        for index, word in enumerate(words):
            words[index] = word.lower()

        tags = nltk.pos_tag(words)
        for tag_index, _ in enumerate(tags):
            if tags[tag_index][1] in pos_tags:
                word = words[tag_index]
                if word in output.keys():
                    output[word] += 1
                else:
                    output[word] = 1

    return output


def find_char_adj(char_to_find, word_window=5):
    # pylint: disable=too-many-locals
    """
    Takes in a document and a Character to look for, and returns a dictionary of adjectives that
    appear within a window of 5 words around each identifier
    :param char_to_find: Character
    :param word_window: number of words to search for in either direction of a gender instance
    :return: dict of adjectives that appear around pronouns mapped to the number of occurrences

    >>> from gender_analysis import document
    >>> from pathlib import Path
    >>> from gender_analysis import common
    >>> document_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter', 'date': \
    '1966', 'filename': 'test_text_7.txt', 'filepath': Path(common.TEST_DATA_PATH, \
    'document_test_files', 'test_text_7.txt')}
    >>> scarlett = document.Document(document_metadata)
    >>> find_gender_adj(scarlett, common.MALE, genders_to_exclude=[common.FEMALE])
    {'handsome': 3, 'sad': 1}

    """

    return find_char_pos('adj', char_to_find, word_window)
