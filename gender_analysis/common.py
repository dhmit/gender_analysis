import gzip
import os
import pickle
import sys
from pathlib import Path

import nltk
from nltk.corpus import stopwords
import seaborn as sns
from chardet.universaldetector import UniversalDetector

from gender_analysis.pronouns import PronounSeries
from gender_analysis.gender import Gender

SWORDS_ENG = stopwords.words('english')

BASE_PATH = Path(os.path.abspath(os.path.dirname(__file__)))
TEST_DATA_PATH = Path(BASE_PATH, 'testing', 'test_data')

# Common Pronoun Collections
HE_SERIES = PronounSeries('Masc', {'he', 'his', 'him', 'himself'}, subj='he', obj='him')
SHE_SERIES = PronounSeries('Fem', {'she', 'her', 'hers', 'herself'}, subj='she', obj='her')
THEY_SERIES = PronounSeries('Andy', {'they', 'them', 'theirs', 'themself'}, subj='they', obj='them')
IT_SERIES = PronounSeries('It', {'it', 'itself'}, subj='it', obj='it')
XE_SERIES = PronounSeries('Xe', {'xe', 'xem', 'xyr', 'xyrs', 'xemself'}, subj='xe', obj='xem')
AE_SERIES = PronounSeries('Ae', {'ae', 'aer', 'aers', 'aerself'}, subj='ae', obj='aer')
FAE_SERIES = PronounSeries('Fae', {'fae', 'faer', 'faers', 'faerself'}, subj='fae', obj='faer')
EY_SERIES = PronounSeries('Ey', {'ey', 'em', 'eir', 'eirs', 'eirself'}, subj='ey', obj='em')
VE_SERIES = PronounSeries('Ve', {'ve', 'ver', 'vis', 'verself'}, subj='ve', obj='ver')
PER_SERIES = PronounSeries('Per', {'per', 'pers', 'perself'}, subj='per', obj='per')
ZE_HIR_SERIES = PronounSeries('Ze', {'ze', 'hir', 'hirs', 'hirself'}, subj='ze', obj='hir')

# Common Gender Collections
MALE = Gender('Male', HE_SERIES)
FEMALE = Gender('Female', SHE_SERIES)
NONBINARY = Gender('Nonbinary', THEY_SERIES)

# Using `Gender` class as a way of tracking neopronouns defined above
NEOGENDERS = Gender('Neo', [
    XE_SERIES,
    AE_SERIES,
    FAE_SERIES,
    EY_SERIES,
    VE_SERIES,
    PER_SERIES,
    ZE_HIR_SERIES,
])

BINARY_GROUP = [FEMALE, MALE]
TRINARY_GROUP = [FEMALE, MALE, NONBINARY]


def load_csv_to_list(file_path):
    """
    Loads a csv file from the given filepath and returns its contents as a list of strings.

    :param file_path: str or Path object
    :return: a list of strings

    >>> from pathlib import Path
    >>> from gender_analysis import common
    >>> from gender_analysis.testing.common import LARGE_TEST_CORPUS_CSV
    >>> corpus_metadata_path = LARGE_TEST_CORPUS_CSV
    >>> corpus_metadata = load_csv_to_list(corpus_metadata_path)
    >>> type(corpus_metadata)
    <class 'list'>

    """
    if isinstance(file_path, str):
        file_path = Path(file_path)

    file_type = file_path.suffix

    if file_type != '.csv':
        raise Exception('Cannot load if current file type is not .csv')

    with open(file_path, encoding='utf-8') as csv_file:
        result = csv_file.readlines()

    return result


def load_txt_to_string(file_path):
    """
    Loads a txt file and returns a str representation of it.

    :param file_path: str or Path object
    :return: The file's text as a string

    >>> from pathlib import Path
    >>> from gender_analysis import common
    >>> novel_path = Path(common.TEST_DATA_PATH, 'sample_novels', 'texts', 'austen_persuasion.txt')
    >>> novel_text = load_txt_to_string(novel_path)
    >>> type(novel_text), len(novel_text)
    (<class 'str'>, 466887)

    """
    if isinstance(file_path, str):
        file_path = Path(file_path)

    file_type = file_path.suffix

    if file_type != '.txt':
        raise Exception('Cannot load if current file type is not .txt')

    with open(file_path, encoding='utf-8') as txt_file:
        result = txt_file.read()

    return result


def store_pickle(obj, filepath):
    """
    Store a compressed "pickle" of the object in the "pickle_data" directory
    and return the full path to it.

    :param obj: Any Python object that can be pickled
    :param filepath: str or  Path object
    :return: Path object

    Example in lieu of Doctest to avoid writing out a file.

        my_object = {'a': 4, 'b': 5, 'c': [1, 2, 3]}
        gender_analysis.common.store_pickle(my_object, 'path_to_pickle/example_pickle.pgz')

    """
    if isinstance(filepath, str):
        filepath = Path(filepath)
    if not isinstance(filepath, Path):
        raise ValueError(f'filepath must be a str or path object, not type {type(filepath)}')

    if filepath.stem == '':
        filepath = Path(str(filepath) + '.pgz')

    with gzip.GzipFile(filepath, 'w') as fileout:
        pickle.dump(obj, fileout)
    return filepath


def load_pickle(filepath):
    """
    Loads the pickle stored at a given filepath, and returns the Python object that was stored.

    :param filepath: str or Path object
    :return: Previously-pickled object

    >>> from gender_analysis import common
    >>> from pathlib import Path
    >>> pickle_filepath = Path(common.BASE_PATH, 'testing', 'test_data',\
                               'test_pickle.pgz')
    >>> loaded_object = common.load_pickle(pickle_filepath)
    >>> loaded_object
    {'a': 4, 'b': 5, 'c': [1, 2, 3]}

    """
    if filepath is None:
        raise IOError('No path supplied')

    if isinstance(filepath, str):
        filepath = Path(filepath)
    if not isinstance(filepath, Path):
        raise ValueError(f'filepath must be an str or Path object, not type {type(filepath)}')

    if filepath.stem == '':
        filepath = Path(str(filepath) + '.pgz')

    with gzip.GzipFile(filepath, 'r') as filein:
        obj = pickle.load(filein)
    return obj


def get_text_file_encoding(filepath):
    """
    Returns the text encoding as a string for a txt file at the given filepath.

    :param filepath: str or Path object
    :return: Name of encoding scheme as a string

    >>> from gender_analysis import common
    >>> from pathlib import Path
    >>> import os
    >>> path=Path(common.TEST_DATA_PATH, 'sample_novels', 'texts', 'hawthorne_scarlet.txt')
    >>> common.get_text_file_encoding(path)
    'ascii'

    Note: For files containing only ascii characters, this function will return 'ascii' even if
    the file was encoded with utf-8

    >>> import os
    >>> from pathlib import Path
    >>> from gender_analysis import common
    >>> text = 'here is an ascii text'
    >>> file_path = Path(common.BASE_PATH, 'example_file.txt')
    >>> with open(file_path, 'w', encoding='utf-8') as source:
    ...     _ = source.write(text)
    ...     source.close()
    >>> common.get_text_file_encoding(file_path)
    'ascii'
    >>> file_path = Path(common.BASE_PATH, 'example_file.txt')
    >>> os.remove(file_path)

    """
    detector = UniversalDetector()

    with open(filepath, 'rb') as file:
        for line in file:
            detector.feed(line)
            if detector.done:
                break
        detector.close()
    return detector.result['encoding']


def convert_text_file_to_new_encoding(source_path, target_path, target_encoding):
    """
    Converts a text file in source_path to the specified encoding in target_encoding

    Note: Currently only supports encodings utf-8, ascii and iso-8859-1

    :param source_path: str or Path
    :param target_path: str or Path
    :param target_encoding: str
    :return: None

    >>> from gender_analysis.common import BASE_PATH
    >>> text = ' ¶¶¶¶ here is a test file'
    >>> source_path = Path(BASE_PATH, 'source_file.txt')
    >>> target_path = Path(BASE_PATH, 'target_file.txt')
    >>> with open(source_path, 'w', encoding='iso-8859-1') as source:
    ...     _ = source.write(text)
    >>> get_text_file_encoding(source_path)
    'ISO-8859-1'
    >>> convert_text_file_to_new_encoding(source_path, target_path, target_encoding='utf-8')
    >>> get_text_file_encoding(target_path)
    'utf-8'
    >>> import os
    >>> os.remove(source_path)
    >>> os.remove(target_path)
    """

    valid_encodings = ['utf-8', 'utf8', 'UTF-8-SIG', 'ascii', 'iso-8859-1', 'ISO-8859-1',
                       'Windows-1252']

    # if the source_path or target_path is a string, turn to Path object.
    if isinstance(source_path, str):
        source_path = Path(source_path)
    if isinstance(target_path, str):
        target_path = Path(target_path)

    # check if source and target encodings are valid
    source_encoding = get_text_file_encoding(source_path)
    if source_encoding not in valid_encodings:
        raise ValueError('convert_text_file_to_new_encoding() only supports the following source '
                         f'encodings: {valid_encodings} but not {source_encoding}.')
    if target_encoding not in valid_encodings:
        raise ValueError('convert_text_file_to_new_encoding() only supports the following target '
                         f'encodings: {valid_encodings} but not {target_encoding}.')

    # print warning if filenames don't end in .txt
    if not source_path.parts[-1].endswith('.txt') or not target_path.parts[-1].endswith('.txt'):
        print(f"WARNING: Changing encoding to {target_encoding} on a file that does not end with "
              f".txt. Source: {source_path}. Target: {target_path}")

    with open(source_path, 'r', encoding=source_encoding) as source_file:
        text = source_file.read()
    with open(target_path, 'w', encoding=target_encoding) as target_file:
        _ = target_file.write(text)


def load_graph_settings(show_grid_lines=True):
    """
    Sets the seaborn graph settings to the defaults for the project.
    Defaults to displaying gridlines. To remove gridlines, call with False.

    :param show_grid_lines: Boolean; Determines whether to show gridlines in graphs.
    :return: None

    """
    show_grid_lines_string = str(show_grid_lines)
    palette = "colorblind"
    style_name = "white"
    background_color = (252 / 255, 245 / 255, 233 / 255, 0.4)
    style_list = {
        'axes.edgecolor': '.6',
        'grid.color': '.9',
        'axes.grid': show_grid_lines_string,
        'font.family': 'serif',
        'axes.facecolor': background_color,
        'figure.facecolor': background_color,
    }
    sns.set_color_codes(palette)
    sns.set_style(style_name, style_list)


def download_nltk_package_if_not_present(package_name):
    """
    Checks to see whether the user already has a given nltk package,
    and if not, prompts the user whether to download it.

    We download all necessary packages at install time, but this is just
    in case the user has deleted them.

    :param package_name: name of the nltk package
    :return:
    """

    # nltk.data.find uses the folder + name, but download uses just the name...
    package_download_name = package_name.split('/')[1]

    try:
        nltk.data.find(package_name)
    except LookupError:
        user_key = input(
            f'This function requires the NLTK package {package_name}, '
            + 'which you do not have installed.\n'
            + 'Press ENTER to download and install this package, '
            + 'or n then enter to cancel and exit.\n'
        )
        if user_key.strip() == 'n':
            sys.exit()

        nltk.download(package_download_name)
        print('\n')


class MissingMetadataError(Exception):
    """
    Raised when a function that assumes certain metadata is called on a corpus without that
    metadata
    """
    def __init__(self, metadata_fields, message=''):
        super().__init__()
        self.metadata_fields = metadata_fields
        self.message = message

    def __str__(self):
        metadata_string = ''

        for i, field in enumerate(self.metadata_fields):
            metadata_string += field
            if i != len(self.metadata_fields) - 1:
                metadata_string += ', '

        is_plural = len(self.metadata_fields) > 1

        return (
            'This Corpus is missing the following metadata field' + ('s' if is_plural else '')
            + ':\n' + '    ' + metadata_string + '\n'
            + self.message + ('\n' if self.message else '')
            + 'In order to run this function, you must create a new metadata csv\n'
            + 'with ' + ('these ' if is_plural else 'this ')
            + 'field' + ('s ' if is_plural else ' ')
            + 'and run Corpus.update_metadata().'
        )


def create_path_object_and_directories(output_dir, filename=None):
    """
    Creates a path object for the file with the absolute output_dir and with the
    given filename (if provided). It will create the path to the output_dir if it is non-existent
    """
    try:
        path = output_dir

        if isinstance(output_dir, str):
            path = Path(output_dir)

        if not os.path.isdir(output_dir):
            path.mkdir(parents=True)

        if filename:
            path = path / filename

        return path

    except OSError:
        print("Could not create the correct file and path")
        return None


if __name__ == '__main__':
    from dh_testers.testRunner import main_test

    main_test(import_plus_relative=True)  # this allows for relative calls in the import.
