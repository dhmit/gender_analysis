import gzip
import os
import pickle


from pathlib import Path, PurePosixPath
import codecs

import seaborn as sns

DEBUG = False

BASE_PATH = Path(os.path.abspath(os.path.dirname(__file__)))

METADATA_LIST = ['gutenberg_id', 'author', 'date', 'title', 'country_publication', 'author_gender',
                 'subject', 'notes']

# books from gutenberg downloaded from Dropbox folder shared by Keith
INITIAL_BOOK_STORE = r'corpora/test_books_30'
#TODO: change to actual directory when generating corpus

# plus some extras
AUTHOR_NAME_REGEX = r"(?P<last_name>(\w+ )*\w*)\, (?P<first_name>(\w+\.* )*(\w\.*)*)(?P<suffix>\, \w+\.)*(\((?P<real_name>(\w+ )*\w*)\))*"
outputDir = 'converted'
TEXT_START_MARKERS = frozenset((
    "*END*THE SMALL PRINT",
    "*** START OF THE PROJECT GUTENBERG",
    "*** START OF THIS PROJECT GUTENBERG",
    "This etext was prepared by",
    "E-text prepared by",
    "Produced by",
    "Distributed Proofreading Team",
    "Proofreading Team at http://www.pgdp.net",
    "http://gallica.bnf.fr)",
    "      http://archive.org/details/",
    "http://www.pgdp.net",
    "by The Internet Archive)",
    "by The Internet Archive/Canadian Libraries",
    "by The Internet Archive/American Libraries",
    "public domain material from the Internet Archive",
    "Internet Archive)",
    "Internet Archive/Canadian Libraries",
    "Internet Archive/American Libraries",
    "material from the Google Print project",
    "*END THE SMALL PRINT",
    "***START OF THE PROJECT GUTENBERG",
    "This etext was produced by",
    "*** START OF THE COPYRIGHTED",
    "The Project Gutenberg",
    "http://gutenberg.spiegel.de/ erreichbar.",
    "Project Runeberg publishes",
    "Beginning of this Project Gutenberg",
    "Project Gutenberg Online Distributed",
    "Gutenberg Online Distributed",
    "the Project Gutenberg Online Distributed",
    "Project Gutenberg TEI",
    "This eBook was prepared by",
    "http://gutenberg2000.de erreichbar.",
    "This Etext was prepared by",
    "This Project Gutenberg Etext was prepared by",
    "Gutenberg Distributed Proofreaders",
    "Project Gutenberg Distributed Proofreaders",
    "the Project Gutenberg Online Distributed Proofreading Team",
    "**The Project Gutenberg",
    "*SMALL PRINT!",
    "More information about this book is at the top of this file.",
    "tells you about restrictions in how the file may be used.",
    "l'authorization à les utilizer pour preparer ce texte.",
    "of the etext through OCR.",
    "*****These eBooks Were Prepared By Thousands of Volunteers!*****",
    "We need your donations more than ever!",
    " *** START OF THIS PROJECT GUTENBERG",
    "****     SMALL PRINT!",
    '["Small Print" V.',
    '      (http://www.ibiblio.org/gutenberg/',
    'and the Project Gutenberg Online Distributed Proofreading Team',
    'Mary Meehan, and the Project Gutenberg Online Distributed Proofreading',
    '                this Project Gutenberg edition.',
))
TEXT_END_MARKERS = frozenset((
    "*** END OF THE PROJECT GUTENBERG",
    "*** END OF THIS PROJECT GUTENBERG",
    "***END OF THE PROJECT GUTENBERG",
    "End of the Project Gutenberg",
    "End of The Project Gutenberg",
    "Ende dieses Project Gutenberg",
    "by Project Gutenberg",
    "End of Project Gutenberg",
    "End of this Project Gutenberg",
    "Ende dieses Projekt Gutenberg",
    "        ***END OF THE PROJECT GUTENBERG",
    "*** END OF THE COPYRIGHTED",
    "End of this is COPYRIGHTED",
    "Ende dieses Etextes ",
    "Ende dieses Project Gutenber",
    "Ende diese Project Gutenberg",
    "**This is a COPYRIGHTED Project Gutenberg Etext, Details Above**",
    "Fin de Project Gutenberg",
    "The Project Gutenberg Etext of ",
    "Ce document fut presente en lecture",
    "Ce document fut présenté en lecture",
    "More information about this book is at the top of this file.",
    "We need your donations more than ever!",
    "END OF PROJECT GUTENBERG",
    " End of the Project Gutenberg",
    " *** END OF THIS PROJECT GUTENBERG",
))
LEGALESE_START_MARKERS = frozenset(("<<THIS ELECTRONIC VERSION OF",))
LEGALESE_END_MARKERS = frozenset(("SERVICE THAT CHARGES FOR DOWNLOAD",))

# TODO(elsa): Investigate doctest errors in this file, may be a result of
# my own system, not actual code errors

def load_csv_to_list(file_path):
    """
    Loads a csv file

    :param file_path: can be a string or Path object
    :return: a list of strings
    """
    file_type = PurePosixPath(file_path).suffix

    if isinstance(file_path, str):
        file_path = Path(file_path)

    if file_type != '.csv':
        raise Exception(
            'Cannot load if current file type is not .csv'
        )
    else:
        file = open(file_path, encoding='utf-8')
        result = file.readlines()

    file.close()
    return result

def load_txt_to_string(file_path):
    """
    Loads a txt file

    :param file_path: can be a string or Path object
    :return: the text as a string type
    """
    file_type = PurePosixPath(file_path)

    if isinstance(file_path, str):
        file_path = Path(file_path)

    if file_type != '.txt':
        raise Exception(
            'Cannot load if current file type is not .txt'
        )
    else:
        try:
            file = open(file_path, encoding='utf-8')
            result = file.read()
        except UnicodeDecodeError as err:
            print (f'Unicode file loading error {file_path}.')
            raise err

    file.close()
    return result


def store_pickle(obj, filename):
    """
    Store a compressed "pickle" of the object in the "pickle_data" directory
    and return the full path to it.

    The filename should not contain a directory or suffix.

    Example in lieu of Doctest to avoid writing out a file.

        my_object = {'a': 4, 'b': 5, 'c': [1, 2, 3]}
        gender_analysis.common.store_pickle(my_object, 'example_pickle')

    :param obj: Any Python object to be pickled
    :param filename: str | Path
    :return: Path
    """
    filename = BASE_PATH / 'pickle_data' / (str(filename) + '.pgz')
    with gzip.GzipFile(filename, 'w') as fileout:
        pickle.dump(obj, fileout)
    return filename


def load_pickle(filename):
    """
    Load the pickle stored at filename where filename does not contain a
    directory or suffix.

    Example in lieu of Doctest to avoid writing out a file.

        my_object = gender_analysis.common.load_pickle('example_pickle')
        my_object
        {'a': 4, 'b': 5, 'c': [1, 2, 3]}

    :param filename: str | Path
    :return: object
    """
    raise IOError

    filename = BASE_PATH / 'pickle_data' / (str(filename) + '.pgz')
    with gzip.GzipFile(filename, 'r') as filein:
        obj = pickle.load(filein)
    return obj


def get_text_file_encoding(filepath):
    """
    For text file at filepath returns the text encoding as a string (e.g. 'utf-8')

    >>> from gender_analysis import common
    >>> from pathlib import Path
    >>> import os

    >>> path=Path(common.BASE_PATH,'testing', 'corpora','sample_novels','texts','hawthorne_scarlet.txt')
    >>> common.get_text_file_encoding(path)
    'UTF-8-SIG'

    Note: For files containing only ascii characters, this function will return 'ascii' even if
    the file was encoded with utf-8

    >>> import os
    >>> from pathlib import Path
    >>> from gender_analysis import common
    >>> text = 'here is an ascii text'
    >>> file_path = Path(common.BASE_PATH, 'example_file.txt')
    >>> with codecs.open(file_path, 'w', 'utf-8') as source:
    ...     source.write(text)
    ...     source.close()
    >>> common.get_text_file_encoding(file_path)
    'ascii'
    >>> file_path = Path(common.BASE_PATH, 'example_file.txt')
    >>> os.remove(file_path)

    :param filepath: fstr
    :return: str
    """
    from chardet.universaldetector import UniversalDetector
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
    Note: Currentyl only supports encodings utf-8, ascii and iso-8859-1

    :param source_path: str or Path
    :param target_path: str or Path
    :param target_encoding: str

    >>> from gender_analysis.common import BASE_PATH
    >>> text = ' ¶¶¶¶ here is a test file'
    >>> source_path = Path(BASE_PATH, 'source_file.txt')
    >>> target_path = Path(BASE_PATH, 'target_file.txt')
    >>> with codecs.open(source_path, 'w', 'iso-8859-1') as source:
    ...     source.write(text)
    >>> get_text_file_encoding(source_path)
    'ISO-8859-1'
    >>> convert_text_file_to_new_encoding(source_path, target_path, target_encoding='utf-8')
    >>> get_text_file_encoding(target_path)
    'utf-8'
    >>> import os
    >>> os.remove(source_path)
    >>> os.remove(target_path)

    :return:
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

    with codecs.open(source_path, 'rU', encoding=source_encoding) as source_file:
        text = source_file.read()
    with codecs.open(target_path, 'w', encoding=target_encoding) as target_file:
        target_file.write(text)


def load_graph_settings(show_grid_lines=True):
    '''
    This function sets the seaborn graph settings to the defaults for our project.
    Defaults to displaying gridlines. To remove gridlines, call with False.
    :return:
    '''
    show_grid_lines_string = str(show_grid_lines)
    palette = "colorblind"
    style_name = "white"
    background_color = (252/255,245/255,233/255,0.4)
    style_list = {'axes.edgecolor': '.6', 'grid.color': '.9', 'axes.grid': show_grid_lines_string,
                  'font.family': 'serif', 'axes.facecolor':background_color,
                  'figure.facecolor':background_color}
    sns.set_color_codes(palette)
    sns.set_style(style_name, style_list)

if __name__ == '__main__':
    from dh_testers.testRunner import main_test
    main_test(import_plus_relative=True)  # this allows for relative calls in the import.
