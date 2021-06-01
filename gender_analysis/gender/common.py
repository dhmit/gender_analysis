import os
import sys
import zipfile

from clint import textui
import requests
from nltk.parse.stanford import StanfordDependencyParser
from nltk.corpus import stopwords

from gender_analysis.common import BASE_PATH

from gender_analysis.gender.pronouns import PronounSeries
from gender_analysis.gender.gender import Gender

SWORDS_ENG = stopwords.words('english')

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


def _get_parser_download_if_not_present():
    # pylint: disable=too-many-locals
    """
    Initializes and returns the NLTK wrapper for the Stanford Dependency Parser.

    Prompts the user to download the jar files for the parser if they're not already
    downloaded.

    """

    parser_dir = BASE_PATH / 'stanford_parser'
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
