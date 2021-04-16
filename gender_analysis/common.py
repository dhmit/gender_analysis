import os
from pathlib import Path

from nltk.corpus import stopwords

from gender_analysis.pronouns import PronounSeries
from gender_analysis.gender import Gender

SWORDS_ENG = stopwords.words('english')

BASE_PATH = Path(os.path.abspath(os.path.dirname(__file__)))
TEST_DATA_PATH = Path(BASE_PATH, '../corpus_analysis/testing', 'test_data')

# Common Pronoun Collections
HE_SERIES = PronounSeries('Masc', {'he', 'his', 'him', 'himself'}, subj='he', obj='him')
SHE_SERIES = PronounSeries('Fem', {'she', 'her', 'hers', 'herself'}, subj='she', obj='her')
THEY_SERIES = PronounSeries('Andy', {'they', 'them', 'theirs', 'themself'}, subj='they', obj='them')

# Common Gender Collections
MALE = Gender('Male', HE_SERIES)
FEMALE = Gender('Female', SHE_SERIES)
NONBINARY = Gender('Nonbinary', THEY_SERIES)

BINARY_GROUP = [FEMALE, MALE]
TRINARY_GROUP = [FEMALE, MALE, NONBINARY]
