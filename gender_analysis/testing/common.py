from pathlib import Path
from gender_analysis.common import BASE_PATH

TEST_DATA_DIR = Path(BASE_PATH, 'testing', 'test_data')

# A small corpus with only 10 documents
SMALL_TEST_CORPUS_PATH = Path(TEST_DATA_DIR, 'test_corpus')
SMALL_TEST_CORPUS_CSV = Path(TEST_DATA_DIR, 'test_corpus', 'test_corpus.csv')

# A larger corpus with 99 documents
LARGE_TEST_CORPUS_PATH = Path(TEST_DATA_DIR, 'sample_novels', 'texts')
LARGE_TEST_CORPUS_CSV = Path(TEST_DATA_DIR, 'sample_novels', 'sample_novels.csv')

# A corpus that is comprised of Reddit posts and comments
REDDIT_CORPUS_PATH = Path(TEST_DATA_DIR, 'r_starwars_data', 'posts')
REDDIT_CORPUS_CSV = Path(TEST_DATA_DIR, 'r_starwars_data', 'metadata.csv')

# A directory to a collection of test documents
DOCUMENT_TEST_PATH = Path(TEST_DATA_DIR, 'document_test_files')
DOCUMENT_TEST_CSV = Path(TEST_DATA_DIR, 'document_test_files', 'document_test_files.csv')
