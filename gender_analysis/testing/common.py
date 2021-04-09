from pathlib import Path
from gender_analysis.common import BASE_PATH

TEST_DATA_DIR = Path(BASE_PATH, 'testing', 'test_data')

# All tests using one of the novel corpora points here.
TEST_CORPUS_PATH = Path(TEST_DATA_DIR, 'sample_novels', 'texts')

# A tiny corpus with only 4 documents
CONTROLLED_TEST_CORPUS_CSV = Path(TEST_DATA_DIR, 'sample_novels', 'controlled_test_corpus.csv')

# A tiny corpus with only 4 documents
TINY_TEST_CORPUS_CSV = Path(TEST_DATA_DIR, 'sample_novels', 'tiny_test_corpus.csv')

# A small corpus with only 10 documents
SMALL_TEST_CORPUS_CSV = Path(TEST_DATA_DIR, 'sample_novels', 'small_test_corpus.csv')

# A larger corpus with 99 documents
LARGE_TEST_CORPUS_CSV = Path(TEST_DATA_DIR, 'sample_novels', 'large_test_corpus.csv')

# A corpus that is comprised of Reddit posts and comments
REDDIT_CORPUS_PATH = Path(TEST_DATA_DIR, 'r_starwars_data', 'posts')
REDDIT_CORPUS_CSV = Path(TEST_DATA_DIR, 'r_starwars_data', 'metadata.csv')

# A directory to a collection of test documents
DOCUMENT_TEST_PATH = Path(TEST_DATA_DIR, 'document_test_files')
DOCUMENT_TEST_CSV = Path(TEST_DATA_DIR, 'document_test_files', 'document_test_files.csv')
