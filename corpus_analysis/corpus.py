import csv
import random
import os
from collections import Counter
from copy import deepcopy
from pathlib import Path

from nltk import tokenize as nltk_tokenize

from corpus_analysis import common
from corpus_analysis.common import MissingMetadataError
from corpus_analysis.common import load_csv_to_list
from corpus_analysis.document import Document


class Corpus:

    """The corpus class is used to load the metadata and full
    texts of all documents in a corpus

    Once loaded, each corpus contains a list of Document objects

    :param path_to_files: Must be either the path to a directory of txt files
                          or an already-pickled corpus
    :param name: Optional name of the corpus, for ease of use and readability
    :param csv_path: Optional path to a csv metadata file
    :param pickle_on_load: Filepath to save a pickled copy of the corpus

    >>> from corpus_analysis.corpus import Corpus
    >>> from gender_analysis.common import TEST_DATA_PATH
    >>> path = TEST_DATA_PATH / 'sample_novels' / 'texts'
    >>> c = Corpus(path)
    >>> type(c.documents), len(c)
    (<class 'list'>, 99)

    """

    def __init__(
        self,
        path_to_files,
        name=None,
        csv_path=None,
        pickle_on_load=None,
        ignore_warnings=False
    ):
        if isinstance(path_to_files, str):
            path_to_files = Path(path_to_files)

        if not isinstance(path_to_files, Path):
            raise ValueError(
                f'path_to_files must be a str or Path object, not type {type(path_to_files)}'
            )

        self.name = name
        self.documents, self.metadata_fields =\
            self._load_documents_and_metadata(path_to_files,
                                              csv_path,
                                              ignore_warnings=ignore_warnings)

        if pickle_on_load is not None:
            common.store_pickle(self, pickle_on_load)

    def _load_documents_and_metadata(self, path_to_files, csv_path, ignore_warnings=False):
        """
        Loads documents into the corpus with metadata from a csv file given at initialization.
        """
        # pylint: disable=too-many-locals

        # load pickle if provided
        if path_to_files.suffix == '.pgz':
            pickle_data = common.load_pickle(path_to_files)
            return pickle_data.documents, pickle_data.metadata_fields

        # load documents without metadata csv
        elif path_to_files.suffix == '' and not csv_path:
            files = os.listdir(path_to_files)
            metadata_fields = ['filename', 'filepath']
            ignored = []
            documents = []
            for filename in files:
                if filename.endswith('.txt'):
                    metadata_dict = {'filename': filename, 'filepath': path_to_files / filename}
                    documents.append(Document(metadata_dict))
                elif filename.endswith('.csv'):
                    continue  # let's ignore csv files, they're probably metadata
                else:
                    ignored.append(filename)

            if len(documents) == 0:  # path led to directory with no .txt files
                raise ValueError(
                    'path_to_files must lead to a previously pickled corpus '
                    'or directory of .txt files'
                )

            if ignored:
                print(
                    'WARNING: '
                    + 'the following files were not loaded because they are not .txt files.\n'
                    + str(ignored)
                    + '\n'
                    + 'If you would like to analyze the text in these files, '
                    + 'convert these files to .txt and create a new Corpus.'
                )

            return documents, metadata_fields

        # load documents based on the metadata csv
        elif csv_path and path_to_files.suffix == '':
            documents = []
            metadata = set()

            try:
                csv_list = load_csv_to_list(csv_path)
            except FileNotFoundError as err:
                raise FileNotFoundError(
                    'Could not find the metadata csv file for the '
                    + f"'{self.name}' corpus in the expected location "
                    + f'({csv_path}).'
                ) from err
            csv_reader = csv.DictReader(csv_list)

            loaded_document_filenames = []
            for document_metadata in csv_reader:
                filename = document_metadata['filename']
                document_metadata['name'] = self.name
                document_metadata['filepath'] = path_to_files / filename
                this_document = Document(document_metadata)
                documents.append(this_document)
                loaded_document_filenames.append(filename)
                metadata.update(list(document_metadata))

            all_txt_files = [f for f in os.listdir(path_to_files) if f.endswith('.txt')]
            num_loaded = len(documents)
            num_txt_files = len(all_txt_files)
            if not ignore_warnings and num_loaded != num_txt_files:
                # some txt files aren't in the metadata, so issue a warning
                # we don't need to handle the inverse case, because that
                # will have broken the document init above
                print(
                    'WARNING: The following .txt files were not loaded because they '
                    + 'are not your metadata csv:\n'
                    + str(list(set(all_txt_files) - set(loaded_document_filenames)))
                    + '\nYou may want to check that your metadata matches your files '
                    + 'to avoid incorrect results.'
                )

            return sorted(documents), list(metadata)

        else:
            raise ValueError(
                'path_to_files must lead to a previously pickled corpus or directory of .txt files'
            )

    def __len__(self):
        """
        For convenience: returns the number of documents in
        the corpus.

        >>> from corpus_analysis.corpus import Corpus
        >>> from corpus_analysis.testing.common import TEST_CORPUS_PATH
        >>> path = TEST_CORPUS_PATH
        >>> c = Corpus(path)
        >>> len(c)
        99

        :return: number of documents in the corpus as an int
        """
        return len(self.documents)

    def __iter__(self):
        """
        Yield each of the documents from the .documents list.

        For convenience.

        >>> from corpus_analysis.corpus import Corpus
        >>> from corpus_analysis.testing.common import TEST_CORPUS_PATH
        >>> path = TEST_CORPUS_PATH
        >>> c = Corpus(path)
        >>> docs = []
        >>> for doc in c:
        ...    docs.append(doc)
        >>> len(docs)
        99

        """
        for this_document in self.documents:
            yield this_document

    def __eq__(self, other):
        """
        Returns true if both corpora contain the same documents
        Note: ignores differences in the corpus name as that attribute is not used apart from
        initializing a corpus.
        Presumes the documents to be sorted. (They get sorted by the initializer)

        >>> from corpus_analysis.corpus import Corpus
        >>> from gender_analysis.common import TEST_DATA_PATH
        >>> path = TEST_DATA_PATH / 'sample_novels' / 'texts'
        >>> sample_corpus = Corpus(path)
        >>> sorted_docs = sorted(sample_corpus.documents[:20])
        >>> sample_corpus.documents = sorted_docs
        >>> corp1 = sample_corpus.clone()
        >>> corp1.documents = corp1.documents[:10]
        >>> corp2 = sample_corpus.clone()
        >>> corp2.documents = corp2.documents[10:]
        >>> sample_corpus == corp1 + corp2
        True
        >>> sample_corpus == Corpus(path) + corp1
        False

        :return: bool
        """
        if not isinstance(other, Corpus):
            raise NotImplementedError("Only a Corpus can be compared to another Corpus.")

        if len(self) != len(other):
            return False

        for i in range(len(self)):
            if self.documents[i] != other.documents[i]:
                return False

        return True

    def __add__(self, other):
        """
        Adds two corpora together and returns a copy of the result
        Note: retains the name of the first corpus

        >>> from corpus_analysis.corpus import Corpus
        >>> from corpus_analysis.testing.common import TEST_CORPUS_PATH
        >>> path = TEST_CORPUS_PATH
        >>> sample_corpus = Corpus(path)
        >>> sorted_docs = sorted(sample_corpus.documents[:20])
        >>> sample_corpus.documents = sorted_docs
        >>> corp1 = sample_corpus.clone()
        >>> corp1.documents = corp1.documents[:10]
        >>> corp2 = sample_corpus.clone()
        >>> corp2.documents = corp2.documents[10:]
        >>> sample_corpus == corp1 + corp2
        True

        :return: Corpus
        """
        if not isinstance(other, Corpus):
            raise NotImplementedError("Only a Corpus can be added to another Corpus.")

        output_corpus = self.clone()
        for document in other:
            output_corpus.documents.append(document)
        output_corpus.documents = sorted(output_corpus.documents)

        return output_corpus

    def clone(self):
        """
        Return a copy of the Corpus object

        :return: Corpus object

        >>> from corpus_analysis.corpus import Corpus
        >>> from corpus_analysis.testing.common import TEST_CORPUS_PATH
        >>> path = TEST_CORPUS_PATH
        >>> sample_corpus = Corpus(path)
        >>> corpus_copy = sample_corpus.clone()
        >>> len(corpus_copy) == len(sample_corpus)
        True

        """
        return deepcopy(self)

    def count_authors_by_gender(self, gender):
        """
        This function returns the number of authors in the corpus with the specified gender.

        *NOTE:* there must be an 'author_gender' field in the metadata of all documents.

        :param gender: gender identifier to search for in the metadata (i.e. 'female', 'male', etc.)
        :return: Number of authors of the given gender

        >>> from corpus_analysis.corpus import Corpus
        >>> from corpus_analysis.testing.common import (
        ...     TEST_CORPUS_PATH as path,
        ...     SMALL_TEST_CORPUS_CSV as path_to_csv
        ... )
        >>> c = Corpus(path, csv_path=path_to_csv, ignore_warnings = True)
        >>> c.count_authors_by_gender('female')
        7

        """
        count = 0
        for document in self.documents:
            try:
                if document.author_gender.lower() == gender.lower():
                    count += 1
            except AttributeError as err:
                raise MissingMetadataError(['author_gender']) from err

        return count

    def filter_by_gender(self, gender):
        """
        Return a new Corpus object that contains documents only with authors whose gender
        matches the given parameter.

        :param gender: gender identifier (i.e. 'male', 'female', 'unknown', etc.)
        :return: Corpus object

        >>> from corpus_analysis.corpus import Corpus
        >>> from corpus_analysis.testing.common import (
        ...     TEST_CORPUS_PATH as path,
        ...     LARGE_TEST_CORPUS_CSV as path_to_csv
        ... )
        >>> c = Corpus(path, csv_path=path_to_csv)
        >>> female_corpus = c.filter_by_gender('female')
        >>> len(female_corpus)
        39
        >>> female_corpus.documents[0].title
        'The Indiscreet Letter'

        >>> male_corpus = c.filter_by_gender('male')
        >>> len(male_corpus)
        59

        >>> male_corpus.documents[0].title
        'Lisbeth Longfrock'

        """

        return self.subcorpus('author_gender', gender)

    def get_wordcount_counter(self):
        """
        This function returns a Counter object that stores
        how many times each word appears in the corpus.

        :return: Python Counter object

        >>> from corpus_analysis.corpus import Corpus
        >>> from corpus_analysis.testing.common import (
        ...     TEST_CORPUS_PATH as path,
        ...     SMALL_TEST_CORPUS_CSV as path_to_csv
        ... )
        >>> c = Corpus(path, csv_path=path_to_csv, ignore_warnings = True)
        >>> word_count = c.get_wordcount_counter()
        >>> word_count['fire']
        157

        """
        corpus_counter = Counter()
        for current_document in self.documents:
            document_counter = current_document.get_wordcount_counter()
            corpus_counter += document_counter
        return corpus_counter

    def get_field_vals(self, field):
        """
        This function returns a sorted list of the values present
        in the corpus for a given metadata field.

        :param field: field to search for (i.e. 'location', 'author_gender', etc.)
        :return: list of strings

        >>> from corpus_analysis.corpus import Corpus
        >>> from corpus_analysis.testing.common import (
        ...     TEST_CORPUS_PATH as path,
        ...     LARGE_TEST_CORPUS_CSV as path_to_csv
        ... )
        >>> c = Corpus(path, name='sample_novels', csv_path=path_to_csv)
        >>> c.get_field_vals('author_gender')
        ['both', 'female', 'male']

        """

        if field not in self.metadata_fields:
            raise MissingMetadataError([field])

        values = set()
        for document in self.documents:
            values.add(getattr(document, field))

        return sorted(list(values))

    def subcorpus(self, metadata_field, field_value):
        """
        Returns a new Corpus object that contains only documents
        with a given field_value for metadata_field

        :param metadata_field: metadata field to search
        :param field_value: search term
        :return: Corpus object

        >>> from corpus_analysis.corpus import Corpus
        >>> from corpus_analysis.testing.common import (
        ...     TEST_CORPUS_PATH as path,
        ...     LARGE_TEST_CORPUS_CSV as path_to_csv
        ... )
        >>> corp = Corpus(path, csv_path=path_to_csv)
        >>> female_corpus = corp.subcorpus('author_gender','female')
        >>> len(female_corpus)
        39
        >>> female_corpus.documents[0].title
        'The Indiscreet Letter'

        >>> male_corpus = corp.subcorpus('author_gender','male')
        >>> len(male_corpus)
        59
        >>> male_corpus.documents[0].title
        'Lisbeth Longfrock'

        >>> eighteen_fifty_corpus = corp.subcorpus('date','1850')
        >>> len(eighteen_fifty_corpus)
        1
        >>> eighteen_fifty_corpus.documents[0].title
        'The Scarlet Letter'

        >>> jane_austen_corpus = corp.subcorpus('author','Austen, Jane')
        >>> len(jane_austen_corpus)
        2
        >>> jane_austen_corpus.documents[0].title
        'Emma'

        >>> england_corpus = corp.subcorpus('country_publication','England')
        >>> len(england_corpus)
        51
        >>> england_corpus.documents[0].title
        'Flatland'

        """

        if metadata_field not in self.metadata_fields:
            raise MissingMetadataError([metadata_field])

        corpus_copy = self.clone()
        corpus_copy.documents = []

        # adds documents to corpus_copy
        if metadata_field == 'date':
            for this_document in self.documents:
                if this_document.date == int(field_value):
                    corpus_copy.documents.append(this_document)
        else:
            for this_document in self.documents:
                try:
                    this_value = getattr(this_document, metadata_field, None)
                    if this_value is not None and this_value.lower() == field_value.lower():
                        corpus_copy.documents.append(this_document)
                except AttributeError:
                    continue

        return corpus_copy

    def multi_filter(self, characteristic_dict):
        """
        Returns a copy of the corpus, but with only the documents that fulfill the metadata
        parameters passed in by characteristic_dict. Multiple metadata keys can be searched
        at one time, provided that the metadata is available for the documents in the corpus.

        :param characteristic_dict: dict with metadata fields as keys and search terms as values
        :return: Corpus object

        >>> from corpus_analysis.corpus import Corpus
        >>> from corpus_analysis.testing.common import (
        ...     TEST_CORPUS_PATH as path,
        ...     LARGE_TEST_CORPUS_CSV as path_to_csv
        ... )
        >>> c = Corpus(path, csv_path=path_to_csv)
        >>> corpus_filter = {'author_gender': 'male'}
        >>> len(c.multi_filter(corpus_filter))
        59

        >>> corpus_filter['filename'] = 'aanrud_longfrock.txt'
        >>> len(c.multi_filter(corpus_filter))
        1
        """

        corpus_copy = self.clone()
        corpus_copy.documents = []

        for metadata_field in characteristic_dict:
            if metadata_field not in self.metadata_fields:
                raise MissingMetadataError([metadata_field])

        for this_document in self.documents:
            add_document = True
            for metadata_field in characteristic_dict:
                if metadata_field == 'date':
                    if this_document.date != int(characteristic_dict['date']):
                        add_document = False
                else:
                    metadata_value = getattr(this_document, metadata_field)
                    if metadata_value != characteristic_dict[metadata_field]:
                        add_document = False
            if add_document:
                corpus_copy.documents.append(this_document)

        if not corpus_copy:
            # displays for possible errors in field.value
            raise AttributeError('This corpus is empty. You may have mistyped something.')

        return corpus_copy

    def get_document(self, metadata_field, field_val):
        """
        .. _get-document:

        Returns a specific Document object from self.documents that has metadata matching field_val
        for metadata_field.

        This function will only return the first document in self.documents. It should only be used
        if you're certain there is only one match in the Corpus or if you're not picky about which
        Document you get.  If you want more selectivity use **get_document_multiple_fields**,
        or if you want multiple documents, use **subcorpus**.

        :param metadata_field: metadata field to search
        :param field_val: search term
        :return: Document Object

        >>> from corpus_analysis.corpus import Corpus
        >>> from corpus_analysis.common import MissingMetadataError
        >>> from corpus_analysis.testing.common import (
        ...     TEST_CORPUS_PATH as path,
        ...     LARGE_TEST_CORPUS_CSV as path_to_csv
        ... )

        >>> c = Corpus(path, csv_path=path_to_csv)
        >>> c.get_document("author", "Dickens, Charles")
        <Document (dickens_twocities)>
        >>> c.get_document("date", '1857')
        <Document (bronte_professor)>
        >>> try:
        ...     c.get_document("meme_quality", "over 9000")
        ... except MissingMetadataError as exception:
        ...     print(exception)
        This Corpus is missing the following metadata field:
            meme_quality
        In order to run this function, you must create a new metadata csv
        with this field and run Corpus.update_metadata().

        """

        if metadata_field not in self.metadata_fields:
            raise MissingMetadataError([metadata_field])

        if metadata_field == "date":
            field_val = int(field_val)

        for document in self.documents:
            if getattr(document, metadata_field) == field_val:
                return document

        raise ValueError("Document not found")

    def get_sample_text_passages(self, expression, no_passages):
        """
        Returns a specified number of example passages that include a certain expression.

        The number of passages that you request is a maximum number, and this function may return
        fewer if there are limited cases of a passage in the corpus.

        :param expression: expression to search for
        :param no_passages: number of passages to return
        :return: List of passages as strings

        >>> from corpus_analysis.corpus import Corpus
        >>> from corpus_analysis.testing.common import (
        ...     TEST_CORPUS_PATH as path,
        ...     LARGE_TEST_CORPUS_CSV as path_to_csv
        ... )
        >>> corpus = Corpus(path, csv_path=path_to_csv, ignore_warnings=True)
        >>> results = corpus.get_sample_text_passages('he cried', 2)
        >>> 'he cried' in results[0][1]
        True
        >>> 'he cried' in results[1][1]
        True
        """
        count = 0
        output = []
        phrase = nltk_tokenize.word_tokenize(expression)
        random.seed(expression)
        random_documents = self.documents.copy()
        random.shuffle(random_documents)

        for document in random_documents:
            if count >= no_passages:
                break
            current_document = document.get_tokenized_text()

            for i, _ in enumerate(current_document):
                if current_document[i] == phrase[0]:
                    if current_document[i:i + len(phrase)] == phrase:
                        passage = " ".join(current_document[i - 20:i + len(phrase) + 20])
                        output.append((document.filename, passage))
                        count += 1

        if len(output) <= no_passages:
            return output

        return output[:no_passages]

    def get_document_multiple_fields(self, metadata_dict):
        """
        Returns a specific Document object from the corpus that has metadata
        matching a given metadata dict.

        This method will only return the first document in the corpus.
        It should only be used if you're certain there is only one match in the Corpus
        or if you're not picky about which Document you get.

        If you want multiple documents, use **subcorpus**.

        :param metadata_dict: Dictionary with metadata fields as keys and search terms as values
        :return: Document object

        >>> from corpus_analysis.corpus import Corpus
        >>> from corpus_analysis.testing.common import (
        ...     TEST_CORPUS_PATH as path,
        ...     LARGE_TEST_CORPUS_CSV as path_to_csv
        ... )
        >>> c = Corpus(path, csv_path=path_to_csv)
        >>> c.get_document_multiple_fields({"author": "Dickens, Charles", "author_gender": "male"})
        <Document (dickens_twocities)>
        >>> c.get_document_multiple_fields({"author": "Chopin, Kate", "title": "The Awakening"})
        <Document (chopin_awakening)>

        """

        for field in metadata_dict.keys():
            if field not in self.metadata_fields:
                raise MissingMetadataError([field])

        for document in self.documents:
            match = True
            for field, val in metadata_dict.items():
                if getattr(document, field, None) != val:
                    match = False
            if match:
                return document

        raise ValueError("Document not found")

    def update_metadata(self, new_metadata_path):
        """
        Takes a filepath to a csv with new metadata and updates the metadata in the corpus'
        documents accordingly. The new file does not need to contain every metadata field in
        the documents - only the fields that you wish to update.

        NOTE: The csv file must include at least a filename for the documents that will be altered.

        :param new_metadata_path: Path to new metadata csv file
        :return: None
        """
        metadata = set()
        metadata.update(self.metadata_fields)

        if isinstance(new_metadata_path, str):
            new_metadata_path = Path(new_metadata_path)
        if not isinstance(new_metadata_path, Path):
            raise ValueError(
                f'new_metadata_path must be str or Path object, not type {type(new_metadata_path)}'
            )

        try:
            csv_list = load_csv_to_list(new_metadata_path)
        except FileNotFoundError as err:
            raise FileNotFoundError(
                "Could not find the metadata csv file for the "
                f"corpus in the expected location ({self.csv_path})."
            ) from err
        csv_reader = csv.DictReader(csv_list)

        for document_metadata in csv_reader:
            document_metadata = dict(document_metadata)
            metadata.update(list(document_metadata))
            try:
                document = self.get_document('filename', document_metadata['filename'])
            except ValueError as err:
                raise ValueError(
                    f"Document {document_metadata['filename']} not found in corpus"
                ) from err

            document.update_metadata(document_metadata)

        self.metadata_fields = list(metadata)


if __name__ == '__main__':
    from dh_testers.testRunner import main_test
    main_test()
