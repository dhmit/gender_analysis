import re
import string
from collections import Counter
from pathlib import Path

from gutenberg_cleaner import simple_cleaner
from more_itertools import windowed
import nltk

from gender_analysis.text import common


class Document:
    """
    The Document class loads and holds the full text and
    metadata (author, title, publication date, etc.) of a document

    :param metadata_dict: Dictionary with metadata fields as keys and data as values

    >>> from gender_analysis import Document
    >>> from pathlib import Path
    >>> from gender_analysis.testing.common import TEST_DATA_DIR
    >>> document_metadata = {'author': 'Austen, Jane', 'title': 'Persuasion', 'date': '1818',
    ...                      'filename': 'austen_persuasion.txt',
    ...                      'filepath': Path(TEST_DATA_DIR,
    ...                                       'sample_novels', 'texts', 'austen_persuasion.txt')}
    >>> austen = Document(document_metadata)
    >>> type(austen.text)
    <class 'str'>
    >>> len(austen.text)
    466887
    """

    def __init__(self, metadata_dict):
        if not isinstance(metadata_dict, dict):
            raise TypeError(
                'metadata must be passed in as a dictionary value'
            )

        # Check that the essential attributes for the document exists.
        if 'filename' not in metadata_dict:
            raise ValueError(str(metadata_dict) + 'metadata_dict must have an entry for filename')

        self.members = list(metadata_dict.keys())

        for key in metadata_dict:
            if hasattr(self, str(key)):
                raise KeyError(
                    'Key name ',
                    str(key),
                    ' is reserved in the Document class. Please use another name.'
                )
            setattr(self, str(key), metadata_dict[key])

        # optional attributes
        # Check that the date is a year (where negative represent the BCE year)
        if 'date' in metadata_dict:
            if not re.match(r'^[-]?\d+$', metadata_dict['date']):
                raise ValueError('The document date should be a year, not',
                                 f'{metadata_dict["date"]}. Full metadata: {metadata_dict}')

        try:
            self.date = int(metadata_dict['date'])
        except KeyError:
            self.date = None

        self._word_counts_counter = None
        self._word_count = None
        self._tokenized_text = None

        if not metadata_dict['filename'].endswith('.txt'):
            raise ValueError(
                f"The document filename {metadata_dict['filename']}"
                + f"does not end in .txt . Full metadata: '{metadata_dict}.'"
            )


        if 'label' not in metadata_dict:
            self.label = self.filename[0:len(self.filename) - 4]

        self.text = self._load_document_text()

    @property
    def word_count(self):
        """
        Lazy-loading for **Document.word_count** attribute.
        Returns the number of words in the document.
        The word_count attribute is useful for the get_word_freq function.
        However, it is performance-wise costly, so it's only loaded when it's actually required.

        :return: Number of words in the document's text as an int

        >>> from gender_analysis import Document
        >>> from pathlib import Path
        >>> from gender_analysis.testing.common import TEST_DATA_DIR
        >>> document_metadata = {'author': 'Austen, Jane', 'title': 'Persuasion', 'date': '1818',
        ...                      'filename': 'austen_persuasion.txt',
        ...                      'filepath': Path(TEST_DATA_DIR, 'sample_novels',
        ...                                       'texts', 'austen_persuasion.txt')}
        >>> austen = Document(document_metadata)
        >>> austen.word_count
        83285

        """

        if self._word_count is None:
            self._word_count = len(self.get_tokenized_text())
        return self._word_count

    def __str__(self):
        """
        Overrides python print method for user-defined objects for Document class
        Returns the filename without the extension - author and title word
        :return: str

        >>> from gender_analysis import Document
        >>> from pathlib import Path
        >>> from gender_analysis.testing.common import TEST_DATA_DIR
        >>> document_metadata = {'author': 'Austen, Jane', 'title': 'Persuasion', 'date': '1818',
        ...                      'filename': 'austen_persuasion.txt',
        ...                      'filepath': Path(TEST_DATA_DIR, 'sample_novels',
        ...                                       'texts', 'austen_persuasion.txt')}
        >>> austen = Document(document_metadata)
        >>> document_string = str(austen)
        >>> document_string
        'austen_persuasion'
        """
        return self.label

    def __repr__(self):
        '''
        Overrides the built-in __repr__ method
        Returns the object type (Document) and then the filename without the extension
            in <>.

        :return: string

        >>> from gender_analysis import Document
        >>> from pathlib import Path
        >>> from gender_analysis.testing.common import TEST_DATA_DIR
        >>> document_metadata = {'author': 'Austen, Jane', 'title': 'Persuasion', 'date': '1818',
        ...                      'filename': 'austen_persuasion.txt',
        ...                      'filepath': Path(TEST_DATA_DIR, 'sample_novels',
        ...                                       'texts', 'austen_persuasion.txt')}
        >>> austen = Document(document_metadata)
        >>> repr(austen)
        '<Document (austen_persuasion)>'
        '''
        return f'<Document ({self.label})>'

    def __eq__(self, other):
        """
        Overload the equality operator to enable comparing and sorting documents.
        Returns True if the document filenames and text are the same.

        >>> from gender_analysis import Document
        >>> from pathlib import Path
        >>> from gender_analysis.testing.common import TEST_DATA_DIR
        >>> austen_metadata = {'author': 'Austen, Jane', 'title': 'Persuasion', 'date': '1818',
        ...                    'filename': 'austen_persuasion.txt',
        ...                    'filepath': Path(TEST_DATA_DIR, 'sample_novels',
        ...                                     'texts', 'austen_persuasion.txt')}
        >>> austen = Document(austen_metadata)
        >>> austen2 = Document(austen_metadata)
        >>> austen == austen2
        True
        >>> austen.text += 'no longer equal'
        >>> austen == austen2
        False

        :return: bool
        """
        if not isinstance(other, Document):
            raise NotImplementedError("Only a Document can be compared to another Document.")

        attributes_required_to_be_equal = ['filename']

        for attribute in attributes_required_to_be_equal:
            if not hasattr(other, attribute):
                raise common.MissingMetadataError(
                    [attribute], f'{str(other)} lacks attribute {attribute}.'
                )
            if getattr(self, attribute) != getattr(other, attribute):
                return False

        if self.text != other.text:
            return False

        return True

    def __lt__(self, other):
        """
        Overload less than operator to enable comparing and sorting documents.

        Sorts first by author, title, and then date.

        If these are not available, it sorts by filenames.

        >>> from gender_analysis import Document
        >>> from pathlib import Path
        >>> from gender_analysis.testing.common import TEST_DATA_DIR
        >>> austen_metadata = {'author': 'Austen, Jane', 'title': 'Persuasion', 'date': '1818',
        ...                    'filename': 'austen_persuasion.txt',
        ...                    'filepath': Path(TEST_DATA_DIR, 'sample_novels',
        ...                                     'texts', 'austen_persuasion.txt')}
        >>> austen = Document(austen_metadata)
        >>> hawthorne_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter',
        ...                       'date': '1850', 'filename': 'hawthorne_scarlet.txt',
        ...                       'filepath': Path(TEST_DATA_DIR, 'sample_novels',
        ...                                        'texts', 'hawthorne_scarlet.txt')}
        >>> hawthorne = Document(hawthorne_metadata)
        >>> hawthorne < austen
        False
        >>> austen < hawthorne
        True

        :return: bool
        """
        if not isinstance(other, Document):
            raise NotImplementedError("Only a Document can be compared to another Document.")

        try:
            return (self.author, self.title, self.date) < (other.author, other.title, other.date)
        except AttributeError:
            return self.filename < other.filename

    def __hash__(self):
        """
        Makes the Document object hashable

        :return:
        """

        return hash(repr(self))

    @staticmethod
    def _clean_quotes(text):
        """
        Scans through the text and replaces all of the smart quotes and apostrophes with their
        "normal" ASCII variants

        >>> from gender_analysis import Document
        >>> smart_text = 'This is a “smart” phrase'
        >>> Document._clean_quotes(smart_text)
        'This is a "smart" phrase'

        :param text: The string to reformat
        :return: A string that is idential to `text`, except with its smart quotes exchanged
        """

        # Define the quotes that will be swapped out
        smart_quotes = {
            '“': '"',
            '”': '"',
            "‘": "'",
            "’": "'",
        }

        # Replace all entries one by one
        output_text = text
        for quote in smart_quotes:
            output_text = output_text.replace(quote, smart_quotes[quote])

        return output_text

    @staticmethod
    def _gutenberg_cleaner(text):

        """
        Checks to see if a given text is from Project Gutenberg.
        If it is, removes the header + footer.

        :param text: The string to reformat
        :return: A string that is idential to 'text' unless 'text' is from Gutenberg, in which case
        the Gutenberg header and footer is removed
        """

        output_text = text
        beginning = text[0:100]
        if "project gutenberg" in beginning.lower():
            output_text = simple_cleaner(output_text)

        return output_text

    def _load_document_text(self):
        """
        Loads the text of the document at the filepath specified in initialization.

        :return: str
        """

        file_path = Path(self.filepath)

        try:
            text = common.load_txt_to_string(file_path)
        except FileNotFoundError as original_err:
            err = (
                f'The filename {self.filename} present in your metadata csv does not exist in your '
                + 'files directory.\nPlease check that your metadata matches your dataset.'
            )
            raise FileNotFoundError(err) from original_err

        # Replace smart quotes with regular quotes to standardize input
        text = self._clean_quotes(text)
        return text

    def get_tokenized_text(self):
        """
        Tokenizes the text and returns it as a list of tokens, while removing all punctuation.

        Note: This does not currently properly handle dashes or contractions.

        :return: List of each word in the Document

        >>> from gender_analysis import Document
        >>> from pathlib import Path
        >>> from gender_analysis.testing.common import TEST_DATA_DIR
        >>> document_metadata = {'author': 'Austen, Jane', 'title': 'Persuasion', 'date': '1818',
        ...                      'filename': 'test_text_1.txt',
        ...                      'filepath': Path(TEST_DATA_DIR,
        ...                                       'document_test_files', 'test_text_1.txt')}
        >>> austin = Document(document_metadata)
        >>> tokenized_text = austin.get_tokenized_text()
        >>> tokenized_text
        ['allkinds', 'of', 'punctuation', 'and', 'special', 'chars']

        """

        # Excluded characters: !"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~
        if self._tokenized_text is None:
            excluded_characters = set(string.punctuation)
            cleaned_text = ''
            for character in self.text:
                if character not in excluded_characters:
                    cleaned_text += character

            tokenized_text = cleaned_text.lower().split()
            self._tokenized_text = tokenized_text
            return tokenized_text

        else:
            return self._tokenized_text

    def find_quoted_text(self):
        """
        Finds all of the quoted statements in the document text.

        :return: List of strings enclosed in double-quotations

        >>> from gender_analysis import Document
        >>> from pathlib import Path
        >>> from gender_analysis.testing.common import TEST_DATA_DIR
        >>> document_metadata = {'author': 'Austen, Jane', 'title': 'Persuasion',
        ...                      'date': '1818', 'filename': 'test_text_0.txt',
        ...                      'filepath': Path(TEST_DATA_DIR,
        ...                                       'document_test_files', 'test_text_0.txt')}
        >>> document_novel = Document(document_metadata)
        >>> document_novel.find_quoted_text()
        ['"This is a quote"', '"This is my quote"']

        """
        text_list = self.text.split()
        quotes = []
        current_quote = []
        quote_in_progress = False
        quote_is_paused = False

        for word in text_list:
            if word[0] == "\"":
                quote_in_progress = True
                quote_is_paused = False
                current_quote.append(word)
            elif quote_in_progress:
                if not quote_is_paused:
                    current_quote.append(word)
                if word[-1] == "\"":
                    if word[-2] != ',':
                        quote_in_progress = False
                        quote_is_paused = False
                        quotes.append(' '.join(current_quote))
                        current_quote = []
                    else:
                        quote_is_paused = True

        return quotes

    def get_count_of_word(self, word):
        """
        .. _get-count-of-word:

        Returns the number of instances of a word in the text. Not case-sensitive.

        If this is your first time running this method, this can be slow.

        :param word: word to be counted in text
        :return: Number of occurences of the word, as an int

        >>> from gender_analysis import Document
        >>> from pathlib import Path
        >>> from gender_analysis.testing.common import TEST_DATA_DIR
        >>> document_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter',
        ...                      'date': '2018', 'filename': 'test_text_2.txt',
        ...                      'filepath': Path(TEST_DATA_DIR,
        ...                                       'document_test_files', 'test_text_2.txt')}
        >>> scarlett = Document(document_metadata)
        >>> scarlett.get_count_of_word("sad")
        4
        >>> scarlett.get_count_of_word('ThisWordIsNotInTheWordCounts')
        0

        """

        # If word_counts were not previously initialized, do it now and store it for the future.
        if not self._word_counts_counter:
            self._word_counts_counter = Counter(self.get_tokenized_text())

        return self._word_counts_counter[word]

    def get_wordcount_counter(self):
        """
        Returns a counter object of all of the words in the text.

        If this is your first time running this method, this can be slow.

        :return: Python Counter object

        >>> from gender_analysis import Document
        >>> from pathlib import Path
        >>> from gender_analysis.testing.common import TEST_DATA_DIR
        >>> document_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter',
        ...                      'date': '2018', 'filename': 'test_text_10.txt',
        ...                      'filepath': Path(TEST_DATA_DIR,
        ...                                       'document_test_files', 'test_text_10.txt')}
        >>> scarlett = Document(document_metadata)
        >>> scarlett.get_wordcount_counter()
        Counter({'was': 2, 'convicted': 2, 'hester': 1, 'of': 1, 'adultery': 1})

        """

        # If word_counts were not previously initialized, do it now and store it for the future.
        if not self._word_counts_counter:
            self._word_counts_counter = Counter(self.get_tokenized_text())
        return self._word_counts_counter

    def words_associated(self, target_word):
        """
        .. _words-associated:

        Returns a Counter of the words found after a given word.

        In the case of double/repeated words, the counter would include the word itself and the next
        new word.

        Note: words always return lowercase.

        :param word: Single word to search for in the document's text
        :return: a Python Counter() object with {associated_word: occurrences}

        >>> from gender_analysis import Document
        >>> from pathlib import Path
        >>> from gender_analysis.testing.common import TEST_DATA_DIR
        >>> document_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter',
        ...                      'date': '2018', 'filename': 'test_text_11.txt',
        ...                      'filepath': Path(TEST_DATA_DIR,
        ...                                       'document_test_files', 'test_text_11.txt')}
        >>> scarlett = Document(document_metadata)
        >>> scarlett.words_associated("his")
        Counter({'cigarette': 1, 'speech': 1})

        """
        target_word = target_word.lower()
        word_count = Counter()
        check = False
        text = self.get_tokenized_text()

        for word in text:
            if check:
                word_count[word] += 1
                check = False
            if word == target_word:
                check = True
        return word_count

    # pylint: disable=line-too-long
    def get_word_windows(self, search_terms, window_size=2):
        """
        .. _get-word-windows:

        Finds all instances of `word` and returns a counter of the words around it.
        window_size is the number of words before and after to return, so the total window is
        2*window_size + 1.

        This is not case sensitive.

        :param search_terms: String or list of strings to search for
        :param window_size: integer representing number of words to search for in either direction
        :return: Python Counter object

        >>> from gender_analysis import Document
        >>> from pathlib import Path
        >>> from gender_analysis.testing.common import TEST_DATA_DIR
        >>> document_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter',
        ...                      'date': '2018', 'filename': 'test_text_12.txt',
        ...                      'filepath': Path(TEST_DATA_DIR,
        ...                                       'document_test_files', 'test_text_12.txt')}
        >>> scarlett = Document(document_metadata)

        search_terms can be either a string...

        >>> scarlett.get_word_windows("his", window_size=2)
        Counter({'he': 1, 'lit': 1, 'cigarette': 1, 'and': 1, 'then': 1, 'began': 1, 'speech': 1, 'which': 1})

        ... or a list of strings.

        >>> scarlett.get_word_windows(['purse', 'tears'])
        Counter({'her': 2, 'of': 1, 'and': 1, 'handed': 1, 'proposal': 1, 'drowned': 1, 'the': 1})

        """

        if isinstance(search_terms, str):
            search_terms = [search_terms]

        search_terms = set(i.lower() for i in search_terms)

        counter = Counter()

        for text_window in windowed(self.get_tokenized_text(), 2 * window_size + 1):
            if text_window[window_size] in search_terms:
                for surrounding_word in text_window:
                    if surrounding_word not in search_terms:
                        counter[surrounding_word] += 1

        return counter

    def get_word_freq(self, word):
        """
        .. _get-word-freq:

        Returns the frequency of appearance of a word in the document

        :param word: str to search for in document
        :return: float representing the portion of words in the text that are the parameter word

        >>> from gender_analysis import Document
        >>> from pathlib import Path
        >>> from gender_analysis.testing.common import TEST_DATA_DIR
        >>> document_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter',
        ...                      'date': '1900', 'filename': 'test_text_2.txt',
        ...                      'filepath': Path(TEST_DATA_DIR,
        ...                                       'document_test_files', 'test_text_2.txt')}
        >>> scarlett = Document(document_metadata)
        >>> frequency = scarlett.get_word_freq('sad')
        >>> frequency
        0.13333333333333333
        """

        word_frequency = self.get_count_of_word(word) / self.word_count
        return word_frequency

    def get_part_of_speech_tags(self):
        """
        .. _get-pos:

        Returns the part of speech tags as a list of tuples. The first part of each tuple is the
        term, the second one the part of speech tag.

        Note: the same word can have a different part of speech tags. In the example below,
        see "refuse" and "permit".

        :return: List of tuples (term, speech_tag)

        >>> from gender_analysis import Document
        >>> from pathlib import Path
        >>> from gender_analysis.testing.common import TEST_DATA_DIR
        >>> document_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter',
        ...                      'date': '1900', 'filename': 'test_text_13.txt',
        ...                      'filepath': Path(TEST_DATA_DIR,
        ...                                       'document_test_files', 'test_text_13.txt')}
        >>> document = Document(document_metadata)
        >>> document.get_part_of_speech_tags()[:4]
        [('They', 'PRP'), ('refuse', 'VBP'), ('to', 'TO'), ('permit', 'VB')]
        >>> document.get_part_of_speech_tags()[-4:]
        [('the', 'DT'), ('refuse', 'NN'), ('permit', 'NN'), ('.', '.')]

        """

        common.download_nltk_package_if_not_present('tokenizers/punkt')
        common.download_nltk_package_if_not_present('taggers/averaged_perceptron_tagger')

        text = nltk.word_tokenize(self.text)
        pos_tags = nltk.pos_tag(text)
        return pos_tags

    def update_metadata(self, new_metadata):
        """
        Updates the metadata of the document without requiring a complete reloading
        of the text and other properties.

        'filename' cannot be updated with this method.

        :param new_metadata: dict of new metadata to apply to the document
        :return: None

        This can be used to correct mistakes in the metadata:

        >>> from gender_analysis import Document
        >>> from gender_analysis.testing.common import TEST_CORPUS_PATH
        >>> from pathlib import Path
        >>> metadata = {'filename': 'aanrud_longfrock.txt',
        ...             'filepath': Path(TEST_CORPUS_PATH, 'aanrud_longfrock.txt'),
        ...             'date': '2098'}
        >>> d = Document(metadata)
        >>> new_metadata = {'date': '1903'}
        >>> d.update_metadata(new_metadata)
        >>> d.date
        1903

        Or it can be used to add completely new attributes:

        >>> new_attribute = {'cookies': 'chocolate chip'}
        >>> d.update_metadata(new_attribute)
        >>> d.cookies
        'chocolate chip'
        """

        if not isinstance(new_metadata, dict):
            raise ValueError(
                f'new_metadata must be a dictionary of metadata keys, not type {type(new_metadata)}'
            )
        if 'filename' in new_metadata and new_metadata['filename'] != self.filename:
            raise KeyError(
                'You cannot update the filename of a document; '
                f'consider removing {str(self)} from the Corpus object '
                'and adding the document again with the updated filename'
            )

        for key in new_metadata:
            if key == 'date':
                try:
                    new_metadata[key] = int(new_metadata[key])
                except ValueError as err:
                    raise ValueError(
                        f"the metadata field 'date' must be a number for document {self.filename},"
                        f" not '{new_metadata['date']}'"
                    ) from err
            setattr(self, key, new_metadata[key])
