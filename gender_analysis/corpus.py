import csv
import random
from nltk.tokenize import word_tokenize
from pathlib import Path
from collections import Counter

from gender_analysis import common
from gender_analysis.document import Document
from gender_analysis.gutenburg_loader import download_gutenberg_if_not_locally_available


class Corpus(common.FileLoaderMixin):

    """The corpus class is used to load the metadata and full
    texts of all novels in a corpus

    Once loaded, each corpus contains a list of Document objects

    >>> from gender_analysis.corpus import Corpus
    >>> c = Corpus('sample_novels')
    >>> type(c.novels), len(c)
    (<class 'list'>, 99)

    >>> c.novels[0].author
    'Aanrud, Hans'

    You can use 'test_corpus' to load a test corpus of 10 novels:
    >>> test_corpus = Corpus('test_corpus')
    >>> len(test_corpus)
    10

    """

    def __init__(self, path_to_files, name=None, csv_path=None, pickle_on_load=False):

        """

        :param path_to_files: Must be either the path to a directory of txt files or an already-pickled corpus
        :param name: Optional name of the corpus, for ease of use and readability
        :param csv_path: Optional path to a csv metadata file
        :param pickle_on_load:
        """

        if isinstance(path_to_files, str):
            path_to_files = Path(path_to_files)
        if not isinstance(path_to_files, Path):
            raise ValueError(f'path_to_files must be a str or Path object, not type {type(path_to_files)}')

        self.name = name
        self.csv_path = csv_path
        self.path_to_files = path_to_files
        self.novels = []

        if self.path_to_files.suffix == '.pgz':
            pass
        # if self.name == 'gutenberg' and csv_path is None:
        #     download_gutenberg_if_not_locally_available()
        #
        # self.load_test_corpus = False
        # if self.name == 'test_corpus' and csv_path is None:
        #     self.load_test_corpus = True
        #     self.name = 'sample_novels'

    def __len__(self):
        """
        For convenience: returns the number of novels in
        the corpus.

        >>> from gender_analysis.corpus import Corpus
        >>> c = Corpus('sample_novels')
        >>> len(c)
        99

        >>> female_corpus = c.filter_by_gender('female')
        >>> len(female_corpus)
        39

        :return: int
        """
        return len(self.novels)

    def __iter__(self):
        """
        Yield each of the novels from the .novels list.

        For convenience.

        >>> from gender_analysis.corpus import Corpus
        >>> c = Corpus('sample_novels')
        >>> titles = []
        >>> for this_novel in c:
        ...    titles.append(this_novel.title)
        >>> titles #doctest: +ELLIPSIS
        ['Lisbeth Longfrock', 'Flatland', ... 'The Heir of Redclyffe']

        """
        for this_novel in self.novels:
            yield this_novel

    def __eq__(self, other):
        """
        Returns true if both corpora contain the same novels
        Note: ignores differences in the corpus name as that attribute is not used apart from
        initializing a corpus.
        Presumes the novels to be sorted. (They get sorted by the initializer)

        >>> from gender_analysis.corpus import Corpus
        >>> sample_corpus = Corpus('sample_novels')
        >>> sample_corpus.novels = sample_corpus.novels[:20]
        >>> male_corpus = sample_corpus.filter_by_gender('male')
        >>> female_corpus = sample_corpus.filter_by_gender('female')
        >>> merged_corpus = male_corpus + female_corpus
        >>> merged_corpus == sample_corpus
        True
        >>> sample_corpus == merged_corpus + male_corpus
        False

        :return: bool
        """

        if not isinstance(other, Corpus):
            raise NotImplementedError("Only a Corpus can be compared to another Corpus.")

        if len(self) != len(other):
            return False

        for i in range(len(self)):
            if self.novels[i] != other.novels[i]:
                return False

        return True

    def __add__(self, other):
        """
        Adds two corpora together and returns a copy of the result
        Note: retains the name of the first corpus

        >>> from gender_analysis.corpus import Corpus
        >>> sample_corpus = Corpus('sample_novels')
        >>> sample_corpus.novels = sample_corpus.novels[:20]
        >>> male_corpus = sample_corpus.filter_by_gender('male')
        >>> female_corpus = sample_corpus.filter_by_gender('female')
        >>> merged_corpus = male_corpus + female_corpus
        >>> merged_corpus == sample_corpus
        True

        :return: Corpus
        """
        if not isinstance(other, Corpus):
            raise NotImplementedError("Only a Corpus can be added to another Corpus.")

        output_corpus = self.clone()
        for novel in other:
            output_corpus.novels.append(novel)
        output_corpus.novels = sorted(output_corpus.novels)

        return output_corpus

    def clone(self):
        """
        Return a copy of this Corpus

        >>> from gender_analysis.corpus import Corpus
        >>> sample_corpus = Corpus('sample_novels')
        >>> corpus_copy = sample_corpus.clone()
        >>> len(corpus_copy) == len(sample_corpus)
        True

        :return: Corpus
        """
        corpus_copy = Corpus()
        corpus_copy.name = self.name
        corpus_copy.novels = self.novels[:]
        return corpus_copy

    def _load_novels(self):
        novels = []

        # relative_csv_path = (self.relative_corpus_path
        #                      / f'{self.name}.csv')
        try:
            csv_file = self.load_file(self.csv_path)
        except FileNotFoundError:
            err = "Could not find the metadata csv file for the "
            err += "'{self.name}' corpus in the expected location "
            err += f"({self.csv_path})."
            raise FileNotFoundError(err)
        csv_reader = csv.DictReader(csv_file)

        for novel_metadata in csv_reader:
            novel_metadata['name'] = self.name
            this_novel = Document(document_metadata_dict=novel_metadata)
            novels.append(this_novel)
            if self.load_test_corpus and len(novels) == 10:
                break

        return sorted(novels)

    def count_authors_by_gender(self, gender):
        """
        This function returns the number of authors with the
        specified gender (male, female, non-binary, unknown)

        >>> from gender_analysis.corpus import Corpus
        >>> c = Corpus('sample_novels')
        >>> c.count_authors_by_gender('female')
        39

        Accepted inputs are 'male', 'female', 'non-binary' and 'unknown'
        but no abbreviations.

        >>> c.count_authors_by_gender('m')
        Traceback (most recent call last):
        ValueError: Gender must be male, female, non-binary, unknown but not m.

        :rtype: int
        """
        filtered_corpus = self.filter_by_gender(gender)
        return len(filtered_corpus)

    def filter_by_gender(self, gender):
        """
        Return a new Corpus object that contains only authors whose gender
        matches the given parameter.

        >>> from gender_analysis.corpus import Corpus
        >>> c = Corpus('sample_novels')
        >>> female_corpus = c.filter_by_gender('female')
        >>> len(female_corpus)
        39
        >>> female_corpus.novels[0].title
        'The Indiscreet Letter'

        >>> male_corpus = c.filter_by_gender('male')
        >>> len(male_corpus)
        59

        >>> male_corpus.novels[0].title
        'Lisbeth Longfrock'

        :param gender: gender name
        :return: Corpus
        """

        return self.subcorpus('author_gender', gender)

    def get_wordcount_counter(self):
        """
        This function returns a Counter telling how many times a word appears in an entire
        corpus

        >>> from gender_analysis.corpus import Corpus
        >>> c = Corpus('sample_novels')
        >>> c.get_wordcount_counter()['fire']
        2269

        """
        corpus_counter = Counter()
        for current_novel in self.novels:
            novel_counter = current_novel.get_wordcount_counter()
            corpus_counter += novel_counter
        return corpus_counter

    def get_corpus_metadata(self):
        """
        This function returns a sorted list of all metadata fields
        in the corpus as strings. This is different from the get_metadata_fields;
        this returns the fields which are specific to the corpus it is being called on.
        >>> from gender_analysis.corpus import Corpus
        >>> c = Corpus('sample_novels')
        >>> c.get_corpus_metadata()
        ['author', 'author_gender', 'name', 'country_publication', 'date', 'filename', 'notes', 'title']

        :return: list
        """
        metadata_fields = set()
        for novel in self.novels:
            for field in novel.getmembers():
                metadata_fields.add(field)
        return sorted(list(metadata_fields))

    def get_field_vals(self,field):
        """
        This function returns a sorted list of all values for a
        particular metadata field as strings.

        >>> from gender_analysis.corpus import Corpus
        >>> c = Corpus('sample_novels')
        >>> c.get_field_vals('name')
        ['sample_novels']

        :param field: str
        :return: list
        """
        metadata_fields = self.get_corpus_metadata()

        if field not in metadata_fields:
            raise ValueError(
                f'\'{field}\' is not a valid metadata field for this corpus'
            )

        values = set()
        for novel in self.novels:
            values.add(getattr(novel,field))

        return sorted(list(values))

    def subcorpus(self,metadata_field,field_value):
        """
        This method takes a metadata field and value of that field and returns
        a new Corpus object which includes the subset of novels in the original
        Corpus that have the specified value for the specified field.

        Supported metadata fields are 'author', 'author_gender', 'name',
        'country_publication', 'date'

        >>> from gender_analysis.corpus import Corpus

        >>> corp = Corpus('sample_novels')
        >>> female_corpus = corp.subcorpus('author_gender','female')
        >>> len(female_corpus)
        39
        >>> female_corpus.novels[0].title
        'The Indiscreet Letter'

        >>> male_corpus = corp.subcorpus('author_gender','male')
        >>> len(male_corpus)
        59
        >>> male_corpus.novels[0].title
        'Lisbeth Longfrock'

        >>> eighteen_fifty_corpus = corp.subcorpus('date','1850')
        >>> len(eighteen_fifty_corpus)
        1
        >>> eighteen_fifty_corpus.novels[0].title
        'The Scarlet Letter'

        >>> jane_austen_corpus = corp.subcorpus('author','Austen, Jane')
        >>> len(jane_austen_corpus)
        2
        >>> jane_austen_corpus.novels[0].title
        'Emma'

        >>> england_corpus = corp.subcorpus('country_publication','England')
        >>> len(england_corpus)
        51
        >>> england_corpus.novels[0].title
        'Flatland'

        :param metadata_field: str
        :param field_value: str
        :return: Corpus
        """

        supported_metadata_fields = ('author', 'author_gender', 'name',
                                     'country_publication', 'date')
        if metadata_field not in supported_metadata_fields:
            raise ValueError(
                f'Metadata field must be {", ".join(supported_metadata_fields)} '
                + f'but not {metadata_field}.')

        corpus_copy = self.clone()
        corpus_copy.novels = []

        # adds novels to corpus_copy
        if metadata_field == 'date':
            for this_novel in self.novels:
                if this_novel.date == int(field_value):
                    corpus_copy.novels.append(this_novel)
        else:
            for this_novel in self.novels:
                if getattr(this_novel, metadata_field) == field_value.lower():
                    corpus_copy.novels.append(this_novel)

        if not corpus_copy:
            # displays for possible errors in field.value
            err = f'This corpus is empty. You may have mistyped something.'
            raise AttributeError(err)

        return corpus_copy

    def multi_filter(self, characteristic_dict):
        """
        This method takes a dictionary of metadata fields and corresponding values
        and returns a Corpus object which is the subcorpus of the input corpus which
        satisfies all the specified constraints.

        #>>> from gender_analysis.corpus import Corpus
        #>>> c = Corpus('sample_novels')
        #>>> characteristics = {'author':'female',
                                'country_publication':'England'}
        #>>> subcorpus_multi_filtered = c.multi_filter(characteristics)
        #>>> female_subcorpus = c.filter_by_gender('female')
        #>>> subcorpus_repeated_method = female_subcorpus.Subcorpus('country_publication','England')
        #>>> subcorpus_multi_filtered == subcorpus_repeated_method
        True

        :param characteristic_dict: dict
        :return: Corpus
        """

        new_corp = self.clone()
        metadata_fields = self.get_corpus_metadata()

        for field in characteristic_dict:
            if field not in metadata_fields:
                raise ValueError(f'\'{field}\' is not a valid metadata field for this corpus')
            new_corp = new_corp.subcorpus(field, characteristic_dict[field])

        return new_corp

        #TODO: add date range support
        #TODO: apply all filters at once instead of recursing Subcorpus method

    def multi_filter_integrated(self,characteristic_dict):
        """
        This needs documentation and tests but it's 5:59! To be added after moratorium.
        :param characteristic_dict:
        :return:
        """
        supported_metadata_fields = ('author', 'author_gender', 'name',
                                     'country_publication', 'date')

        corpus_copy = self.clone()
        corpus_copy.novels = []

        for metadata_field in characteristic_dict:
            if metadata_field not in supported_metadata_fields:
                raise ValueError(
                    f'Metadata field must be {", ".join(supported_metadata_fields)} '
                    + f'but not {metadata_field}.')

        for this_novel in self.novels:
            add_novel = True
            for metadata_field in characteristic_dict:
                if metadata_field == 'date':
                    if this_novel.date != int(characteristic_dict['date']):
                        add_novel = False
                else:
                    if getattr(this_novel, metadata_field) != field_value:
                        add_novel = False
            if add_novel:
                corpus_copy.novels.append(this_novel)

        if not corpus_copy:
            # displays for possible errors in field.value
            err = f'This corpus is empty. You may have mistyped something.'
            raise AttributeError(err)

        return corpus_copy


    def get_novel(self, metadata_field, field_val):
        """
        Returns a specific Document object from self.novels that has metadata matching field_val for
        metadata_field.  Otherwise raises a ValueError.
        N.B. This function will only return the first novel in the self.novels (which is sorted as
        defined by the Document.__lt__ function).  It should only be used if you're certain there is
        only one match in the Corpus or if you're not picky about which Document you get.  If you want
        more selectivity use get_novel_multiple_fields, or if you want multiple novels use the subcorpus
        function.

        >>> from gender_analysis.corpus import Corpus
        >>> c = Corpus('sample_novels')
        >>> c.get_novel("author", "Dickens, Charles")
        <Document (dickens_twocities)>
        >>> c.get_novel("date", '1857')
        <Document (bronte_professor)>
        >>> try:
        ...     c.get_novel("meme_quality", "over 9000")
        ... except AttributeError as exception:
        ...     print(exception)
        Metadata field meme_quality invalid for this corpus

        :param metadata_field: str
        :param field_val: str/int
        :return: Document
        """

        if metadata_field not in get_metadata_fields(self.name):
            raise AttributeError(f"Metadata field {metadata_field} invalid for this corpus")

        if (metadata_field == "date" or metadata_field == "gutenberg_id"):
            field_val = int(field_val)

        for novel in self.novels:
            if getattr(novel, metadata_field) == field_val:
                return novel

        raise ValueError("Document not found")

    def get_sample_text_passages(self, expression, no_passages):
        """
        Returns a specified number of example passages that include a certain expression.

        >>> corpus = Corpus('sample_novels')
        >>> corpus.get_sample_text_passages('he cried', 2)
        ('james_american.txt', 'flowing river” newman gave a great rap on the floor with his stick and a long grim laugh “good good” he cried “you go altogether too faryou overshoot the mark there isn’t a woman in the world as bad as you would')
        ('james_american.txt', 'the old woman’s hand in both his own and pressed it vigorously “i thank you ever so much for that” he cried “i want to be the first i want it to be my property and no one else’s you’re the wisest')

        """

        count = 0
        output = []
        phrase = word_tokenize(expression)
        random.seed(expression)
        random_novels = self.novels.copy()
        random.shuffle(random_novels)

        for novel in random_novels:
            if count >= no_passages:
                break
            current_novel = novel.get_tokenized_text()
            for index in range(len(current_novel)):
                if current_novel[index] == phrase[0]:
                    if current_novel[index:index+len(phrase)] == phrase:
                        passage = " ".join(current_novel[index-20:index+len(phrase)+20])
                        output.append((novel.filename, passage))
                        count += 1

        random.shuffle(output)
        print_count = 0
        for entry in output:
            if print_count == no_passages:
                break
            print_count += 1
            print(entry)


    def get_novel_multiple_fields(self, metadata_dict):
        """
        Returns a specific Document object from self.novels that has metadata that matches a partial
        dict of metadata.  Otherwise raises a ValueError.
        N.B. This method will only return the first novel in the self.novels (which is sorted as
        defined by the Document.__lt__ function).  It should only be used if you're certain there is
        only one match in the Corpus or if you're not picky about which Document you get.  If you want
        multiple novels use the subcorpus function.

        >>> from gender_analysis.corpus import Corpus
        >>> c = Corpus('sample_novels')
        >>> c.get_novel_multiple_fields({"author": "Dickens, Charles", "author_gender": "male"})
        <Document (dickens_twocities)>
        >>> c.get_novel_multiple_fields({"author": "Chopin, Kate", "title": "The Awakening"})
        <Document (chopin_awakening)>

        :param metadata_dict: dict
        :return: Document
        """

        for field in metadata_dict.keys():
            if field not in get_metadata_fields(self.name):
                raise AttributeError(f"Metadata field {field} invalid for this corpus")

        for novel in self.novels:
            match = True
            for field, val in metadata_dict.items():
                if getattr(novel, field) != val:
                    match = False
            if match:
                return novel

        raise ValueError("Document not found")


def get_metadata_fields(name):
    """
    Gives a list of all metadata fields for corpus
    >>> from gender_analysis import corpus
    >>> corpus.get_metadata_fields('gutenberg')
    ['gutenberg_id', 'author', 'date', 'title', 'country_publication', 'author_gender', 'subject', 'name', 'notes']

    :param: name: str
    :return: list
    """
    if name == 'sample_novels':
        return ['author', 'date', 'title', 'country_publication', 'author_gender', 'filename', 'notes']
    else:
        return common.METADATA_LIST


if __name__ == '__main__':
    from dh_testers.testRunner import main_test
    main_test()
