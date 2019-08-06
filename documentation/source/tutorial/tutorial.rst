==============
User's Guide
==============

If you have not yet installed the package, make sure to look at the Installation instructions:


.. toctree::
    :maxdepth: 2

    installation

.. note::
    This is a work in progress and will be updated as we prepare and expand tutorials. Check back soon!

Document Class
==============

The document class is the base-level unit of metadata storage. It is responsible for keeping track of the metadata,
text, and file location of a single document. Our analysis will work best if your text is split into documents such
that each document has its own author(s), publication date, etc. For example, a collection of Tweets should be split
by individual Tweet.

Initialization
--------------

``Document`` initialization will typically be handled by the ``Corpus`` object, but, in case we want to create an
individual Document, we call the ``Document.__init__`` function with a
metadata dictionary:

    >>> my_doc = Document({'filename': 'example.txt', 'filepath': 'path/to/example.txt',
    ...                     'author': 'John Doe', 'date': 2019})

Each key in the metadata dictionary becomes an attribute of the Document object. Only the ``'filename'`` attribute is
required and its value must end in '.txt'. However, the metadata should also have a ``'filepath'`` attribute that
points to the file of the document's text, as ``Document`` loads the text from this location. Other metadata fields may
be helpful for more powerful analyses if working with a corpus, but are not required.

Operations
----------

If we want to change the metadata of a document after initialization, we can call:
    >>> my_doc.update_metadata({'author': 'Jane Doe', 'date': 2018})

With a Document object, you can perform a variety of analyses within a single document:
    - Word count: ``my_doc.get_count_words()``
    - Distance between occurrences of words: ``my_doc.words_instance_dist()``
    - Adjectives associated with a certain gender: ``my_doc.find_male_adj()`` and ``my_doc.find_female_adj()``


Corpus Class
============

A ``Corpus`` object is the primary tool of the Gender Analysis Toolkit, as it is responsible for aggregating the metadata
and documents of a given collection.

Initialization
--------------

Before creating the corpus, you should first create a folder consisting of all of your documents as .txt files.

To create your own corpus to analyze, import Corpus from ``gender_analysis.corpus``:

    >>> from gender_analysis.corpus import Corpus

After that, create a Corpus object by passing in a full path to a file consisting of all of the documents to be analyzed.
This creates a list of Document objects within Corpus to keep track of the various .txt files:

    >>> corpus = Corpus('path/to/file/directory')

To allow for more powerful analysis, we can add metadata about each document to our Corpus object. First, create a .csv
file with the first row as metadata fields, such as ``'filename'``, ``'author'``, ``'publication_country'``, and
``'publication_date'`` separated by commas. Then add an additional row for every document in your corpus -- fill
in each row with the corresponding metadata field values.

In addition to a directory of txt files, pass in a second parameter, ``csv_path``, which is a full path to the csv
metadata file that you created.

    >>> corpus = Corpus('path/to/file/directory',
    ...                 csv_path='path/to/csv_file.csv')

This will automatically update all of the ``Document`` objects in the corpus object with the metadata provided in the
csv file.

To save some time, the ``Corpus`` class has an optional parameter, ``guess_author_genders``, that will attempt to guess
the gender of the authors in your corpus when set to ``True``. This will automatically create an ``author_gender`` field
in the internal metadata for all of your documents.


Visualizing Your Metadata
=========================

Metadata is the cornerstone of the Gender Analysis Toolkit, with it being required for many of the toolkit's functions
and aiding in the analysis of several more. Sometimes, it is helpful to get a broad look at some instances of your
metadata. For instance, how many of the authors in your corpus are male? How many are female? What time periods are you
looking at? The toolkit provides simple functions for you to visualize all of this, saved as .png files.

To create a visualization of the distribution of author genders, simply use:

    >>> import gender_analysis.analysis.metadata_visualizations
    >>> from gender_analysis import Corpus
    >>> my_corpus = Corpus('path/to/files', csv_path='path/to/metadata')
    >>> plot_gender_breakdown(my_corpus)

Similarly, you can visualize the publication locations via:

    >>> plot_pubcountries(my_corpus)

Or you can visualize publication years with:

    >>> plot_pubyears(my_corpus)

If you want to visualize all three at once, simply call:

    >>> create_corpus_visualizations_summary(my_corpus)


Adding Masculine and Feminine Words
===================================

The Gender Analysis Toolkit comes pre-loaded with several words that it identifies as "masculine" or"feminine", but
sometimes there are words that are unique to a corpus that are generally not considered gendered in other contexts.

Gender Analysis comes with global constants ``MASC_WORDS`` and ``FEM_WORDS``, which are both sets consisting of
masculine and feminine pronouns, respectively.

If your corpus frequently references, for example, a woman named Alice, and you would like the toolkit to recognize
the name Alice as female, then you can add 'Alice' to ``FEM_WORDS``:

    >>> from gender_analysis.common import MASC_WORDS, FEM_WORDS
    >>> FEM_WORDS.add('Alice')


Dunning Analysis
================

Dunning analysis tests for word distinctiveness between two corpora. Given two corpora, any one word present in both
has a Dunning log-likelihood value, which is a measure of how likely the frequency of the word in the first corpus is
greater than the frequency of the word in the second corpus.

Try it out with a single word between two corpora:

    >>> from gender_analysis.analysis import dunning
    >>> dunning.dunn_individual_word_by_corpus(corpus_1,
    ...                                        corpus_2,
    ...                                        my_word)

No metadata is required from either corpora.

Plot Dunning Values for Words Between Two Corpora
-------------------------------------------------

First, we need to calculate the results to be plotted.

    >>> from gender_analysis import Corpus
    >>> from gender_analysis.analysis import dunning
    >>> my_corpus1 = Corpus(filepath1)
    >>> my_corpus2 = Corpus(filepath2)
    >>> counter1 = my_corpus1.get_wordcount_counter()
    >>> counter2 = my_corpus2.get_wordcount_counter()
    >>> my_results = dunning.dunning_total(counter1, counter2)

Alternatively, we can set my_results to the output of any of the following functions:

    - ``dunning_words_by_author_gender``
    - ``male_characters_author_gender_differences``
    - ``female_characters_author_gender_differences``
    - ``masc_fem_associations_dunning``
    - ``compare_word_association_in_corpus_dunning``

Then, to plot Dunning values, call:

    >>> dunning.score_plot_to_show(my_results)

For relative word frequency instead of Dunning values:

    >>> dunning.freq_plot_to_show(my_results)

To find the distinctiveness of all shared words between two corpora, call:

    >>> dunning.dunning_total_by_corpus(my_corpus_1, my_corpus_2)

Again, no metadata is required of your two corpora.

Dunning analysis can also be used between corpora to compare associated words of a certain word. For example,
``female_characters_author_gender_differences`` will identify differences between how female characters are described
by male authors versus female authors. We also provide an equivalent function for words associated with male characters
and can compare associations with any word or list of words with ``compare_word_association_between_corpus_dunning``.
If we want to compare associations with the word 'money', we can call:

    >>> male_corpus = corpus.filter_by_gender('male')
    >>> female_corpus = corpus.filter_by_gender('female')
    >>> dunning.compare_word_association_between_corpus_dunning(word='money', corpus1=male_corpus, corpus2=female_corpus)

Or if we want to compare associations with several words relating to money, we can use:

    >>> words = ['money', 'dollars', 'pounds', 'euros', 'dollar', 'pound', 'euro', 'wealth', 'income']
    >>> dunning.compare_word_association_between_corpus_dunning(word=words, corpus1=male_corpus, corpus2=female_corpus)

Within a single corpus, we can compare words associated with two different words with
``compare_word_association_in_corpus_dunning``. A special case of this analysis is comparing words associated with
male and female characters, which is handled by ``masc_fem_associations_dunning``.

Gender Frequency Analysis
=========================

Gender frequency analysis tests for the frequencies of different genders in the documents given by the corpus.

First, create a corpus to analyze:

    >>> from gender_analysis import Corpus
    >>> my_corpus = Corpus('path/to/documents')

To get the relative frequency of female pronouns in each document in the corpus, call:

    >>> from gender_analysis.analysis.gender_frequency import *
    >>> document_pronoun_freq(my_corpus)

To get the proportion of instances of male and female pronouns that are the subject of a sentence, call:

    >>> subject_vs_object_pronoun_freqs(my_corpus)

This will return a tuple of dictionaries in the form ``(male, female)``, with each document being a key and the proportion
of pronouns that are the subject as the values.

To determine what portion of subject pronouns in each document of the corpus is of a specified gender, call:

    >>> subject_pronouns_gender_comparison(my_corpus,'female')

You can also pass in ``'male'`` as a parameter in order to see the proportion of male pronouns.


Instance Distance Analysis
==========================

When analyzing documents, it's often interesting to see not only how many times a word appears in a corpus, but also
how far apart those instances of the word are.

Document Analysis
-----------------

The first thing you need to do is create a document that you would like to analyze.

    >>> from gender_analysis import Document
    >>> my_document = Document(document_metadata)

To get the distance between all of the occurrences of a certain word in a document, call:

    >>> from gender_analysis.instance_distance import *
    >>> document = my_corpus[0]  # Gets first document from corpus
    >>> word = 'word'  # Can be anything
    >>> instance_dist(document, word)

Similarly, a list of words can be searched for by using ``words_instance_dist(document, words)``.

To get the distances between male words in a document, you can call

    >>> male_instance_dist(document)

Similarly, you can find female distances with

    >>> female_instance_dist(document)

Corpus Analysis
---------------

While seeing results for individual documents can be very interesting, it tends to be tedious when you are trying to
iterate over hundreds or thousands of documents. These functions allow you to perform these checks on large corpora,
and even graph the results.

To get a dictionary mapping each document object to an array of 3 lists containing the median, mean, min, and max
distances between male and female pronouns instances and another list containing the difference between the male
and female values, call:

    >>> results = run_distance_analysis(my_corpus)

Because this function might take some time to run, you can save the results as a pickle with

    >>> store_raw_results(results, 'path/to/pickle_file.pgz')

To retrieve a particular result by author gender, call:

    >>> results_by_author_gender(results, metric)

Where ``metric`` is ``'mean'``, ``'median'``, or ``'mode'``.

Additionally, you can retrieve a given number of the top instance distances with

    >>> results_by_highest_distances(results, num)

To create a bar-and-whisker graph of your results, all you need to call is:

    >>> box_plots(results, 'pastel', x='N/A')

.. note::
    ``'pastel'`` can be replaced with any seaborn palette










