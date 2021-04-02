================
Quickstart Guide
================

If you have not yet installed the package, make sure to look at the Installation instructions:


.. toctree::
    :maxdepth: 2

    installation

.. note::
    This is a work in progress and will be updated as we prepare and expand tutorials. Check back soon!

Initialization
==============

Let's start off by preparing to load up our sample Corpus, which includes 99 nineteenth-century novels written in
English. We'll need to import the :ref:`Corpus <corpus_analysis:corpus module>` class and the path to a corpus:

.. code-block:: python

    >>> from corpus_analysis.corpus import Corpus
    >>> from gender_analysis.testing.common import TEST_CORPUS_PATH


... and then we can generate our first :ref:`Corpus <corpus_analysis:corpus module>` object:

.. code-block:: python

    >>> sample_corpus = Corpus(TEST_CORPUS_PATH)
    >>> len(sample_corpus)
    99

Now we have a ``Corpus`` object, ``sample_corpus``, made up of 99 ``Document``\ s. However, if we
want to do serious analysis, we're going to want to add metadata like ``author``, ``date``, ``filepath``, and ``title``.
Fortunately, our sample corpus comes bundled with a metadata file, ``sample_novels.csv``, so we just need to grab that and
re-initialize our corpus with a second parameter, csv_path, that points to it:

.. code-block:: python

    >>> from gender_analysis.testing.common import LARGE_TEST_CORPUS_CSV
    >>> meta_corpus = Corpus(TEST_CORPUS_PATH, csv_path = LARGE_TEST_CORPUS_CSV)
    >>> len(meta_corpus)
    99

Now that we have a Corpus initialized, let's take a look at some of the different functions the Toolkit can perform.
Start by making a Document object for one novel from the collection, Bram Stoker's `Dracula`. We'll do that by using the
:ref:`Corpus.get_document <get-document>` function, which fetches a single document using two parameters, ``metadata_field``
and ``field_argument``:

.. code-block:: python

    >>> dracula = meta_corpus.get_document('title', 'Dracula')
    >>> dracula.title
    'Dracula'
    >>> dracula.author
    'Stoker, Bram'

Analyzing a Document
====================

We can use the :ref:`Document <corpus_analysis:document module>` class's methods to perform a variety of simple analytic
functions. Let's start by looking at how the word "sleep" shows up in the novel. We'll use
:ref:`get_count_of_word <get-count-of-word>` to see how many times it occurs,
:ref:`get_word_freq <get-word-freq>` to find the frequency with which it occurs compared to all
words in the document, and :ref:`words_associated <words-associated>` to see what words come up immediately after it:

.. code-block:: python

    >>> from gender_analysis import document
    >>> dracula.get_count_of_word('sleep')
    179
    >>> dracula.get_word_freq('sleep')
    0.0011139253109967453
    >>> sleep_associations = dracula.words_associated('sleep')
    >>> sleep_associations
    Counter({'and': 18, 'i': 9, 'but': 8, 'for': 7, ...

This is a little messy, but since words_associated returns a ``Counter`` object, we can just get the top few results
with the ``most_common`` method. Let’s look at the 10 most common words associated with “sleep” in `Dracula`:

.. code-block:: python

    >>> sleep_associations.most_common(10)
    [('and', 18), ('i', 9), ('but', 8), ('for', 7), ('the', 6), ('in', 6), ('well', 5), ('as', 5), ('on', 5), ('so', 5)]

Not terribly helpful, right? This is a common problem with natural language processing. What we have here is a list of
"stopwords," which are common words like articles and prepositions. While those are sometimes useful, in an analysis
like this the stopwords are mostly noise. Fortunately, we can get rid of them pretty easily. We'll start by grabbing a
list of stopwords sourced from NLTK:

.. code-block:: python

    >>> from gender_analysis.common import SWORDS_ENG as swords_eng
    >>> for word in list(sleep_associations):
    >>> 	if word in swords_eng:
    >>>		del sleep_associations[word]
    >>> sleep_associations.most_common(10)
    [('well', 5), ('tonight', 3), ('without', 2), ('comes', 1), ('yesterday', 1), ('unwisely', 1), ('brings', 1), ('wind', 1), ('moaning', 1), ('began', 1)]

The :ref:`get_word_windows <get-word-windows>` method is a more powerful version of ``words_associated`` that looks for
``window_size`` (in the example below, 4) words around the search term and returns a ``Counter``. Let’s look at the five
most common words that appear around “sleep”

.. code-block:: python

    >>> sleep_windows = dracula.get_word_windows('sleep', 4)
    >>> sleep_windows.most_common(5)
    [('i', 58), ('to', 58), ('and', 58), ('the', 48), ('her', 37)]

And now we can write a loop that'll remove stopwords from our ``Counter``:

.. code-block:: python

    >>> for word in list(sleep_windows):
    >>> 	if word in swords_eng:
    >>>		del sleep_windows[word]
    >>> sleep_windows.most_common(10)
    [('go', 10), ('shall', 9), ('could', 9), ('tonight', 8), ('fear', 8), ('even', 7), ('still', 7), ('well', 6), ('though', 6), ('must', 6)]

Most of these make a lot of sense - "go" and "tonight," for example - but "fear" is a bit of a surprise and worthy of
further analysis!

The ``Document`` class also includes a part-of-speech tagger, :ref:`get_part_of_speech_tags <get-pos>`, which returns a
list of tuples in the form ``(term, speech_tag)``. This function can take awhile, especially if you’re using it on more
than one ``Document`` at a time. Let's do this and grab a random sentence from the results to see what it looks like.

.. code-block:: python

    >>> dracula_pos = dracula.get_part_of_speech_tags()
    >>> dracula_pos[500:510]
    [('get', 'VB'), ('it', 'PRP'), ('anywhere', 'RB'), ('along', 'IN'), ('the', 'DT'), ('Carpathians', 'NNPS'), ('.', '.'), ('I', 'PRP'), ('found', 'VBD'), ('my', 'PRP$')]

Of course, this is the **gender**\_analysis toolkit — so let’s do some gendered analysis!

The Gender Analysis Toolkit uses a flexible object-oriented framework to represent genders as a collection of pronouns
and other 'identifiers,' like character names. It comes pre-loaded with three Gender objects: ``MALE``, ``FEMALE``,
and ``NONBINARY``. Let's start by taking a look at those.

.. code-block:: python

    >>> from gender_analysis.common import MALE as male, FEMALE as female, NONBINARY as nonbinary
    >>> female.pronouns
    {'herself', 'hers', 'her', 'she'}
    >>> nonbinary.pronouns
    {'them', 'themself', 'they', 'theirs'}

A major feature of the Gender Analysis Toolkit is the ability to create your own gender objects. For more on that,
see :ref:`tutorial/tutorial:Defining Pronouns and Genders`.

Defining Pronouns and Genders
=============================
