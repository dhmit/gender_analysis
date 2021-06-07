================
Quickstart Guide
================

If you have not yet installed the package, make sure to look at the Installation instructions:


.. toctree::
    :maxdepth: 2

    installation

.. note::
    This is a work in progress and will be updated as we prepare and expand tutorials. Check back soon!

**************
Initialization
**************

Let's start off by preparing to load up our sample Corpus, which includes 99 nineteenth-century novels written in
English. We'll need to import the :ref:`Corpus <gender_analysis.text:corpus module>` class and the path to a corpus:

.. code-block:: python

    >>> from gender_analysis.text.corpus import Corpus
    >>> from gender_analysis.testing.common import TEST_CORPUS_PATH


... and then we can generate our first :ref:`Corpus <gender_analysis.text:corpus module>` object:

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

********************
Analyzing a Document
********************

We can use the :ref:`Document <gender_analysis.text:document module>` class's methods to perform a variety of simple analytic
functions. Let's start by looking at how the word "sleep" shows up in the novel. We'll use
:ref:`get_count_of_word <get-count-of-word>` to see how many times it occurs,
:ref:`get_word_freq <get-word-freq>` to find the frequency with which it occurs compared to all
words in the document, and :ref:`words_associated <words-associated>` to see what words come up immediately after it:

.. code-block:: python

    >>> from gender_analysis.text.document import Document
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

    >>> from gender_analysis.text.common import SWORDS_ENG as swords_eng
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

    >>> from gender_analysis.gender.common import MALE as male, FEMALE as female, NONBINARY as nonbinary
    >>> female.pronouns
    {'herself', 'hers', 'her', 'she'}
    >>> nonbinary.pronouns
    {'them', 'themself', 'they', 'theirs'}

A major feature of the Gender Analysis Toolkit is the ability to create your own gender objects. For more on that,
see :ref:`tutorial/tutorial:Defining Pronouns and Genders`.

Proximity Analysis
=======================

We can use the class ``GenderProximityAnalyzer`` in :ref:`gender_adjective <gender_analysis.analysis:proximity module>` to see what
words are associated with different gendered word sets. Let's use the
``by_gender`` method, which needs a document and a gender to look for. This analyzer accepts either a ``Corpus`` instance or the filepaths to relevant texts and metadata.

.. code-block:: python

    >>> from gender_analysis.analysis.proximity import GenderProximityAnalyzer
    >>> from gender_analysis.testing.common import TEST_CORPUS_PATH, TUTORIAL_TEST_CORPUS_CSV
    >>> analyzer = GenderProximityAnalyzer(file_path=TEST_CORPUS_PATH, csv_path=TUTORIAL_TEST_CORPUS_CSV, genders=[female, male, nonbinary])
    >>> analyzer.by_gender().get('Male').get('dead')
    7

This function returns a dictionary of three gender dictionaries, each of which contains a list of every adjective that
shows up around the gender and how often it shows up. It's challenging to extract information in its current form,
though, so we'll pass in the optional flags `diff=True` and `sort=True`, which figure out which words
are disproportionately associated with a specific gender.

.. code-block:: python

    >>> analyzer.by_gender(diff=True)
    {
        'Female': [('dead', 8), ('beautiful', 8), ('sweet', 8), ('sleep', 8), ('asleep', 7), ('hypnotic', 6), ('anæmic', 5), ('unconscious', 5), ('whilst', 5), ('cold', 4)],
        'Male': [('good', 34), ('first', 29), ('old', 24), ('great', 22), ('poor', 21), ('new', 18), ('strong', 16), ('much', 15), ('last', 14), ('right', 12)],
        'Nonbinary': [('clumsy', 3), ('fash', 2), ('warnt', 2), ('afeared', 2), ('difficult', 2), ('proper', 2), ('former', 2), ('thirsty', 2), ('patient', 2), ('eleventh', 1)]
    }


The results here are ``(word, count)`` pairs, where the count is the number of times the word occurs near a given
gender's pronoun minus the how often the word occurs near every other gender's pronoun. So, "dead," for example, occurs
17 times around female pronouns minus 7 times around male pronouns and 2 times around nonbinary pronouns (which, here,
is a 'they/them' series) for a result of ``('dead', 8)``.

The differences here is striking! Of the top 10 words predominantly associated with female pronouns, 7 of them are
somnolent and relate to either sleep, weakness, or death. Contrast that with the 'masculine' adjectives - "good,"
"great," "strong," and "right" - and you have the beginnings of a hypothesis of the different kinds of scenarios in
which Stoker puts his characters.

This sort of data has a clear thrust to it and can be used as evidence to support a hypothesis about the text in
question, or as the grounds of a new hypothesis. One might investigate further by opening the text file and looking up
words of particular interest, like "hypnotic."

The ``GenderProximityAnalyzer`` accepts NLTK tags to determine which part of speech to include in the results set.
You can examine all possible NLTK tags by calling ``GenderProximityAnalyzer.list_nltk_tags()``.

Frequency Analysis
==================

We can also study how often gendered pronouns show up in their documents. For instance, we can find the comparative
frequencies of a list of genders:

.. code-block:: python

    >>> from gender_analysis.analysis.frequency import GenderFrequencyAnalyzer
    >>> from gender_analysis.gender.common import MALE as male, FEMALE as female, NONBINARY as nonbinary
    >>> analyzer = GenderFrequencyAnalyzer(file_path=TEST_CORPUS_PATH, csv_path=TUTORIAL_TEST_CORPUS_CSV, genders=[female, male, nonbinary])
    >>> analyzer.by_gender(format_by='relative', group_by='aggregate')
    {
        'Female': 0.2411177139967126,
        'Male': 0.6416740422303705,
        'Nonbinary': 0.11720824377291693
    }

This means that, of pronoun usages between male, female, and nonbinary (they/them) identifiers, he/him pronouns come up
roughly 65% of the time, she/her 24% of the time, and they/them 12%.

We can get pass in optional keyword arguments to format and shape the data as needed. Above we've provided
``format_by='relative'`` and ``group_by='aggregate'``. Changing ``group_by`` from ``'aggregate'`` to ``'label'``
provides us additional helpful context:

.. code-block:: python

    >>> analyzer.by_gender(format_by='relative', group_by='label')
    {
        'Female': {'subject': 0.1024149702870148, 'object': 0.13351877607788595, 'other': 0.00518396763181186},
        'Male': {'subject': 0.32127955493741306, 'object': 0.11948413200151727, 'other': 0.20091035529144013},
        'Nonbinary': {'subject': 0.0584144645340751, 'object': 0.058793779238841826, 'other': 0.0}
    }


These results largely square with our part-of-speech results; female characters, who tend to be described as asleep or
dead, are the subjects of sentences 10% of the time and the objects 13%; male characters, conversely, whose verbs and
adjectives are more active, are subjects 32% of the time and objects 12%. They/them occur about evenly across the two.

Many of the more powerful methods of these analyzers work better when they're undertaken with more than one text,
so let's give that a whirl.

******************
Analyzing a Corpus
******************

Most of the analytic functions in the Gender Analysis toolkit operate on corpora as well as on ``Documents``. For
example, we can run a full gendered adjective analysis on an entire corpus with the ``GenderProximityAnalyzer``,
which takes as arguments a ``Corpus`` object or relevant filepaths and a list of ``Gender`` objects.
For this example, let's use our small test corpus of 10 Documents. We'll not pass explicit ``Gender`` instances
into the analyzer and instead let it default to the the ``BINARY_GROUP`` list of ``Gender`` objects,
which includes ``MALE`` and ``FEMALE``. We'll also allow the analyzer to default to searching for adjectives.

.. code-block:: python

    >>> from gender_analysis.testing.common import TEST_CORPUS_PATH, SMALL_TEST_CORPUS_CSV
    >>> from gender_analysis.analysis.proximity import GenderProximityAnalyzer
    >>> analyzer = GenderProximityAnalyzer(file_path=TEST_CORPUS_PATH, csv_path=SMALL_TEST_CORPUS_CSV)
    >>> analyzer.by_document()

The ``by_document`` methods returns a dictionary of result dictionaries, with one for each
``Document`` input to the analyzer. It may take a little while to run. Each ``Document`` result contains one Dictionary
for each ``Gender`` provided, with a list of adjectives and their frequencies across the ``Document``. We can access the
values for any particular ``Document`` by using that ``Document`` .label (which is the same
as the filepath minus the extension) as a key in our result dictionary.

However, we're now working at the ``Corpus`` level. Instead of looking at individual ``Document`` results, let's
explore some of the other methods available to us.

.. code-block:: python

    >>> analyzer.by_gender()

These dictionaries are a bit hard to handle, but we can print out some reasonable results with display functions:

.. code-block:: python

    >>> analyzer.by_gender(sort=True)
    {
        'Female': [('little', 673), ('good', 302), ('much', 284), ('great', 203), ('first', 201), ('old', 172), ('last', 171), ('young', 144), ('happy', 142), ('many', 137)],
        'Male': [('little', 480), ('good', 336), ('much', 240), ('great', 211), ('old', 193), ('young', 156), ('best', 150), ('first', 145), ('last', 136), ('many', 125)]
    }

These are the top ten adjectives associated with each gender across our test corpus! Unfortunately, as you can see, it's
not terribly informative; the lists overlap quite a bit. If we want to see instead what adjectives are
disproportionately associated with each gender, we can instead pass in the `diff=True` optional flag:

.. code-block:: python

    >>> analyzer.by_gender(sort=True, diff=True)
    {
        'Female': [('little', 193), ('mrs', 91), ('miss', 75), ('happy', 65), ('first', 56), ('least', 54), ('long', 45), ('amy', 45), ('much', 44), ('next', 36)],
        'Male': [('mr', 41), ('good', 34), ('rough', 26), ('nat', 24), ('old', 21), ('fellow', 20), ('best', 18), ('french', 16), ('english', 16), ('handsome', 15)]
    }

Gender Frequencies Across Corpora
=================================

Before we saw how to perform frequency analysis of genders across a single document, but these functions are available
for corpora as well. Many of the functions that act on a corpus return a dictionary mapping each of the documents to
their individual results, which allows users to examine the corpus results on a document-by-document level. These are
stored with the ``Document`` instance's ``.label`` preoperty as keys to the dictionary.

*****************************
Defining Pronouns and Genders
*****************************

While the Gender Analysis Toolkit provides representations for three common gender identities and pronoun series, that
is by no means the complete list. If you wish to perform an analysis using a pronoun series or gender identity that is
not provided by the toolkit, it’s actually pretty easy!

.. _declaring-new-pronounseries-objects:

Declaring New :ref:`PronounSeries <gender_analysis.gender:pronouns module>` Objects
===================================================================================

For the purposes of this toolkit, a :ref:`PronounSeries <gender_analysis.gender:pronouns module>` is defined
by four things:

- An identifier (basically the name of the pronoun series)
- A collection of pronouns (such as “she”, “her”, “hers”, “herself”, etc.)
- A subject pronoun
- An object pronoun

Here's an example using the Xe series:

.. _xe_series:

.. code-block:: python

    >>> from gender_analysis.gender.pronouns import PronounSeries
    >>> xe_pronouns = {'xe', 'xem', 'xyr', 'xyrs', 'xemself'}
    >>> xe_series = PronounSeries('Xe', xe_pronouns, 'xe', 'xem')

Let’s break this down. Our use of ``'Xe'`` is just a placeholder for the series - this could be anything that you would
like to use to identify the pronoun collection. ``xe_pronouns`` is basically just a declaration of all of the words that
we want to include as part of that pronoun series. While these words could theoretically be anything, it is important
that they be pronouns - if you want names to be associated with a pronoun series, check out the
:ref:`Gender <declaring-new-gender-objects>` tutorial.

The last two terms define what the subject and object pronouns for the series are, respectively. These do not have to be
included in the collection of words that you provide the object, although they will be counted as part of the series
regardless.

The pronouns associated with a pronoun series can be found by calling the following:

.. code-block:: python

    >>> xe_series.pronouns
    {'xe', 'xem', 'xyr', 'xyrs', 'xemself'}

While this is a handy tool, there is often not much need to call on the field directly. The ``PronounSeries`` object
allows for simple containment checks:

.. code-block:: python

    >>> 'Xyrs' in xe_series
    True
    >>> 'her' in xe_series
    False

Note that these checks are not case sensitive. Along with these containment checks, the object also has built-in
iteration support:

.. code-block:: python

    >>> for pronoun in xe_series:
    >>>     print(pronoun)
    'xem'
    'xe'
    'xyrs'
    'xyr'
    'xemself'

This is useful for performing some analysis using all of the pronouns in a series, but it is worth mentioning that all
pronouns returned are in lower case. Additionally, there is no guaranteed order to which pronouns are given at any given
time (for instance, the first pronoun in the loop could be "xe" one time, and then "xyr" the next). If order is
something that matters, consider creating a sorted list of the pronouns as follows:

.. code-block:: python

    >>> sorted_pronouns = sorted(xe_series)
    >>> sorted_pronouns
    ['xe', 'xem', 'xemself', 'xyr', 'xyrs']

.. _declaring-new-gender-objects:

Declaring New :ref:`Gender <gender_analysis.gender:gender module>` Objects
==========================================================================

While the :ref:`PronounSeries <declaring-new-pronounseries-objects>` object opens the door for a lot of potential
analyses with the toolkit, there’s an even larger abstraction that could be made. The main tool used for analysis in the
toolkit is the :ref:`Gender <gender_analysis.gender:gender module>` object, which is a way of merging different
``PronounSeries`` and names into one compact bundle.

We can create a new agender ``Gender`` object as follows, using the ``xe_series`` ``PronounSeries`` we
:ref:`defined earlier <xe_series>`:

.. code-block:: python

    >>> from gender_analysis.gender.gender import Gender
    >>> agender = Gender('Agender', xe_series)

And that’s it! You’ve now defined a new ``Gender`` object, and you can plug this into any of the functions just as you
would the gender definitions the package defines already.

However, you’ll notice that we haven’t actually done any work defining names in the object, which is something that was
mentioned in the introduction to the section. There is an additional field that you can supply the ``Gender`` object,
which allows you to list specific names that will be associated with the gender. For instance, let’s say that you have a
list of character names that identify as agender, and would like the analysis functions to recognize those as well. We
could do so as follows:

.. code-block:: python

    >>> agender_names = ['Taylor', 'Alex', 'Jessie']
    >>> agender = Gender('Agender', xe_series, agender_names)

An additional thing of note is that not all people that identify as agender use the “xe” pronoun series. Some use
“they”, others use “ze/hir”, and there are even others that use “he” or “she” (this isn’t even a comprehensive list!).
Luckily, the ``Gender`` class has a way of handling this variety as well. Instead of passing in a single
``PronounSeries`` object to the ``Gender``, we can pass a collection of them, like so:

.. code-block:: python

    >>> ze_hir = ['ze', 'hir', 'hirs', 'hirself']
    >>> ze_series = PronounSeries('Ze', ze_hir, 'ze', 'hir')
    >>> agender = Gender('Agender', {xe_series, ze_series}, agender_names)

With this object now defined with several new terms, we can extract a bunch of useful information out of it. In addition
to being able to use this in any of the functions in the toolkit that ask for a ``Gender``, we can also perform some
useful analyses with the objects themselves. For instance, we can retrieve a set of all of the words associated with the
gender through the ``identifiers`` field:

.. code-block:: python

    >>> agender.identifiers
    {'xe', 'xem', 'xemself', 'xyr', 'xyrs', 'ze', 'hir', 'hirs', 'hirself', 'Taylor', 'Alex', 'Jessie'}

If we only wanted to retrieve a set of the names or pronouns associated with the gender, we could do so with the
``names`` or ``pronouns`` properties:

.. code-block:: python

    >>> agender.names
    {'Taylor', 'Alex', 'Jessie'}
    >>> agender.pronouns
    {'xe', 'xem', 'xemself', 'xyr', 'xyrs', 'ze', 'hir', 'hirs', 'hirself'}

Finally, we can retrieve the subject or object pronouns associated with the gender by calling ``subj`` or ``obj``,
respectively.

.. code-block:: python

    >>> agender.subj
    {'xe', 'ze'}
    >>> agender.obj
    {'xem', 'hir'}

**************
Visualizations
**************

We can also visualize some of the different properties of our corpora to understand them a little bit better.

To get started, let’s initialize a new ``Corpus``, if you don’t have one initialized already:

.. code-block:: python

    >>> from gender_analysis.analysis.metadata_visualizations import *
    >>> from gender_analysis.text.corpus import Corpus
    >>> from gender_analysis.testing.common import TEST_CORPUS_PATH, LARGE_TEST_CORPUS_CSV
    >>> meta_corpus = Corpus(TEST_CORPUS_PATH, csv_path = LARGE_TEST_CORPUS_CSV)

With this corpus in mind, we can now begin to visualize the metadata that go along with it so that we have a better
understanding of our other analyses. We have the ability to visualize 3 main metadata attributes of the corpus:

- publication years
- the publication countries
- author genders

If we want to visualize the author genders of the different ``Documents`` in the corpus, we can run
:ref:`plot_gender_breakdown <plot_gender_breakdown>` which has one required parameter, ``corpus``, which is the corpus
that we want to visualize. We also have two optional parameters that allow us to specify an output directory and a
special name to call the file. The default output directory will create a visualization directory in your current
working directory, if one is not there already. The neat part of specifying the output directory is that you don’t have
to create it! All the directories that don’t exist yet will be created for you. The default name of the plot that will
be saved is a descriptive name of what the plot represents + the name of the corpus used to create the plot.

.. code-block:: python

    >>> plot_gender_breakdown(meta_corpus)

Running this line will produce a graph that is saved to the default output directory.

We can also plot the other metadata that was discussed above by running :ref:`plot_pubcountries <plot_pubcountries>` or
:ref:`plot_pubyears <plot_pubyears>`, which use the same parameters as described above.

There is an additional plot function, :ref:`plot_metadata_pie <plot_metadata_pie>`, that allows us to visualize the
metadata regarding author gender and publication country, two important factors of analyses, in regards to how many
documents in the corpus contain that metadata information:

.. code-block:: python

    >>> plot_metadata_pie(meta_corpus)

This example is not as appealing or useful for this corpus as the other visualization functions because our test corpus
is well documented and contains all the metadata that we might want or need for analyses. When using the metadata
visualizations on your own handcrafted corpus, this function can help you tell what data you might want to try and
collect more information on.

Finally, :ref:`create_corpus_summary_visualizations <create_corpus_summary_visualizations>` creates each of the above
visualizations using the default naming pattern described earlier in this section and saves all of them to the directory
of your choosing.

