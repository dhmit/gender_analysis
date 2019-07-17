==============
User's Guide
==============

If you have not yet installed the package, make sure to look at the Installation instructions:


.. toctree::
    :maxdepth: 2

    installation

This is a work in progress and will be updated as we prepare more tutorials.

Document Class
==============

The document class is the base-level unit of metadata storage. It is responsible for keeping track of the metadata,
text, and file location of a single document. Our analysis will work best if your text is split into documents such
that each document has its own author(s), publication date, etc. For example, a collection of Tweets should be split
by individual Tweet.

Initialization
--------------

``Document`` initialization will typically be handled by the ``Corpus`` object, but, in case we want to create an
individual Document, we call the ``Document.__init__`` function with a metadata dictionary::
    >>> my_doc = Document({'filename': 'example.txt', 'filepath': 'path/to/example.txt',
    ...                     'author': 'John Doe', 'date': 2019})

Each key in the metadata dictionary becomes an attribute of the Document object. Only the ``'filename'`` attribute is
required and its value must end in '.txt'. However, the metadata should also have a ``'filepath'`` attribute that
points to the file of the document's text, as ``Document`` loads the text from this location. Other metadata fields may
be helpful for more powerful analyses if working with a corpus, but are not required.

Operations
----------

If we want to change the metadata of a document after initialization, we can call::
    >>> my_doc.update_metadata({'author': 'Jane Doe', 'date': 2018})

With a Document object, you can perform a variety of analyses within a single document:
    - Word count: ``my_doc.get_count_words()``
    - Distance between occurrences of words: ``my_doc.words_instance_dist()``
    - Adjectives associated with a certain gender: ``my_doc.find_male_adj()`` and ``my_doc.find_female_adj()``

