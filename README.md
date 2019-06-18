# Gender Analysis Toolkit
A toolkit for analyzing of gender and gender relations across documents

##  History

This toolkit is an extension of work we began in Fall 2018 analyzing gender in English language novels.

The results of that research is available at [The Gender Novels Project](http://gendernovels.digitalhumanitesmit.org) website, and our research code is available at https://github.com/dhmit/gender_novels/

*This MIT Digital Humanities Lab project is part of the [MIT/SHASS Programs in Digital Humanities](https://digitalhumanities.mit.edu/) funded by the [Mellon Foundation](https://www.mellon.org/).*

## Usage
To use our tools or contribute to the project, please view our guide to contributing, `CONTRIBUTING.md`. It includes information on how to install the tools we used as well as style guidelines for adding code. We are open to contributions and would love to see other people’s ideas, thoughts, and additions to this project, so feel free to leave comments or make a pull request!

## Navigating Gender / Novels

For anybody who wants to use our code, here’s a little outline of where everything is.
In the [`gender_novels/gender_novels`](https://github.com/dhmit/gender_novels/tree/master/gender_novels) folder, there are six folders: 

1. [`analysis`](https://github.com/dhmit/gender_novels/tree/master/gender_novels/analysis) — programming files focused on textual analysis and research write-ups, including data visualizations and conclusions
2. [`corpora`](https://github.com/dhmit/gender_novels/tree/master/gender_novels/corpora) — metadata information on each book (including author, title, publication year, etc.), including sample data sets and instructions for generating a [Gutenberg mirror](https://github.com/dhmit/gender_novels/tree/master/gender_novels/corpora/gutenberg_mirror_sample)
3. [`pickle_data`](https://github.com/dhmit/gender_novels/tree/master/gender_novels/pickle_data) — pickled data for various analyses to avoid running time-consuming computation
4. [`testing`](https://github.com/dhmit/gender_novels/tree/master/gender_novels/testing) — files for code tests
5. [`tutorials`](https://github.com/dhmit/gender_novels/tree/master/gender_novels/tutorials) — tutorials used by the lab to learn about various technical subjects needed to complete this project

For a user who’ll need some readily available methods for analyzing documents, the files you’ll most likely want are [`corpus.py`](https://github.com/dhmit/gender_novels/blob/master/gender_novels/corpus.py) and [`novel.py`](https://github.com/dhmit/gender_novels/blob/master/gender_novels/novel.py). These include methods used for loading and analyzing texts from the corpora. If you’d like to generate your own corpus rather than use the one provided in the repo, you’ll want to use [`corpus_gen.py`](https://github.com/dhmit/gender_novels/blob/master/gender_novels/corpus_gen.py). If you’d only like a specific part of our corpus, the method `get_subcorpus()` may be useful.  

*This document was prepared by the MIT Digital Humanities Lab.*

Copyright © 2018-2019, [MIT Programs in Digital Humanities](https://digitalhumanities.mit.edu/). Released under the [BSD license](https://github.com/dhmit/gender_novels/blob/master/LICENSE).
Some included texts might not be out of copyright in all jurisdictions of the world.
