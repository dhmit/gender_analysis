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
In the [`gender_analysis/gender_analysis`](https://github.com/dhmit/gender_analysis/tree/master/gender_analysis) folder, there are three folders: 

1. [`analysis`](https://github.com/dhmit/gender_analysis/tree/master/gender_analysis/analysis) — programming files focused on textual analysis and research write-ups, including data visualizations and conclusions
2. [`pickle_data`](https://github.com/dhmit/gender_analysis/tree/master/gender_analysis/pickle_data) — pickled data for various analyses to avoid running time-consuming computation
3. [`testing`](https://github.com/dhmit/gender_analysis/tree/master/gender_analysis/testing) — files for code tests

For a user who’ll need some readily available methods for analyzing documents, the files you’ll most likely want are [`corpus.py`](https://github.com/dhmit/gender_analysis/blob/master/gender_analysis/corpus.py) and [`document.py`](https://github.com/dhmit/gender_analysis/blob/master/gender_analysis/document.py). These include methods used for loading and analyzing texts from the corpora. 

*This document was prepared by the MIT Digital Humanities Lab.*

Copyright © 2018-2019, [MIT Programs in Digital Humanities](https://digitalhumanities.mit.edu/). Released under the [BSD license](https://github.com/dhmit/gender_novels/blob/master/LICENSE).
Some included texts might not be out of copyright in all jurisdictions of the world.
