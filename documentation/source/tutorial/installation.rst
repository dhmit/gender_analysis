============
Installation
============

Before you can get started with the Gender Analysis Toolkit, you will need to make sure that you have
Python 3.6 or greater installed on your computer along with the toolkit and its requirements.

Installation on Windows
-----------------------

Most of the installation process will happen through Windows Powershell, or another terminal program of your choice.
To ensure that everything installs properly, ensure that you open your command-line tool in Administrator mode.

Installing Python
*****************

To use this toolkit, Windows users should ensure they are using a supported version of Python. To check what version
of Python is on your computer (if at all), run the following command::
    python --version

If you already have Python installed on your computer, your terminal should return something similar to ``Python 3.6.2``.
If your version is older than ``Python 3.6.x``, you must download a newer version to utilize the Gender Analysis Toolkit.
You can download the latest version from `Python's Website <http://www.python.org/download/>`_.

For the rest of the installation, we will be using pip to install the Gender Analysis Package. To ensure that you are
using the latest version, make sure to use the following command::
    python -m pip install -U pip

This should present progress bars and other information while pip is being updated. Once the process is finished, you
can move on to the next step.

Installing the Toolkit
**********************

After Python is installed on your computer, you can now install the toolkit. We have uploaded the package to the
Python Package Index (PyPI), so it can be installed through the use of pip. If you would like to set up a `virtual
environment <https://docs.python.org/3/library/venv.html>`_, you should do this prior to installing the module.

To install the Gender Analysis Toolkit and its packages, you only need to enter::
    pip install gender_analysis

After this has finished running, you should have the package installed. From here, we recommend taking a look at our
:doc:`User's Guide <tutorial>` to learn the basics, or take a look at the `Reference Guide <https://gender-analysis.rtfd.io>`_ to
look at the Documentation of the toolkit.



Installation on MacOS/Linux
---------------------------
