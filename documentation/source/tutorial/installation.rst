============
Installation
============

Before you can get started with the Gender Analysis Toolkit, you will need to make sure that you have
Python 3.6 or greater installed on your computer along with the toolkit and its requirements.

Installation on Windows
-----------------------

Most of the installation process will happen through Windows Powershell, or another terminal program of your choice.
To ensure that everything installs properly, make sure to open your command-line tool in Administrator mode.

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
Python Package Index (PyPI), so it can be installed through the use of pip.

.. note::
    If you would like to set up a `virtual environment <https://docs.python.org/3/library/venv.html>`_, you should
    do this prior to installing the module.

To install the Gender Analysis Toolkit and its packages, you only need to enter::

    pip install gender_analysis

.. warning::
    The Gender Analysis Toolkit is not yet on PyPI, but will be soon! This will work in the very near future, but in the
    meantime you can checkout and download the test version on `PyPI <https://test.pypi.org/project/gender-analysis/0.1.0/>`_

After this has finished running, you should have the package installed. From here, we recommend taking a look at our
:doc:`User's Guide <tutorial>` to learn the basics, or take a look at the `Reference Guide <https://gender-analysis.rtfd.io>`_ to
look at the documentation for the toolkit.



Installation on MacOS/Linux
---------------------------

The installation will happen through your operating system's terminal, which will allow us to perform tasks
such as look at Python's version and install the package requirements for the toolkit.

Installing Python
*****************

In order to use the toolkit, you need to make sure that you are using Python version 3.6 or newer. To check to see what
version is installed on your computer, simply run the command::

    $ python --version

.. note::
    If you are using macOS, you may need to try the command::

        $ python3 --version

    in order to see the proper version.

You should see the terminal report a statement similar to ``Python 3.6.2``. If the version number is older than ``3.6``,
or you do not have Python installed on your system, you will need to download the latest version from the `Python
website <http://www.python.org/download/>`_.

Once Python is set up with the proper version, the next step is to make sure that your version of pip
is up to date::

    $ pip install -U pip

.. note::
    If you previously had to run the command::

        $ python3 --version

    you should now use::

        $ pip3 install -U pip

After ensuring that pip is using the latest version, you can move on to the next step.

Installing the Toolkit
**********************

After Python is installed and updated to the correct version, you are able to install the toolkit. Because the package
is uploaded to the Python Package Index (PyPI), you are able to do this with one command through the use of pip.

.. note::
    If you would like to set up a `virtual environment <https://docs.python.org/3/library/venv.html>`_, you should
    do this prior to installing the module.

To install the Gender Analysis Toolkit and its packages, you only need
to enter::

    $ pip install gender_analysis

.. note::
    If you are using macOS or some versions of Linux, it may be necessary to use the command::

        $ pip3 install gender_analysis

.. warning::
    The Gender Analysis Toolkit is not yet on PyPI, but will be soon! This will work in the very near future, but in the
    meantime you can checkout and download the test version on `PyPI <https://test.pypi.org/project/gender-analysis/0.1.0/>`_

After this has finished running, you should have the package installed. From here, we recommend taking a look at our
:doc:`User's Guide <tutorial>` to learn the basics, or take a look at the
`Reference Guide <https://gender-analysis.rtfd.io>`_ to look at the documentation for the toolkit.
