"""
Setup script for gender_analysis installation
"""
import sys
import setuptools
from setuptools.command.develop import develop
from setuptools.command.install import install


def download_nltk_packages():
    import nltk
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')


class PostDevelopCommand(develop):
    """ Post-installation for development mode. """
    def run(self):
        download_nltk_packages()
        develop.run(self)


class PostInstallCommand(install):
    """ Post-installation for installation mode. """
    def run(self):
        download_nltk_packages()
        install.run(self)


with open('requirements.txt') as f:
    REQUIRED_PACKAGES = f.read().strip().split('\n')

with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

# Check if Python 3.6 or 3.7 is installed.
PYTHON_VERSION = sys.version_info
if PYTHON_VERSION.major < 3 or (PYTHON_VERSION.major == 3 and PYTHON_VERSION.minor < 6):
    ERR = ('gender_analysis only supports Python Versions 3.6 and higher'
           + 'Your current Python version is {0}.{1}.'.format(
               str(PYTHON_VERSION.major),
               str(PYTHON_VERSION.minor)
           ))
    sys.exit(ERR)

setuptools.setup(
    name="gender-analysis",
    version="0.1.0",
    author="MIT Digital Humanities Lab",
    author_email="digitalhumanities@mit.edu",
    description="Toolkit for analyzing gender in documents",
    install_requires=REQUIRED_PACKAGES,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/dhmit/gender_analysis",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    cmdclass={
        'develop': PostDevelopCommand,
        'install': PostInstallCommand,
    },
)
