import csv
import os
import re
import requests
from pathlib import Path
from shutil import copyfile

import gender_guesser.detector as gender_guesser
import pywikibot
from gutenberg.acquire import get_metadata_cache
from gutenberg.cleanup import strip_headers
from gutenberg.query import get_metadata

from gender_analysis import common
from gender_analysis.common import AUTHOR_NAME_REGEX, BASE_PATH, METADATA_LIST


SUBJECTS_TO_IGNORE = ["nonfiction", "dictionaries", "bibliography", "poetry", "short stories", "biography", "encyclopedias",
             "atlases", "maps", "words and phrase lists", "almanacs", "handbooks, manuals, etc.", "periodicals",
             "textbooks", "terms and phrases", "essays", "united states. constitution", "bible", "directories",
             "songbooks", "hymns", "correspondence", "drama", "reviews", "translations into english", 'religion']
TRUNCATORS = ["\r", "\n", r"; Or, ", r"; or, "]
COUNTRY_ID_TO_NAME = {"Q30":  "United States", "Q145": "United Kingdom", "Q21": "United Kingdom", "Q16": "Canada",
                      "Q408": "Australia", "Q2886622": "Narnia"}

# This directory contains 11 sample books.
GUTENBERG_RSYNC_PATH = Path(BASE_PATH, 'corpora', 'gutenberg_mirror_sample')

TEXT_START_MARKERS = frozenset((
    "*END*THE SMALL PRINT",
    "*** START OF THE PROJECT GUTENBERG",
    "*** START OF THIS PROJECT GUTENBERG",
    "This etext was prepared by",
    "E-text prepared by",
    "Produced by",
    "Distributed Proofreading Team",
    "Proofreading Team at http://www.pgdp.net",
    "http://gallica.bnf.fr)",
    "      http://archive.org/details/",
    "http://www.pgdp.net",
    "by The Internet Archive)",
    "by The Internet Archive/Canadian Libraries",
    "by The Internet Archive/American Libraries",
    "public domain material from the Internet Archive",
    "Internet Archive)",
    "Internet Archive/Canadian Libraries",
    "Internet Archive/American Libraries",
    "material from the Google Print project",
    "*END THE SMALL PRINT",
    "***START OF THE PROJECT GUTENBERG",
    "This etext was produced by",
    "*** START OF THE COPYRIGHTED",
    "The Project Gutenberg",
    "http://gutenberg.spiegel.de/ erreichbar.",
    "Project Runeberg publishes",
    "Beginning of this Project Gutenberg",
    "Project Gutenberg Online Distributed",
    "Gutenberg Online Distributed",
    "the Project Gutenberg Online Distributed",
    "Project Gutenberg TEI",
    "This eBook was prepared by",
    "http://gutenberg2000.de erreichbar.",
    "This Etext was prepared by",
    "This Project Gutenberg Etext was prepared by",
    "Gutenberg Distributed Proofreaders",
    "Project Gutenberg Distributed Proofreaders",
    "the Project Gutenberg Online Distributed Proofreading Team",
    "**The Project Gutenberg",
    "*SMALL PRINT!",
    "More information about this book is at the top of this file.",
    "tells you about restrictions in how the file may be used.",
    "l'authorization à les utilizer pour preparer ce texte.",
    "of the etext through OCR.",
    "*****These eBooks Were Prepared By Thousands of Volunteers!*****",
    "We need your donations more than ever!",
    " *** START OF THIS PROJECT GUTENBERG",
    "****     SMALL PRINT!",
    '["Small Print" V.',
    '      (http://www.ibiblio.org/gutenberg/',
    'and the Project Gutenberg Online Distributed Proofreading Team',
    'Mary Meehan, and the Project Gutenberg Online Distributed Proofreading',
    '                this Project Gutenberg edition.',
))
TEXT_END_MARKERS = frozenset((
    "*** END OF THE PROJECT GUTENBERG",
    "*** END OF THIS PROJECT GUTENBERG",
    "***END OF THE PROJECT GUTENBERG",
    "End of the Project Gutenberg",
    "End of The Project Gutenberg",
    "Ende dieses Project Gutenberg",
    "by Project Gutenberg",
    "End of Project Gutenberg",
    "End of this Project Gutenberg",
    "Ende dieses Projekt Gutenberg",
    "        ***END OF THE PROJECT GUTENBERG",
    "*** END OF THE COPYRIGHTED",
    "End of this is COPYRIGHTED",
    "Ende dieses Etextes ",
    "Ende dieses Project Gutenber",
    "Ende diese Project Gutenberg",
    "**This is a COPYRIGHTED Project Gutenberg Etext, Details Above**",
    "Fin de Project Gutenberg",
    "The Project Gutenberg Etext of ",
    "Ce document fut presente en lecture",
    "Ce document fut présenté en lecture",
    "More information about this book is at the top of this file.",
    "We need your donations more than ever!",
    "END OF PROJECT GUTENBERG",
    " End of the Project Gutenberg",
    " *** END OF THIS PROJECT GUTENBERG",
))
LEGALESE_START_MARKERS = frozenset(("<<THIS ELECTRONIC VERSION OF",))
LEGALESE_END_MARKERS = frozenset(("SERVICE THAT CHARGES FOR DOWNLOAD",))


def generate_corpus_gutenberg():
    """
    Generate metadata sheet of all novels we want from gutenberg
    To test this run main
    """

    # Check if gutenberg corpus and text directories exists. Create if necessary.
    for path in [Path(BASE_PATH, 'corpora', 'gutenberg'), Path(BASE_PATH, 'corpora', 'gutenberg',
                                                               'texts')]:
        if not os.path.isdir(path):
            os.mkdir(path)

    # write csv header
    with open(Path(BASE_PATH, 'corpora', 'gutenberg', 'gutenberg.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=METADATA_LIST)
        writer.writeheader()
        print("Wrote metadata header")
    # check if cache is populated, if it isn't, populates it
    cache = get_metadata_cache()
    if (not cache.exists):
        print("Populating cache...")
        cache.populate()
        print("Done populating cache")

    number_books = 0

    # Generate filepaths
    # gutenberg_id.txt means ascii file (which works fine for our purposes)
    # gutenberg_id-0.txt means utf8 file (preferred)
    filepaths = []
    for gutenberg_id in range(70000):
        novel_directory = generate_gutenberg_rsync_path(gutenberg_id)
        utf8_path = novel_directory.joinpath(f'{gutenberg_id}-0.txt')
        if os.path.isfile(utf8_path):
            filepaths.append(utf8_path)
            continue
        ascii_path = novel_directory.joinpath(f'{gutenberg_id}.txt')
        if os.path.isfile(ascii_path):
            filepaths.append(ascii_path)

    print(f"Total number of files to process: {len(filepaths)}")

    for filepath in filepaths:
        gutenberg_id = int(filepath.parts[-2])
        try:
            print("Filepath:", filepath)
            print("ID:", gutenberg_id)
            # check if book is valid novel by our definition
            if (not is_valid_novel_gutenberg(gutenberg_id)):
                print("Not a novel")
                print('')
                continue
            novel_metadata = get_gutenberg_metadata_for_single_novel(gutenberg_id)
            write_metadata(novel_metadata)

            # Strip headers when storing the text file.
            with open(filepath, encoding='utf-8') as infile:
                text_raw = infile.read()
            text_clean = strip_headers(text_raw).strip()
            with open(Path(BASE_PATH, 'corpora', 'gutenberg', 'texts', f'{gutenberg_id}.txt'),
                      mode='w', encoding='utf-8') as outfile:
                outfile.write(text_clean)
            number_books += 1

        except Exception as exception:
            print("Ran into exception:", type(exception))
            print(exception)
            import traceback
            print(traceback.format_exc())
            print("")
            continue

    print("Done!")
    print("No. Books:", number_books)


def get_gutenberg_metadata_for_single_novel(gutenberg_id):
    """
    Retrieves the novel_metadata dict for one gutenberg book based on the gutenberg_id

    >>> get_gutenberg_metadata_for_single_novel(98) # doctest: +ELLIPSIS
    {'gutenberg_id': 98, 'corpus_name': 'gutenberg', 'author': ['Dickens, Charles'], ...

    :param gutenberg_id: int
    :return: dict
    """

    author = get_author_gutenberg(gutenberg_id)
    title = get_title_gutenberg(gutenberg_id)
    date = get_publication_date(author, title, gutenberg_id)
    country_publication = get_country_publication(author, title)
    author_gender = get_author_gender(author)
    subject = get_subject_gutenberg(gutenberg_id)

    novel_metadata = {
        'gutenberg_id':         gutenberg_id,
        'corpus_name':          'gutenberg',
        'author':               author,
        'title':                title,
        'date':                 date,
        'country_publication':  country_publication,
        'author_gender':        author_gender,
        'subject':              subject
    }
    return novel_metadata


def is_valid_novel_gutenberg(gutenberg_id):
    """
    Determines whether book with this gutenberg id is actually a "novel".  Returns false if the book
    is not or doesn't actually exist.
    Should check:
    If book is English
    If book is under public domain
    If book is a "novel"
    if novel is in correct publication range
    That novel is not a translation

    >>> from gender_analysis.gutenberg_loader import is_valid_novel_gutenberg
    >>> is_valid_novel_gutenberg(98) # Dickens, tale of two cities
    A Tale of Two Cities
    True
    >>> is_valid_novel_gutenberg(96)
    The Monster Men
    Not in date range
    False
    >>> is_valid_novel_gutenberg(1404)
    The Federalist Papers
    False

    :param gutenberg_id: int
    :return: boolean
    TODO: increase selectivity (Federalist Papers still failing doctest)
    """
    title = get_title_gutenberg(gutenberg_id)
    print(title)
    if language_invalidates_entry(gutenberg_id):
        print("Not in English")
        return False
    if rights_invalidate_entry(gutenberg_id):
        print("Not public domain")
        return False
    if subject_invalidates_entry(gutenberg_id):
        print("Bad subject")
        return False
    if date_invalidates_entry(gutenberg_id):
        print("Not in date range")
        return False
    # title = get_title_gutenberg(gutenberg_id)
    if title_invalidates_entry(title):
        print("Invalid title")
        return False
    text = get_novel_text_gutenberg_with_boilerplate(gutenberg_id)
    if text_invalidates_entry(text):
        print("Something wrong with text")
        return False
    return True


def language_invalidates_entry(gutenberg_id):
    """
    Returns False if book with gutenberg id is in English, True otherwise

    >>> from gender_analysis.gutenberg_loader import language_invalidates_entry
    >>> language_invalidates_entry(46) # A Christmas Carol
    False
    >>> language_invalidates_entry(27217) # Some Chinese thing
    True

    :param gutenberg_id: int
    :return: boolean
    """
    language = list(get_metadata('language', gutenberg_id))[0]
    if language != 'en':
        return True
    else:
        return False


def rights_invalidate_entry(gutenberg_id):
    """
    Returns False if book with gutenberg id is in public domain in US, True otherwise

    >>> from gender_analysis.gutenberg_loader import rights_invalidate_entry
    >>> rights_invalidate_entry(5200) # Metamorphosis by Franz Kafka
    True
    >>> rights_invalidate_entry(8066) # The Bible, King James version, Book 66: Revelation
    False

    :param gutenberg_id: int
    :return: boolean
    """
    rights = get_metadata('rights', gutenberg_id)
    if 'Public domain in the USA.' in rights:
        return False
    else:
        return True


def subject_invalidates_entry(gutenberg_id):
    """
    Checks if the Gutenberg subject indicates that the book is not a novel

    >>> from gender_analysis.gutenberg_loader import subject_invalidates_entry
    >>> subject_invalidates_entry(2240) # Much Ado About Nothing
    True
    >>> subject_invalidates_entry(33) # The Scarlet Letter
    False
    >>> subject_invalidates_entry(1404) # Federalist Papers
    True

    :param gutenberg_id: int
    :return: boolean
    """

    subjects = get_subject_gutenberg(gutenberg_id)
    for subject in subjects:
        for word in SUBJECTS_TO_IGNORE:
            if (subject.lower()).find(word) != -1:
                return True

    # Subject must include fiction in some form
    if not str(subjects).lower().find('fiction') > -1:
        return True

    return False


def date_invalidates_entry(gutenberg_id):
    """
    Checks if book with gutenberg id is in correct date range.  If it can't get the date, simply
    returns False
    >>> from gender_analysis.gutenberg_loader import date_invalidates_entry
    >>> date_invalidates_entry(33) # Hawthorne, Scarlet Letter
    False
    >>> date_invalidates_entry(173) # no publication date
    True

    :param gutenberg_id: int
    :return: boolean
    """

    author = get_author_gutenberg(gutenberg_id)
    title = get_title_gutenberg(gutenberg_id)
    try:
        date = int(get_publication_date(author, title, gutenberg_id))
        if date < 1770 or date > 1922:
            return True
        else:
            return False
    except TypeError:
        return True


def title_invalidates_entry(title):
    """
    Determines if the title contains phrases that indicate that the book is invalid

    >>> from gender_analysis.gutenberg_loader import title_invalidates_entry
    >>> title_invalidates_entry("Index of the Project Gutenberg Works of Michael Cuthbert")
    True
    >>> title_invalidates_entry("Pride and Prejudice")
    False

    :param title: str
    :return: boolean
    """

    title = title.lower()
    if title.find("index of the project gutenberg ") != -1:
        # print("Was an index")
        return True
    if title.find("complete project gutenberg") != -1:
        # print("Was a compilation thing")
        return True
    if title.find("translated by ") != -1:
        # print("Was a translation")
        return True
    # if (title.find("vol. ") != -1):
    #     return True
    # if re.match(r"volume \d+", title, flags= re.IGNORECASE):
    #     return True
    return False


def text_invalidates_entry(text):
    """
    Determine if there is anything obvious in the text that would invalidate it as a valid novel

    >>> from gender_analysis.gutenberg_loader import text_invalidates_entry
    >>> text_invalidates_entry("Translator: George Fyler Townsend")
    True
    >>> from gender_analysis.gutenberg_loader import get_novel_text_gutenberg
    >>> import os
    >>> current_dir = os.path.abspath(os.path.dirname(__file__))
    >>> filepath = Path(current_dir, r"corpora/sample_novels/texts/hawthorne_scarlet.txt")
    >>> scarlet_letter = get_novel_text_gutenberg(filepath)
    >>> text_invalidates_entry(scarlet_letter)
    False

    :param text: str
    :return: boolean
    """
    if text.find("Translator: ", 0, 650) != -1:
        return True
    text = strip_headers(text)
    text_length = len(text)
    # Animal Farm is roughly 166700 characters including boilerplate
    # Guiness World Records states that the longest novel is 9,609,000 characters long
    if text_length < 140000 or text_length > 9609000:
        return True
    return False


def get_author_gutenberg(gutenberg_id):
    """
    Gets author or authors for novel with this gutenberg id

    >>> from gender_analysis import gutenberg_loader
    >>> get_author_gutenberg(33)
    ['Hawthorne, Nathaniel']
    >>> get_author_gutenberg(3178)
    ['Twain, Mark', 'Warner, Charles Dudley']

    :param gutenberg_id: int
    :return: list
    """
    # TODO: should we format author names like this?

    return list(get_metadata('author', gutenberg_id))


def get_title_gutenberg(gutenberg_id):
    """
    Gets title for novel with this gutenberg id

    >>> from gender_analysis import gutenberg_loader
    >>> get_title_gutenberg(33)
    'The Scarlet Letter'

    """

    title = list(get_metadata('title', gutenberg_id))[0]
    for sep in TRUNCATORS:
        title = title.split(sep,1)[0]
    return title


def get_novel_text_gutenberg(gutenberg_id):
    """
    Extract text as as string from file, with boilerplate removed

    >>> from gender_analysis.gutenberg_loader import get_novel_text_gutenberg
    >>> text = get_novel_text_gutenberg(32)
    >>> text[:7]
    'HERLAND'

    :param gutenberg_id: int
    :return: str
    """
    return strip_headers(get_novel_text_gutenberg_with_boilerplate(gutenberg_id)).strip()


def get_novel_text_gutenberg_with_boilerplate(gutenberg_id):

    """
    Extract text as as string from file

    >>> from gender_analysis.gutenberg_loader import get_novel_text_gutenberg
    >>> text = get_novel_text_gutenberg_with_boilerplate(32)
    >>> text.split()[:3]
    ['The', 'Project', 'Gutenberg']

    :param gutenberg_id: int
    :return: str
    """

    novel_directory = generate_gutenberg_rsync_path(gutenberg_id)
    filepath = novel_directory.joinpath(f'{gutenberg_id}-0.txt')
    if not os.path.isfile(filepath):
        filepath = novel_directory.joinpath(f'{gutenberg_id}.txt')

    try:
        with open(filepath, mode='r', encoding='utf-8-sig') as text:
            text_with_headers = text.read()
    except UnicodeDecodeError:
        common.convert_text_file_to_new_encoding(source_path=filepath, target_path=filepath,
                                                 target_encoding='utf-8')
        with open(filepath, mode='r', encoding='utf-8-sig') as text:
            text_with_headers = text.read()

    return text_with_headers


def get_publication_date(author, title, gutenberg_id):
    """
    For a given novel with id gutenberg_id this function attempts a variety of
    methods to try and find the publication date
    If it can't returns None

    >>> from gender_analysis import gutenberg_loader
    >>> get_publication_date("Hawthorne, Nathaniel", "The Scarlet Letter", 33)
    1850

    # >>> from gender_analysis import gutenberg_loader
    # >>> get_publication_date("Dick, Phillip K.", "Mr. Spaceship", 32522)
    # 1953

    :param author: list
    :param title: str
    :param gutenberg_id: int
    :return: int
    """
    #TODO: remember to uncomment worldcat function when it is done

    novel_text = get_novel_text_gutenberg(gutenberg_id)
    date = get_publication_date_from_copyright_certain(novel_text)
    if date:
        return date
    else:
        date = get_publication_date_wikidata(author, title)
#    if (date == None):
#        date = get_publication_date_from_copyright_uncertain(novel_text)
    return date


def get_publication_date_wikidata(author, title):
    """
    For a given novel with this author and title this function attempts to pull the publication
    year from Wikidata Otherwise returns None
    N.B.: This fails if the title is even slightly wrong (e.g. The Adventures of Huckleberry Finn vs
    Adventures of Huckleberry Finn).  Should it be tried to fix that?
    Function also doesn't use author parameter

    >>> from gender_analysis import gutenberg_loader
    >>> get_publication_date_wikidata("Bacon, Francis", "Novum Organum")
    1620
    >>> get_publication_date_wikidata("Duan, Mingfei", "How I Became a Billionaire and also the President")

    >>> get_publication_date_wikidata("Austen, Jane", "Persuasion")
    1818
    >>> get_publication_date_wikidata("Scott, Walter", "Ivanhoe: A Romance")
    1820

    :param author: list
    :param title: str
    :return: int
    """

    try:
        site = pywikibot.Site("en", "wikipedia")
        page = pywikibot.Page(site, title)
        item = pywikibot.ItemPage.fromPage(page)
        dictionary = item.get()
        clm_dict = dictionary["claims"]
        clm_list = clm_dict["P577"]
        year = None
        for clm in clm_list:
            clm_trgt = clm.getTarget()
            year = clm_trgt.year
    except (KeyError):
        try:
            return get_publication_date_wikidata(author, title + " (novel)")
        except pywikibot.exceptions.NoPage:
            return None
    except pywikibot.exceptions.NoPage:
        try:
            if title == title.split(":", 1)[0].split(";,1")[0]:
                return None
            else:
                return get_publication_date_wikidata(author, title.split(":", 1)[0].split(";,1")[0])
        except pywikibot.exceptions.NoPage:
            return None
    except pywikibot.exceptions.InvalidTitle:
        return None
    return year


def get_publication_date_from_copyright_certain(novel_text):
    """
    Tries to extract the publication date from the copyright statement in the
    given text, if it is prefaced with some variation of 'COPYRIGHT'
    Otherwise returns None

    >>> novel_text = "This work blah blah blah blah COPYRIGHT, 1894 blah"
    >>> novel_text += "and they all died."
    >>> from gender_analysis.gutenberg_loader import get_publication_date_from_copyright_certain
    >>> get_publication_date_from_copyright_certain(novel_text)
    1894

    TODO: should this function take the novel's text as a string or the id or?
    TODO: split into two functions
    :return: int
    """
    match = re.search(r"(COPYRIGHT,?\s*) (\d{4})", novel_text, flags = re.IGNORECASE)
    if match:
        return int(match.group(2))
    else:
        return None


def get_publication_date_from_copyright_uncertain(novel_text):
    """
    Tries to extract the publication date from the copyright statement in the
    given text when it is not prefaced with some variation of 'COPYRIGHT'
    Otherwise returns None

    >>> novel_text = "This work blah blah blah blah Brams and Co. 1894 blah"
    >>> novel_text += "and they all died."
    >>> from gender_analysis.gutenberg_loader import get_publication_date_from_copyright_uncertain
    >>> get_publication_date_from_copyright_uncertain(novel_text)
    1894

    :return: int
    """
    match = re.search(r"\d{4}", novel_text[:3000])
    if match:
        year = int(match.group(0))
        if (year < 2038 and year > 1492):
            return year
        else:
            return None
    else:
        return None


def get_country_publication(author, title):
    """
    Tries to get the country of novel
    @TODO: Country of origin or residence of author?
    USA should be written as United States
    Don't separate countries of UK (England, Wales, etc.)
    TODO: should we separate those countries?  Easier to integrate later than separate

    >>> from gender_analysis import gutenberg_loader
    >>> get_country_publication("Hawthorne, Nathaniel", "The Scarlet Letter")
    'United States'

    :param author: list
    :param title: str
    :return: str
    """
    # TODO: uncomment worldcat function once it is added
    country = None
    # country = get_country_publication_worldcat(author, title)
    if (country == None):
        country = get_country_publication_wikidata(author, title)
    return country

def get_country_publication_wikidata(author, title):
    """
    Tries to get country of publication from wikidata
    Otherwise, returns None
    >>> from gender_analysis import gutenberg_loader
    >>> get_country_publication_wikidata("Trump, Donald", "Trump: The Art of the Deal")
    'United States'

    :param author: list
    :param title: str
    :return: str
    """
    # try to get publication country from wikidata page with title of book
    try:
        wikipedia = pywikibot.Site("en", "wikipedia")
        page = pywikibot.Page(wikipedia, title)
        item = pywikibot.ItemPage.fromPage(page)
        dictionary = item.get()
        clm_dict = dictionary["claims"]
        clm_list = clm_dict["P495"] # get claim "publication country", which is 495
        year = None
        for clm in clm_list:
            clm_trgt = clm.getTarget()
            country_id = clm_trgt.id
    # in case of disambiguation
    except (KeyError):
        try:
            return get_country_publication_wikidata(author, title + " (novel)")
        except (pywikibot.exceptions.NoPage):
            return None
    except (pywikibot.exceptions.NoPage):
        try:
            if (title == title.split(":", 1)[0].split(";,1")[0]):
                return None
            else:
                return get_country_publication_wikidata(author, title.split(":", 1)[0].split(";,1")[0])
        except (pywikibot.exceptions.NoPage):
            return None
    except(pywikibot.exceptions.InvalidTitle):
        return None

    # try to match country_id to major English-speaking countries
    if country_id in COUNTRY_ID_TO_NAME:
        return COUNTRY_ID_TO_NAME[country_id]

    # if not try look up wikidata page of country with that id to try and get short name
    try:
        wikidata = pywikibot.Site("wikidata", "wikidata")
        repo = wikidata.data_repository()
        item = pywikibot.ItemPage(repo, country_id)
        dictionary = item.get()
        clm_dict = dictionary["claims"]
        clm_list = clm_dict["P1813"] # P1813 is short name
        for clm in clm_list:
            clm_trgt = clm.getTarget()
            if (clm_trgt.language == 'en'):
                return clm_trgt.text
    except (KeyError, pywikibot.exceptions.NoPage, pywikibot.exceptions.InvalidTitle):
        #if that doesn't work just get one of the aliases. Name may be awkwardly long but should be consistent
        country = dictionary['aliases']['en'][-1]
        return country


def format_author(author):
    """
    Formats a string in the form "Lastname, Firstname" into "Firstname Lastname"
    and a string in the form "Lastname, Firstname Middle" into "Firstname Middle Lastname"
    and a string in the form "Lastname, F. M. (First Middle)" into "First Middle Lastname"
    and a string in the form "Lastname, Firstname, Sfx." into "Firstname Lastname, Sfx."
    If string is not in any of the above forms just returns the string.

    >>> from gender_analysis.gutenberg_loader import format_author
    >>> format_author("Washington, George")
    'George Washington'
    >>> format_author("Hurston, Zora Neale")
    'Zora Neale Hurston'
    >>> format_author('King, Martin Luther, Jr.')
    'Martin Luther King, Jr.'
    >>> format_author("Montgomery, L. M. (Lucy Maud)")
    'Lucy Maud Montgomery'
    >>> format_author("Socrates")
    'Socrates'

    :param author: str
    :return: str
    """

    author_formatted = ''
    try:
        match = re.match(AUTHOR_NAME_REGEX, author)
        first_name = match.groupdict()['first_name']
        if (match.groupdict()['real_name'] != None):
            first_name = match.groupdict()['real_name']
        author_formatted = first_name + " " + match.groupdict()['last_name']
        if (match.groupdict()['suffix'] != None):
            author_formatted += match.groupdict()['suffix']
    except (TypeError, AttributeError):
        author_formatted = author
    return author_formatted


def get_author_gender(authors):
    """
    Tries to get gender of author, 'female', 'male', 'non-binary', or 'both' (if there are multiple
    authors of different genders)
    Returns 'unknown' if it fails
    #TODO: should get functions that fail to find anything return 'unknown'
    #TODO: 'both' is ambiguous; Does it mean both female and male?  female and unknown?
    #TODO: male and nonbinary?

    >>> from gender_analysis.gutenberg_loader import get_author_gender
    >>> get_author_gender(["Hawthorne, Nathaniel"])
    'male'
    >>> get_author_gender(["Cuthbert, Michael"])
    'male'
    >>> get_author_gender(["Duan, Mingfei"])
    'unknown'
    >>> get_author_gender(["Collins, Suzanne", "Riordan, Rick"])
    'both'
    >>> get_author_gender(["Shelley, Mary", "Austen, Jane"])
    'female'

    # should return 'both' because it doesn't recognize my name

    >>> get_author_gender(['Shakespeare, William', "Duan, Mingfei"])
    'both'
    >>> get_author_gender(["Montgomery, L. M. (Lucy Maud)"])
    'female'
    >>> get_author_gender(['Shelley, Mary Wollstonecraft'])
    'female'

    :param authors: list
    :return: str
    """

    author_gender = None
    if type(authors) == str:
        authors = list().append(authors)
    if len(authors) == 1:
        author = authors[0]
        # author_gender = get_author_gender_worldcat(author)
        if author_gender == None:
            author_gender = get_author_gender_wikidata(author)
        if author_gender == None:
            author_gender = get_author_gender_guesser(author)
        if author_gender == None:
            return 'unknown'
    elif len(authors) > 1:
        author_gender = get_author_gender([authors[0]])
        for author in authors:
            if (get_author_gender([author]) != author_gender):
                author_gender = 'both'
    else:
        return 'unknown'

    return author_gender


def get_author_gender_wikidata(author):
    """
    Tries to get gender of author, 'female', 'male', 'non-binary' from wikidata
    If it fails returns None
    N.B. Wikidata's categories for transgender male and female are treated as male and female, respectively

    >>> from gender_analysis.gutenberg_loader import get_author_gender_wikidata
    >>> get_author_gender_wikidata("Obama, Barack")
    'male'
    >>> get_author_gender_wikidata("Hurston, Zora Neale")
    'female'
    >>> get_author_gender_wikidata("Bush, George W.")
    'male'

    :param author: str
    :return: str
    """
    author_formatted = format_author(author)
    try:
        site = pywikibot.Site("en", "wikipedia")
        page = pywikibot.Page(site, author_formatted)
        item = pywikibot.ItemPage.fromPage(page)
        dictionary = item.get()
        clm_dict = dictionary["claims"]
        return get_gender_from_wiki_claims(clm_dict)
    except (pywikibot.exceptions.NoPage, pywikibot.exceptions.InvalidTitle):
        return None


def get_gender_from_wiki_claims(clm_dict):
    """
    From clim_dict produced by pywikibot from a page, tries to extract the gender of the object of the page
    If it fails returns None
    N.B. Wikidata's categories for transgender male and female are treated as male and female, respectively

    >>> from gender_analysis.gutenberg_loader import get_gender_from_wiki_claims
    >>> site = pywikibot.Site("en", "wikipedia")
    >>> page = pywikibot.Page(site, 'Zeus')
    >>> item = pywikibot.ItemPage.fromPage(page)
    >>> dictionary = item.get()
    >>> clm_dict = dictionary["claims"]
    >>> get_gender_from_wiki_claims(clm_dict)
    'male'

    :param clm_dict: dict
    :return: str
    """
    try:
        clm_list = clm_dict["P21"]
        gender_id = None
        for clm in clm_list:
            clm_trgt = clm.getTarget()
            gender_id = clm_trgt.id
        if (gender_id == 'Q6581097' or gender_id == 'Q2449503'):
            return 'male'
        if (gender_id == 'Q6581072' or gender_id == 'Q1052281'):
            return 'female'
        if (gender_id == 'Q1097630'):
            return 'non-binary'
    except (KeyError, AttributeError):
        return None


def get_author_gender_guesser(author):
    """
    Tries to get gender of author, 'female', 'male', 'non-binary' from the gender guesser module

    >>> from gender_analysis.gutenberg_loader import get_author_gender_guesser
    >>> get_author_gender_guesser("Cuthbert, Michael")
    'male'
    >>> get_author_gender_guesser("Li, Michelle")
    'female'
    >>> get_author_gender_guesser("Duan, Mingfei") # should return None


    :param author: str
    :return: str
    """

    first_name = format_author(author).split()[0]
    guesser = gender_guesser.Detector()
    gender_guess = guesser.get_gender(first_name)
    if (gender_guess == 'andy' or gender_guess == 'unknown'):
        return None
    if (gender_guess == 'male' or gender_guess == 'mostly_male'):
        return 'male'
    if (gender_guess == 'female' or gender_guess == 'mostly_female'):
        return 'female'


def get_subject_gutenberg(gutenberg_id):
    """
    Tries to get subjects

    >>> from gender_analysis import gutenberg_loader
    >>> get_subject_gutenberg(5200)
    ['Metamorphosis -- Fiction', 'PT', 'Psychological fiction']

    :param: author: str
    :param: title: str
    :param: id: int
    :return: list
    """
    # TODO: run doctest on computer with populated cache

    return sorted(list(get_metadata('subject', gutenberg_id)))


def write_metadata(novel_metadata):
    """
    Writes a row of metadata for a novel into the csv at path
    Subject to change as metadata changes

    Running this doctest actually generates a file
    # >>> from gender_analysis import gutenberg_loader
    # >>> gutenberg_loader.write_metadata({'id': 105, 'author': 'Austen, Jane', 'title': 'Persuasion',
    # ...                            'corpus_name': 'gutenberg', 'date': '1818',
    # ...                            'country_publication': 'England', 'subject': ['England -- Social life and customs -- 19th century -- Fiction'],
    # ...                            'author_gender': 'female'})

    :param: novel_metadata: dict
    :param: path: Path
    """
    current_dir = os.path.abspath(os.path.dirname(__file__))
    corpus = novel_metadata['corpus_name']
    path = Path(current_dir, 'corpora', corpus, f'{corpus}.csv')
    with open(path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=METADATA_LIST)
        writer.writerow(novel_metadata)


def generate_gutenberg_rsync_path(gutenberg_id):
    """
    Generates the rsync path for a gutenberg novel based on the gutenberg_id

    >>> generate_gutenberg_rsync_path(9)
    PosixPath('/home/stephan/gutenberg_data/0/9')
    >>> generate_gutenberg_rsync_path(19)
    PosixPath('/home/stephan/gutenberg_data/1/19')
    >>> generate_gutenberg_rsync_path(125)
    PosixPath('/home/stephan/gutenberg_data/1/2/125')
    >>> generate_gutenberg_rsync_path(1113)
    PosixPath('/home/stephan/gutenberg_data/1/1/1/1113')
    >>> generate_gutenberg_rsync_path(11177)
    PosixPath('/home/stephan/gutenberg_data/1/1/1/7/11177')

    :return: Path
    """

    id_str = str(gutenberg_id)

    novel_path = Path(GUTENBERG_RSYNC_PATH)

    if gutenberg_id < 10:
        return novel_path.joinpath(Path('0', id_str))

    for i in range(5, 1, -1):
        try:
            novel_path = novel_path.joinpath(Path(id_str[-i]))
        except IndexError:
            pass

    novel_path = novel_path.joinpath(Path(id_str))

    return novel_path


def download_gutenberg_if_not_locally_available():
    """
    Checks if the gutenberg corpus is available locally. If not, downloads the corpus
    and extracts it to corpora/gutenberg

    No tests because the function depends on user input

    :return:
    """

    import os
    gutenberg_path = Path(common.BASE_PATH, 'corpora', 'gutenberg')

    # if there are more than 4000 text files available, we know that the corpus was downloaded
    # and can return
    try:
        no_gutenberg_novels=  len(os.listdir(Path(gutenberg_path, 'texts')))
        if no_gutenberg_novels > 4000:
            gutenberg_available = True
        else:
            gutenberg_available = False
    # if the texts path was not found, we know that we need to download the corpus
    except FileNotFoundError:
        gutenberg_available = False

    if not gutenberg_available:

        print("The Project Gutenberg corpus is currently not available on your system.",
              "It consists of more than 4000 novels and 1.8 GB of data.")
        download_prompt = input(
              "If you want to download the corpus, please enter (y). Any other input will "
              "terminate the program: ")
        if download_prompt not in ['y', '(y)']:
            raise ValueError("Project Gutenberg corpus will not be downloaded.")

        import zipfile
        import urllib
        url = 'https://s3-us-west-2.amazonaws.com/gutenberg-cache/gutenberg_corpus.zip'
        urllib.request.urlretrieve(url, 'gutenberg_corpus.zip')
        zipf = zipfile.ZipFile('gutenberg_corpus.zip')
        if not os.path.isdir(gutenberg_path):
            os.mkdir(gutenberg_path)
        zipf.extractall(gutenberg_path)
        os.remove('gutenberg_corpus.zip')
        metadata_url = r'https://raw.githubusercontent.com/dhmit/gender_novels/master' \
            r'/gender_novels/corpora/gutenberg/gutenberg.csv'
        r = requests.get(metadata_url, allow_redirects=True)
        with open(gutenberg_path/Path('gutenberg.csv'), 'wb') as metadata_file:
            metadata_file.write(r.content)

        # check that we now have 4000 novels available
        try:
            no_gutenberg_novels = len(os.listdir(Path(gutenberg_path, 'texts')))
            print(f'Successfully downloaded {no_gutenberg_novels} novels.')
        except FileNotFoundError:
            raise FileNotFoundError("Something went wrong while downloading the gutenberg"
                                    "corpus.")


def remove_boilerplate_text_with_gutenberg(text):
    """
    Removes the boilerplate text from an input string of a document.
    Currently only supports boilerplate removal for Project Gutenberg ebooks. Uses the
    strip_headers() function from the gutenberg module, which can remove even nonstandard
    headers.

    (see book number 3780 for one example of a nonstandard header — james_highway.txt in our
    sample corpus; or book number 105, austen_persuasion.txt, which uses the standard Gutenberg
    header but has had some info about the ebook's production inserted after the standard
    boilerplate).

    :return: str

    >>> from gender_analysis import document
    >>> from pathlib import Path
    >>> from gender_analysis import common
    >>> document_metadata = {'author': 'Austen, Jane', 'title': 'Persuasion',
    ...                   'date': '1818', 'filename': 'james_highway.txt', 'filepath': Path(common.BASE_PATH, 'testing', 'corpora', 'sample_novels', 'texts', 'james_highway.txt')}
    >>> austen = document.Document(document_metadata)
    >>> file_path = Path('testing', 'corpora', 'sample_novels', 'texts', austen.filename)
    >>> raw_text = austen.load_file(file_path)
    >>> raw_text = remove_boilerplate_text_with_gutenberg(raw_text)
    >>> title_line = raw_text.splitlines()[0]
    >>> title_line
    "THE KING'S HIGHWAY"

    TODO: neither version of remove_boilerplate_text works on Persuasion, and it doesn't look like it's
    easily fixable
    """

    try:
        return strip_headers(text).strip()
    except NameError:
        return _remove_boilerplate_text_without_gutenberg(text)


def _remove_boilerplate_text_without_gutenberg(text):
    """
    Removes the boilerplate text from an input string of a document.
    Currently only supports boilerplate removal for Project Gutenberg ebooks. Uses the
    strip_headers() function, somewhat inelegantly copy-pasted from the gutenberg module, which can remove even nonstandard
    headers.

    (see book number 3780 for one example of a nonstandard header — james_highway.txt in our
    sample corpus; or book number 105, austen_persuasion.txt, which uses the standard Gutenberg
    header but has had some info about the ebook's production inserted after the standard
    boilerplate).

    :return: str

    >>> from gender_analysis import document
    >>> from pathlib import Path
    >>> from gender_analysis import common
    >>> document_metadata = {'author': 'Austen, Jane', 'title': 'Persuasion',
    ...                   'date': '1818', 'filename': 'james_highway.txt', 'filepath': Path(common.BASE_PATH, 'testing', 'corpora', 'sample_novels', 'texts', 'james_highway.txt')}
    >>> austen = document.Document(document_metadata)
    >>> file_path = Path('testing', 'corpora', 'sample_novels', 'texts', austen.filename)
    >>> raw_text = austen.load_file(file_path)
    >>> raw_text = austen._remove_boilerplate_text_without_gutenberg(raw_text)
    >>> title_line = raw_text.splitlines()[0]
    >>> title_line
    "THE KING'S HIGHWAY"
    """

    # new method copy-pasted from Gutenberg library
    lines = text.splitlines()
    sep = '\n'

    out = []
    i = 0
    footer_found = False
    ignore_section = False

    for line in lines:
        reset = False

        if i <= 600:
            # Check if the header ends here
            if any(line.startswith(token) for token in TEXT_START_MARKERS):
                reset = True

            # If it's the end of the header, delete the output produced so far.
            # May be done several times, if multiple lines occur indicating the
            # end of the header
            if reset:
                out = []
                continue

        if i >= 100:
            # Check if the footer begins here
            if any(line.startswith(token) for token in TEXT_END_MARKERS):
                footer_found = True

            # If it's the beginning of the footer, stop output
            if footer_found:
                break

        if any(line.startswith(token) for token in LEGALESE_START_MARKERS):
            ignore_section = True
            continue
        elif any(line.startswith(token) for token in LEGALESE_END_MARKERS):
            ignore_section = False
            continue

        if not ignore_section:
            out.append(line.rstrip(sep))
            i += 1

    return sep.join(out).strip()


if __name__ == '__main__':
    # from dh_testers.testRunner import main_test
    # main_test()
    generate_corpus_gutenberg()
