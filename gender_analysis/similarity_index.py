from gender_analysis.character import Character


def get_conflated_characters(char_list, document):

    """
Takes in a list of characters and a document and runs through a human-computer collaboration
to determine which names are nicknames of one another. Creates a dictionary of Character objects.
"""
    similarity_dict = {}
    conflated_characters = []
    for name in char_list:
        similarity_dict[name] = [
            (potential_nickname, get_similarity_index(name, potential_nickname)) for potential_nickname in
            char_list]
        most_likely_candidates = sorted(similarity_dict[name])[:5]

        print("Select nicknames for ", name, " from the following candidates:")
        for n in len(most_likely_candidates):
            print(n, ": ", most_likely_candidates[n])
        nickname_indices = input(
            "Key in the numbers for nickname matches separated by spaces [e.g. '1 4 5'].").split(
            " ")
    nicknames = [most_likely_candidates[index][0] for index in nickname_indices]
    for n in len(nicknames):
        print(n, ": ", nicknames[n])
    canonical_name = nicknames[input(
        "Choose the Canonical Name for this character. "
        "Ideally, canonical names should be Firstname Lastname with no titles.")]
    gender = input("Select a gender for the character, or let the computer take its best guess.")
    # Handle this.

    new_character = Character(canonical_name, document, nicknames=[nicknames])
    conflated_characters[canonical_name] = new_character
    # We'll want to remove all selected names/nicknames from the character list
    # before proceeding in the interest of efficiency.

    return conflated_characters


def get_similarity_index(name, potential_nickname):
    """
    Takes a canonical name and a potential nickname, calculates a few different similarit indices, conflates them, and returns
    a confidence index that determines the likelihood of potential_nickname being a nickname for name.
    """

    total_similarity_index = 0

    # Each of these indices should give a result between 0 and 1,
    # with 0 meaning 'no chance of similarity' and 1 as 'exact match'.

    ngram_word_similarity_index = get_ngram_word_similarity_index(name, potential_nickname)
    ngram_char_similarity_index = get_ngram_char_similarity_index(name, potential_nickname)
    manual_nickname_checker_index = get_manual_nickname_checker_index(name, potential_nickname)
    honorific_similarity_index = get_honorific_similarity_index(name, potential_nickname)

    total_similarity_index = (
                                     ngram_word_similarity_index + ngram_char_similarity_index + manual_nickname_checker_index + honorific_similarity_index) / 4

    return total_similarity_index


def get_ngram_word_similarity_index(name, potential_nickname):
    """
    Takes a canonical name and a potential nickname, calculates how many ngrams within 'name' match within 'potential_nickname'.
    Returns a number between 0 and 1.
    """

    nwsi = 0

    name_ngrams = name.split(" ")
    nickname_ngrams = potential_nickname.split(" ")

    for ngram in name_ngrams:
        if ngram in nickname_ngrams:
            nwsi += 1

    nwsi = nwsi / len(name_ngrams)

    return nwsi


def get_ngram_char_similarity_index(name, potential_nickname):
    """
    Takes a canonical name and a potential nickname, calculates how many char-based
    ngrams within 'name' match within 'potential_nickname'.
    Returns a number between 0 and 1.
    """

    ncsi = 0

    # compare substrings
    return ncsi

def get_honorific_similarity_index(name, potential_nickname):
    """
    Takes a canonical name and potential nickname, checks for honorific similarities, ayda yada.
    """

    hsi = 0

    return hsi


def prepare_nickname_checker(path_to_nickname_list = "gender_analysis/nickname_list.txt"):
    """
    Turns the nickname list into a set of checkers.
    """

    nickname_list_raw = open(path_to_nickname_list, "r")
    nickname_list = [line for line in nickname_list_raw]
    nickname_name_list = []
    for name_pair in nickname_list:
        pair = name_pair.split(" = ")
        try:
            tuple_pair = (pair[0].strip(), pair[1].strip())
        except IndexError:
            print(pair, " is an invalid pair.")
        nickname_name_list.append(tuple_pair)

    return nickname_name_list


def run_nickname_check(char_list):

    char_list_dict = {}
    nickname_list = prepare_nickname_checker()
    for name in char_list:
        char_list_dict[name] = []
        for nickname in char_list:
            if name != nickname:
                mnci = get_manual_nickname_checker_index(name, nickname, nickname_list)
                char_list_dict[name].append((nickname, mnci))
                print(name, "matches", nickname)

    return char_list_dict

def get_manual_nickname_checker_index(name, potential_nickname, nickname_list):
    """
    Takes canonical name and potential nickname, uses handwritten nickname_list to check
    for similarities.
    """

    mnci = 0
    if name == potential_nickname:
        print("Exact match. Not interesting. mnci = -1")
        mnci = -1
        return mnci

    else:
        for nick_pair in nickname_list:
            if name in nick_pair:
                if potential_nickname in nick_pair:
                    print("Match between ", name, " and ", potential_nickname)
                    mnci = 1

    return mnci
