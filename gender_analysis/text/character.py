class Character:
    """
    Defines a character that will be operated on in analysis functions
    """

    def __init__(self, name, gender=None, mentions=None):
        """Initializes a character object which is associated with a document object
        :param name: a string of the name of the character
        :param gender: a gender object which is optional
        :param mentions: a list of strings of other references of that character, optional

        >>> from gender_analysis.text import character
        >>> from gender_analysis.gender.common import FEMALE
        >>> emma_name = 'Emma'
        >>> emma_gender = FEMALE
        >>> emma_mentions = ["Emma Woodhouse", "Emma", "Miss Woodhouse"]
        >>> emma = Character(emma_name, emma_gender, emma_mentions)
        >>> type(emma)
        <class 'gender_analysis.text.character.Character'>
        >>> emma.name
        'Emma'
        >>> emma.mentions
        ['Emma Woodhouse', 'Emma', 'Miss Woodhouse']
        >>> emma.gender
        <Female>
        """

        self.name = name
        if mentions is None:
            mentions = []
        self.mentions = mentions
        self.gender = gender
        if not gender:
            self.gender = self.get_char_gender()

    def __str__(self):
        """
        Overrides python print method for user-defined objects for Character class
        Returns the character name, gender, and all mentions (if applicable)
        :return: str
        >>> from gender_analysis.text.character import Character
        >>> from gender_analysis.gender.common import FEMALE
        >>> emma_name = 'Emma'
        >>> emma_gender = FEMALE
        >>> emma_mentions = ["Emma Woodhouse", "Emma", "Miss Woodhouse"]
        >>> emma = Character(emma_name, emma_gender, emma_mentions)
        >>> emma_string = str(emma)
        >>> emma_string
        "Emma,  Female, with mentions:  ['Emma Woodhouse', 'Emma', 'Miss Woodhouse']"
        """
        character = self.name + ', ' + ' ' + str(self.gender) + ', ' + 'with mentions:  ' + \
                    str(self.mentions)
        return character

    def __repr__(self):
        """
        Overrides the built-in __repr__ method
        Returns the object type (character) and then the document name
            in <>.
        :return: string
        >>> from gender_analysis.text.character import Character
        >>> from gender_analysis.gender.common import FEMALE
        >>> emma_name = 'Emma'
        >>> emma_gender = FEMALE
        >>> emma_mentions = ["Emma Woodhouse", "Emma", "Miss Woodhouse"]
        >>> emma = Character(emma_name, emma_gender, emma_mentions)
        >>> repr(emma)
        '<Character (Emma)>'
        """
        char_name = self.name
        return f'<Character ({char_name})>'

    def get_char_gender(self):
        """
        Get the gender for the character based on:
        1. If user entry exists, fetch entered gender
        2. If not, infer Character's gender based on coreference resolution and pronouns
        Currently, this function only retrieves user entered gender for the character objects
        :return: a gender object
        >>> from gender_analysis.text.character import Character
        >>> from gender_analysis.gender.common import FEMALE
        >>> emma_name = 'Emma'
        >>> emma_gender = FEMALE
        >>> emma_mentions = ["Emma Woodhouse", "Emma", "Miss Woodhouse"]
        >>> emma = Character(emma_name, emma_gender, emma_mentions)
        >>> emma.get_char_gender()
        <Female>
        """

        if self.gender:
            return self.gender
        else:
            return None
