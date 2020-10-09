class PronounSeries:
    """
    A class that allows users to define a custom series of pronouns to be used in
    `gender_analysis` functions
    """

    def __init__(self, identifier, pronouns):
        """
        Creates a new series of pronouns, designated by the given identifier. Pronouns
        are any collection of strings that can be used to identify someone.

        Note that pronouns are case-insensitive.

        :param identifier: String used to identify what the particular series represents
        :param pronouns: Iterable of Strings that are to be used as pronouns for this group
        """

        self.identifier = identifier

        self.pronouns = set()
        for pronoun in pronouns:
            self.pronouns.add(pronoun.lower())

    def __contains__(self, pronoun):
        """
        Checks to see if the given pronoun exists in this group. This check is case-insensitive

        >>> from gender_analysis.pronouns import PronounSeries
        >>> pronoun_group = PronounSeries('Andy', {'They', 'Them', 'Theirs', 'Themself'})
        >>> 'they' in pronoun_group
        True
        >>> 'hers' in pronoun_group
        False

        :param pronoun: The pronoun to check for in this group
        :return: true if the pronoun is in the group, false otherwise
        """

        return pronoun.lower() in self.pronouns

    def __iter__(self):
        """
        Allows the user to iterate over all of the pronouns in this group. Pronouns
        are returned in lowercase and order is not guaranteed.

        >>> from gender_analysis.pronouns import PronounSeries
        >>> pronoun_group = PronounSeries('Fem', {'She', 'Her', 'hers', 'herself'})
        >>> sorted(pronoun_group)
        ['her', 'hers', 'herself', 'she']

        """
        yield from self.pronouns

    def __repr__(self):
        """
        >>> from gender_analysis.pronouns import PronounSeries
        >>> PronounSeries('Masc', {'he', 'him', 'his'})
        <Masc: ['he', 'him', 'his']>

        :return: A console-friendly representation of the pronoun series
        """

        return f'<{self.identifier}: {sorted(self.pronouns)}>'

    def __str__(self):
        """
        >>> from gender_analysis.pronouns import PronounSeries
        >>> str(PronounSeries('Andy', {'Xe', 'Xis', 'Xer'}))
        'Andy-series'

        :return: A string-representation of the pronoun series
        """

        return self.identifier + '-series'
