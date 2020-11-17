class PronounSeries:
    """
    A class that allows users to define a custom series of pronouns to be used in
    `gender_analysis` functions
    """

    def __init__(self, identifier, pronouns, subj, obj):
        """
        Creates a new series of pronouns, designated by the given identifier. Pronouns
        are any collection of strings that can be used to identify someone.

        `subj` and `obj` will be considered part of the pronoun series regardless of whether
        they are listed in `pronouns`.

        Note that pronouns are case-insensitive.

        :param identifier: String used to identify what the particular series represents
        :param pronouns: Iterable of Strings that are to be used as pronouns for this group
        :param subj: String used as the "subject" pronoun of the series
        :param obj: String used as the "object" pronoun of the series
        """

        self.identifier = identifier
        self.subj = subj.lower()
        self.obj = obj.lower()

        self.pronouns = {self.subj, self.obj}
        for pronoun in pronouns:
            self.pronouns.add(pronoun.lower())

    def __contains__(self, pronoun):
        """
        Checks to see if the given pronoun exists in this group. This check is case-insensitive

        >>> from gender_analysis.pronouns import PronounSeries
        >>> pronouns = {'They', 'Them', 'Theirs', 'Themself'}
        >>> pronoun_group = PronounSeries('Andy', pronouns, 'they', 'them')
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
        >>> pronouns = {'She', 'Her', 'hers', 'herself'}
        >>> pronoun_group = PronounSeries('Fem', pronouns, subj='she', obj='her')
        >>> sorted(pronoun_group)
        ['her', 'hers', 'herself', 'she']

        """
        yield from self.pronouns

    def __repr__(self):
        """
        >>> from gender_analysis.pronouns import PronounSeries
        >>> PronounSeries('Masc', {'he', 'himself', 'his'}, subj='he', obj='him')
        <Masc: ['he', 'him', 'himself', 'his']>

        :return: A console-friendly representation of the pronoun series
        """

        return f'<{self.identifier}: {sorted(self.pronouns)}>'

    def __str__(self):
        """
        >>> from gender_analysis.pronouns import PronounSeries
        >>> str(PronounSeries('Andy', {'Xe', 'Xis', 'Xem'}, subj='xe', obj='xem'))
        'Andy-series'

        :return: A string-representation of the pronoun series
        """

        return self.identifier + '-series'

    def __hash__(self):
        """
        Makes the `PronounSeries` class hashable
        """

        return self.identifier.__hash__()

    def __eq__(self, other):
        """
        Determines whether two `PronounSeries` are equal. Note that they are only equal if
        they have the same identifier and the exact same set of pronouns.

        >>> from gender_analysis.pronouns import PronounSeries
        >>> fem_series = PronounSeries('Fem', {'she', 'her', 'hers'}, subj='she', obj='her')
        >>> second_fem_series = PronounSeries('Fem', {'she', 'her', 'hers'}, subj='she', obj='her')
        >>> fem_series == second_fem_series
        True
        >>> masc_series = PronounSeries('Masc', {'he', 'him', 'his'}, subj='he', obj='him')
        >>> fem_series == masc_series
        False

        :param other: The `PronounSeries` object to compare
        :return: `True` if the two series are the same, `False` otherwise.
        """

        return (
            self.identifier == other.identifier
            and self.pronouns == other.pronouns
            and self.obj == other.obj
            and self.subj == other.subj
        )
