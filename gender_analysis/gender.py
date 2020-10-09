from gender_analysis.pronouns import PronounSeries


class Gender:
    """
    Defines a gender that will be operated on in analysis functions
    """

    def __init__(self, identifier, pronoun_series):
        """
        Initializes a Gender object that can be used for comparing and contrasting in the
        analysis functions.

        The gender accepts one or more `PronounSeries` that will be used to identify the gender.

        :param identifier: String name of the gender
        :param pronoun_series: `PronounSeries` or collection of `PronounSeries` that the gender uses.
        """

        self.identifier = identifier

        # Allow the user to input a single PronounSeries if only one applies
        if type(pronoun_series) == PronounSeries:
            self.pronoun_series = {pronoun_series}
        else:
            self.pronoun_series = set(pronoun_series)

    def __repr__(self):
        """
        :return: A console-friendly representation of the gender

        >>> from gender_analysis.pronouns import PronounSeries
        >>> from gender_analysis.gender import Gender
        >>> fem_pronouns = PronounSeries('Fem', {'she', 'her', 'hers'})
        >>> Gender('Female', fem_pronouns)
        <Female: {<Fem: ['her', 'hers', 'she']>}>
        """

        return f'<{self.identifier}: {self.pronoun_series}>'

    def __str__(self):
        """
        :return: A string representation of the gender

        >>> from gender_analysis.pronouns import PronounSeries
        >>> from gender_analysis.gender import Gender
        >>> fem_pronouns = PronounSeries('Fem', {'she', 'her', 'hers'})
        >>> str(Gender('Female', fem_pronouns))
        'Female'
        """

        return self.identifier

    def __hash__(self):
        """
        Allows the Gender object to be hashed
        """

        return self.identifier.__hash__()

    def __eq__(self, other):
        """
        Performs a check to see whether two `Gender`s are equivalent. This is true if and only
        if the `Gender`s' identifiers and pronoun series are identical.

        Note that this comparison works:
        >>> from gender_analysis.pronouns import PronounSeries
        >>> from gender_analysis.gender import Gender
        >>> fem_pronouns = PronounSeries('Fem', {'she', 'her', 'hers'})
        >>> female = Gender('Female', fem_pronouns)
        >>> another_female = Gender('Female', fem_pronouns)
        >>> female == another_female
        True

        But this one does not:
        >>> from gender_analysis.pronouns import PronounSeries
        >>> from gender_analysis.gender import Gender
        >>> they_series = PronounSeries('They', {'they', 'their', 'theirs'})
        >>> xe_series = PronounSeries('Xe', {'xe', 'xer', 'xis'})
        >>> androgynous_1 = Gender('NB', they_series)
        >>> androgynous_2 = Gender('NB', xe_series)
        >>> androgynous_1 == androgynous_2
        False

        :param other: The other `Gender` object to compare
        :return: `True` if the `Gender`s are the same, `False` otherwise
        """

        return (
            self.identifier == other.identifier and
            self.pronoun_series == other.pronoun_series
        )

    def uses_pronoun(self, pronoun):
        """
        Performs a check for whether the gender uses the given pronoun. Note that this is
        case-insensitive

        >>> from gender_analysis.pronouns import PronounSeries
        >>> from gender_analysis.gender import Gender
        >>> they_series = PronounSeries('They', {'they', 'them', 'theirs'})
        >>> xe_series = PronounSeries('Xe', {'Xe', 'Xer', 'Xis'})
        >>> androgynous = Gender('Androgynous', [they_series, xe_series])
        >>> androgynous.uses_pronoun('xer')
        True
        >>> androgynous.uses_pronoun('she')
        False

        :param pronoun: String representaion of the pronoun to check
        :return: `True` if this gender uses the pronoun, `False` otherwise
        """

        # Check if the gender uses the pronoun in at least one of its series
        for series in self.pronoun_series:
            if pronoun in series:
                return True

        return False

    def get_pronouns(self):
        """
        :return: A set containing all pronouns that this `Gender` uses

        >>> from gender_analysis.pronouns import PronounSeries
        >>> from gender_analysis.gender import Gender
        >>> they_series = PronounSeries('They', {'they', 'them', 'theirs'})
        >>> xe_series = PronounSeries('Xe', {'Xe', 'Xer', 'Xis'})
        >>> androgynous = Gender('Androgynous', [they_series, xe_series])
        >>> androgynous.get_pronouns() == {'they', 'them', 'theirs', 'xe', 'xer', 'xis'}
        True
        """

        pronouns = set()
        for series in self.pronoun_series:
            for pronoun in series:
                pronouns.add(pronoun)

        return pronouns
