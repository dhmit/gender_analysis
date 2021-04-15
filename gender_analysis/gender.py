from gender_analysis.pronouns import PronounSeries



class Gender:
    """
    Defines a gender that will be operated on in analysis functions
    """

    def __init__(self, label, pronoun_series, names=None):
        """
        Initializes a Gender object that can be used for comparing and contrasting in the
        analysis functions.

        The gender accepts one or more `PronounSeries` that will be used to identify the gender.

        >>> from gender_analysis.pronouns import PronounSeries
        >>> from gender_analysis.gender import Gender
        >>> he_series = PronounSeries('Masc', {'he', 'him', 'his'}, subj='he', obj='him')
        >>> masc_names = ['Andrew', 'Richard', 'Robert']
        >>> male = Gender('Male', he_series, names=masc_names)
        >>> 'Richard' in male.names
        True
        >>> 'robert' in male.names
        False
        >>> 'his' in male.pronouns
        True

        :param label: String name of the gender
        :param pronoun_series: `PronounSeries` or collection of `PronounSeries` that the gender uses
        :param names: A collection of names (as strings) that will be associated with the gender.
            Note that the provided names are case-sensitive.

        """

        self.label = label

        # Allow the user to input a single PronounSeries if only one applies
        if isinstance(pronoun_series, PronounSeries):
            self.pronoun_series = {pronoun_series}
        else:
            self.pronoun_series = set(pronoun_series)

        self.names = set(names) if names is not None else set()

    def __repr__(self):
        """
        :return: A console-friendly representation of the gender

        >>> from gender_analysis.pronouns import PronounSeries
        >>> from gender_analysis.gender import Gender
        >>> fem_pronouns = PronounSeries('Fem', {'she', 'her', 'hers'}, subj='she', obj='her')
        >>> Gender('Female', fem_pronouns)
        <Female: {<Fem: ['her', 'hers', 'she']>}>
        """

        return f'<{self.label}: {self.pronoun_series}>'

    def __str__(self):
        """
        :return: A string representation of the gender

        >>> from gender_analysis.pronouns import PronounSeries
        >>> from gender_analysis.gender import Gender
        >>> fem_pronouns = PronounSeries('Fem', {'she', 'her', 'hers'}, subj='she', obj='her')
        >>> str(Gender('Female', fem_pronouns))
        'Female'
        """

        return self.label

    def __hash__(self):
        """
        Allows the Gender object to be hashed
        """

        return self.label.__hash__()

    def __eq__(self, other):
        """
        Performs a check to see whether two `Gender`s are equivalent. This is true if and only
        if the `Gender`s' identifiers, pronoun series, and names are identical.

        Note that this comparison works:
        >>> from gender_analysis.pronouns import PronounSeries
        >>> from gender_analysis.gender import Gender
        >>> fem_pronouns = PronounSeries('Fem', {'she', 'her', 'hers'}, subj='she', obj='her')
        >>> female = Gender('Female', fem_pronouns)
        >>> another_female = Gender('Female', fem_pronouns)
        >>> female == another_female
        True

        But this one does not:
        >>> from gender_analysis.pronouns import PronounSeries
        >>> from gender_analysis.gender import Gender
        >>> they_series = PronounSeries('They', {'they', 'them', 'theirs'}, subj='they', obj='them')
        >>> xe_series = PronounSeries('Xe', {'xe', 'xer', 'xem'}, subj='xe', obj='xem')
        >>> androgynous_1 = Gender('NB', they_series)
        >>> androgynous_2 = Gender('NB', xe_series)
        >>> androgynous_1 == androgynous_2
        False

        :param other: The other `Gender` object to compare
        :return: `True` if the `Gender`s are the same, `False` otherwise
        """

        return (
            self.label == other.label
            and self.pronoun_series == other.pronoun_series
            and self.names == other.names
        )

    @property
    def pronouns(self):
        """
        :return: A set containing all pronouns that this `Gender` uses

        >>> from gender_analysis.pronouns import PronounSeries
        >>> from gender_analysis.gender import Gender
        >>> they_series = PronounSeries('They', {'they', 'them', 'theirs'}, subj='they', obj='them')
        >>> xe_series = PronounSeries('Xe', {'Xe', 'Xer', 'Xis'}, subj='xe', obj='xer')
        >>> androgynous = Gender('Androgynous', [they_series, xe_series])
        >>> androgynous.pronouns == {'they', 'them', 'theirs', 'xe', 'xer', 'xis'}
        True
        """

        pronouns = set()
        for series in self.pronoun_series:
            for pronoun in series:
                pronouns.add(pronoun)

        return pronouns

    @property
    def identifiers(self):
        """
        :return: Set of all words (i.e. pronouns and names) that are used to identify the gender

        >>> from gender_analysis.pronouns import PronounSeries
        >>> from gender_analysis.gender import Gender
        >>> fem_pronouns = PronounSeries('Fem', {'she', 'her', 'hers'}, subj='she', obj='her')
        >>> fem_names = {'Sarah', 'Marigold', 'Annabeth'}
        >>> female = Gender('Female', fem_pronouns, fem_names)
        >>> female.identifiers == {'she', 'her', 'hers', 'Sarah', 'Marigold', 'Annabeth'}
        True
        """

        return self.names | self.pronouns

    @property
    def subj(self):
        """
        :return: set of all subject pronouns used to describe the gender

        >>> from gender_analysis.pronouns import PronounSeries
        >>> from gender_analysis.gender import Gender
        >>> fem_pronouns = PronounSeries('Fem', {'she', 'her', 'hers'}, subj='she', obj='her')
        >>> masc_pronouns = PronounSeries('Masc', {'he', 'him', 'his'}, subj='he', obj='him')
        >>> bigender = Gender('Bigender', [fem_pronouns, masc_pronouns])
        >>> bigender.subj == {'he', 'she'}
        True

        """

        subject_pronouns = set()
        for series in self.pronoun_series:
            subject_pronouns.add(series.subj)

        return subject_pronouns

    @property
    def obj(self):
        """
        :return: set of all object pronouns used to describe the gender

        >>> from gender_analysis.pronouns import PronounSeries
        >>> from gender_analysis.gender import Gender
        >>> fem_pronouns = PronounSeries('Fem', {'she', 'her', 'hers'}, subj='she', obj='her')
        >>> masc_pronouns = PronounSeries('Masc', {'he', 'him', 'his'}, subj='he', obj='him')
        >>> bigender = Gender('Bigender', [fem_pronouns, masc_pronouns])
        >>> bigender.obj == {'him', 'her'}
        True

        """

        object_pronouns = set()
        for series in self.pronoun_series:
            object_pronouns.add(series.obj)

        return object_pronouns
