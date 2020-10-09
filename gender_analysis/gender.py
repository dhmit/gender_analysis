from gender_analysis.pronouns import PronounSeries


class Gender:
    """
    Defines a gender that will be operated on in analysis functions
    """

    def __init__(self, identifier, pronouns):
        """
        Initializes a Gender object that can be used for comparing and contrasting in the
        analysis functions.

        The gender accepts one or more `PronounSeries` that will be used to identify the gender.

        :param identifier: String name of the gender
        :param pronouns: `PronounSeries` or collection of `PronounSeries` that the gender uses.
        """

        self.identifier = identifier

        # Allow the user to input a single PronounSeries if only one applies
        if type(pronouns) == PronounSeries:
            self.pronouns = {pronouns}
        else:
            self.pronouns = set(pronouns)
