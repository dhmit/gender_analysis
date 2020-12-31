from gender_analysis import document
class Character:
    """
    Defines a character that will be operated on in analysis functions
    """

    def __init__(self, name, document, gender=None,nicknames=[]): #maybe need to try/except?
        """Initializes a character object which is associated with a document object
        :param name: a string of the name of the character
        :param document: a document object where the character exists
        :param gender: a gender object which is optional
        :param nicknames: a list of strings of other references of that character, optional"""
        self.name = name
        self.document = document
        self.gender = gender
        self.nicknames = nicknames

    def __str__(self):
        character = self.name + ' in ' + str(self.document)
        return character

    def __repr__(self):
        """
        Overrides the built-in __repr__ method
        Returns the object type (character) and then the document name
            in <>.

        :return: string
        # these need to change!
        >>> from gender_analysis import document
        >>> from pathlib import Path
        >>> from gender_analysis import common
        >>> document_metadata = {'author': 'Austen, Jane', 'title': 'Persuasion', 'date': '1818',
        ...                      'filename': 'austen_persuasion.txt',
        ...                      'filepath': Path(common.TEST_DATA_PATH, 'sample_novels',
        ...                                       'texts', 'austen_persuasion.txt')}
        >>> austen = document.Document(document_metadata)
        >>> repr(austen)
        '<Document (austen_persuasion)>'
        """
        char_name = self.name
        return f'<Character ({char_name})>'

    def get_overall_popularity(self):
        """return the count of all occurrences of the character name and nicknames throughout the
        document
        """
        all_text = self.document.get_tokenized_text()
        names = [self.name] + self.nicknames
        popularity = 0
        for t in all_text:
            if t in names:
                popularity += 1
        return popularity

    def get_overall_popularity(self,document):
        """return the count of all occurrences of the character name and nicknames throughout the
        document
        gives more flexibility compared with the other get_overall_popularity func to customize
        the document we pass in
        :param document: a list of tokenized text"""
        # all_text = document.get_tokenized_text() this should be ideal but doesn't work with
        # current settings
        names = [self.name] + self.nicknames
        popularity = 0
        for t in document:
            if t in names:
                popularity += 1
        return popularity

    def get_staged_popularity(self,stage=10): # future work: make vis!
        """return a list of length stage of the popularity of the character in that stage
        return: a list of counts of length stage, similar to get_overall_popularity"""
        all_text = self.document.get_tokenized_text()
        splitted = self.split_list(all_text,num=stage) #not sure if I could do it with static method
        staged = []
        for t in splitted:
            staged.append(self.get_overall_popularity(splitted))
        return staged

    @staticmethod
    def split_list(seq,num=10):
        """helper function to split a list into num of sublist of almost equal length
        :param seq: a list of word tokens
        :param num: the num of stages to divide the tokens
        return: a list of list of divided list"""
        size = len(seq) // num
        divided = [seq[i:i + size] for i in range(0, len(seq), size)]
        if len(divided) > num:
            divided[-2] = divided[-2] + divided[-1]
            divided.pop()
        return divided

    def get_char_adjectives(self): # don't know if this should be put to the analysis folder
        name = self.name

