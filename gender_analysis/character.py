import pickle
import pandas as pd
import sklearn
import numpy as np
# Libraries for building classifiers - do I need to import those?
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
# need to get nonbinary training data
from gender_analysis.common import MALE as male, FEMALE as female, NONBINARY as nonbinary

# load SVM classifier for gender detection
loaded_model = pickle.load(open('gender_analysis/gender_classifier.sav', 'rb'))


class Character:
    """
    Defines a character that will be operated on in analysis functions
    """

    def __init__(self, name, document, gender=None, nicknames=[]): # maybe need to try/except?
        """Initializes a character object which is associated with a document object
        :param name: a string of the name of the character
        :param document: a document object where the character exists
        :param gender: a gender object which is optional
        :param nicknames: a list of strings of other references of that character, optional"""
        self.name = name
        self.document = document  # change to an array of documents
        self.nicknames = nicknames
        self.gender = gender
        if not gender:
            print("Test")
            new_gender = self.get_char_gender()
            print(new_gender)

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

    def get_overall_popularity(self, document):
        """return the count of all occurrences of the character name and nicknames throughout the
        document
        gives more flexibility compared with the other get_overall_popularity func to customize
        the document we pass in
        :param document: a list of tokenized text"""
        # all_text = document.get_tokenized_text() this should be ideal but doesn't work with
        # current settings
        names = [self.name] + self.nicknames
        for n in names:
            n = n.lower()
        popularity = 0
        for t in document:
            if t in names:
                popularity += 1
        return popularity

    def get_staged_popularity(self, stage=10):
        # future work: make vis!
        """return a list of length stage of the popularity of the character in that stage
        return: a list of counts of length stage, similar to get_overall_popularity"""
        all_text = self.document.get_tokenized_text()
        splitted = self.split_list(all_text, num=stage)
        # not sure if I could do it with static method
        staged = []
        for t in splitted:
            staged.append(self.get_overall_popularity(splitted))
        return staged

    @staticmethod
    def split_list(seq, num=10):
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

    @staticmethod
    def letter_class(name):
        """helper function for get_char_gender
        param: a Character object
        output: a tuple of 2 integers - vowel_counter and consonant_counter - for the char name"""
        name_list = [x for x in name]
        vowel_counter = 0
        consonent_counter = 0
        for letter in name_list:
            if letter in ['a', 'e', 'i', 'o', 'u']:
                vowel_counter += 1
            else:
                consonent_counter += 1

        return vowel_counter, consonent_counter

    @staticmethod
    def ascii_mean(name):
        """helper function for get_char_gender
        param: a char object
        output: a float value of the ascii mean for the char name"""
        ascii_list = [ord(x) for x in name]
        return np.array(ascii_list).mean()

    def get_char_gender(self):
        # just binary for now! Need to add more customizability!
        """
        Use ML model to predict the gender of a character based on name
        return: a string indicating the gender of the character (i.e. 'Female'). If gender has
        been defined, then return the original gender.
        Update Character object with a gender
        """
        if self.gender is not None:
            return self.gender
        else:
            name_list = [self.name]
            ndf = pd.DataFrame([], columns=['name', 'ascii_value', 'name_len',
                                            'num_vowels', 'num_consonents', 'last_letter_vowel',
                                            ])
            ndf['name'] = name_list
            ndf['ascii_value'] = ndf['name'].apply(lambda x: self.ascii_mean(x).round(3))
            ndf['name_len'] = ndf['name'].apply(lambda x: len(x))
            ndf['num_vowels'] = ndf['name'].apply(lambda x: self.letter_class(x)[0])
            ndf['num_consonents'] = ndf['name'].apply(lambda x: self.letter_class(x)[1])
            ndf['last_letter_vowel'] = ndf['name'].apply(
                lambda x: 1 if x[-1] in ['a', 'e', 'i', 'o', 'u'] else 0)
            ndf['ends_with_a'] = ndf['name'].apply(lambda x: 1 if x[-1] == 'a' else 0)
            ndf['ascii_value'] = ndf['name'].apply(lambda x: self.ascii_mean(x).round(3))
            svc_op = loaded_model.predict(ndf.iloc[:, 1:].values)
            gender = svc_op[0]
            if gender == 'F':  # currently only supports binary classes 'F' or 'M', need to add others
                self.gender = female
            else:
                self.gender = male
            return gender

    def get_char_adjectives(self):  # don't know if this should be put to the analysis folder
        name = self.name
        pass

