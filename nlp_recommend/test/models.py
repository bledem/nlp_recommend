import unittest

from nlp_recommend import LoadData
from nlp_recommend import BaseModel, TfIdfModel


class TestStringMethods(unittest.TestCase):
    def test_data(self):
        first_sentence = 'And, nevertheless, the critical investigation of a principle of Judgement in these is the most important part in a Critique of this faculty.'
        first_token_lem = ['nevertheless', 'critical', 'investigation',
                           'principle', 'judgement', 'important', 'part', 'critique', 'faculty']
        corpus = LoadData(random=False)
        self.assertEqual(corpus.sentence.iloc[0], first_sentence)
        self.assertEqual(corpus.sentence.iloc[0], first_token_lem)

    def test_base(self):
        base = BaseModel()

    def test_tfidf(self):

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)


if __name__ == '__main__':
    unittest.main()
