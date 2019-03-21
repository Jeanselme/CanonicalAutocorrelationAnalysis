import unittest
import numpy as np
import pandas as pd
from model.caa import CAA
from model.hash import *

class TestHash(unittest.TestCase):

    def setUp(self):
        self.dim = 10
        rand = np.random.rand(1000, self.dim)
        self.data = pd.DataFrame(np.concatenate([rand, rand], axis = 1))
        self.labels = np.random.choice([0, 1], 1000)
        rand = np.random.rand(np.random.randint(10, 100), self.dim)
        self.test = pd.DataFrame(np.concatenate([rand, rand], axis = 1))
        self.number_cell = 3

    def test_CAA(self):
        caa = CAAModel(self.number_cell)
        caa.fit(self.data)

        train = caa.fit_transform(self.data)
        self.assertEqual(train.shape[0], len(self.data))
        self.assertLessEqual(train.shape[1], self.number_cell * 2)

        test = caa.transform(self.test)
        self.assertEqual(test.shape[0], len(self.test))
        self.assertLessEqual(test.shape[1], self.number_cell * 2)

    def test_HashCAA(self):
        hash = HashCAA(self.number_cell)
        hash.fit(self.data)

        train = hash.fit_transform(self.data)
        self.assertLessEqual(train.shape[0], self.number_cell)

        test = hash.transform(self.test)
        self.assertLessEqual(test.shape[0], self.number_cell)

    def test_HashCAA_MultiClass(self):
        hash = HashCAA(self.number_cell, [0, 1])
        hash.fit(self.data, self.labels)
        self.assertEqual(len(hash.caas), 2)

        train = hash.fit_transform(self.data)
        self.assertLessEqual(train.shape[0], self.number_cell * 2)

        test = hash.transform(self.test)
        self.assertLessEqual(test.shape[0], self.number_cell * 2)


if __name__ == '__main__':
    unittest.main()