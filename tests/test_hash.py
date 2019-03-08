import unittest
import numpy as np
import pandas as pd
from model.hash import *

class TestHash(unittest.TestCase):

    def setUp(self):
        self.dim = 10
        rand = np.random.rand(1000, self.dim)
        self.data = pd.DataFrame(np.concatenate([rand, rand], axis = 1))
        rand = np.random.rand(np.random.randint(10, 100), self.dim)
        self.test = pd.DataFrame(np.concatenate([rand, rand], axis = 1))
        self.number_cell = 3

    def test_HashCAA(self):
        hash = HashCAA(5, self.number_cell)
        hash.fit(self.data)

        train = hash.fit_transform(self.data)
        self.assertEqual(train.shape[0], len(self.data))
        self.assertEqual(train.shape[1], self.number_cell)

        test = hash.transform(self.test)
        self.assertEqual(test.shape[0], len(self.test))
        self.assertEqual(test.shape[1], self.number_cell)

if __name__ == '__main__':
    unittest.main()