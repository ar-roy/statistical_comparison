import unittest
import numpy as np
from shared.common.comparison import Comparison

class TestComparison(unittest.TestCase):

    def setUp(self):
        self.comparison_1 = Comparison('comparison_1', [2,2,2], [1,2,3], 'ref_1', 'test_1')
        self.comparison_2 = Comparison('comparison_2', [2,2,2], [[1,2,3], [2,2,2]],
                                       'ref_2', ['test_1a', 'test_1b'])

    def test_get_bias(self):
        self.assertIsNone(np.testing.assert_array_equal(self.comparison_1.get_bias(),
                                                        np.array([[-1,0,1]])
                                                        )
                          )
        self.assertIsNone(np.testing.assert_array_equal(self.comparison_2.get_bias(),
                                                        np.array([[-1,0,1], [0,0,0]])
                                                        )
                          )

    def test_get_CI_invalid(self, CI_percentage=94):
        with self.assertRaises(ValueError):
            self.comparison_1.get_CI(CI_percentage)

    def test_data_structure(self):
        self.assertIsInstance(self.comparison_1.get_bias(), np.ndarray)
        self.assertIsInstance(self.comparison_1.get_bias()[0], np.ndarray)
        self.assertIsInstance(self.comparison_1.get_bias()[0][0], np.int32)
        self.assertIsInstance(self.comparison_2.get_bias(), np.ndarray)
        self.assertIsInstance(self.comparison_2.get_bias()[0], np.ndarray)
        self.assertIsInstance(self.comparison_2.get_bias()[0][0], np.int32)


if __name__ == '__main__':
    unittest.main()