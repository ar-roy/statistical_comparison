import unittest
from shared.common.comparison_diagnostic import ComparisonDiagnostic

class TestComparisonDiagnostic(unittest.TestCase):

    def setUp(self):
        self.comparison_1 = ComparisonDiagnostic('comparison_1', [1,2,3], [[2,2,2], [2,2,2]],
                                            'ref_1', ['test_1a', 'test_1b'])

    def test_get_diagnostic_accuracy(self, diagnostic_thresholds=(0,0.5,1), ref_diagnosis=(1,1,0)):
        self.comparison_1.get_diagnostic_accuracy(diagnostic_thresholds, ref_diagnosis)
        self.assertListEqual(self.comparison_1.sens, [[0.5,0.5,1.0], [0.5,0.5,1.0]])
        self.assertListEqual(self.comparison_1.spec, [[0.0,0.0,0.0], [0.0,0.0,0.0]])
        self.assertListEqual(self.comparison_1.roc_auc, [[0.25,0.25,0.5], [0.25,0.25,0.5]])

if __name__ == '__main__':
    unittest.main()

