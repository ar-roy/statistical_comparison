from shared.common.comparison import Comparison
import sklearn.metrics as skl
import numpy as np
np.seterr(divide='ignore', invalid='ignore')


class ComparisonDiagnostic(Comparison):


    def __init__(self, name, ref, test, ref_name, test_names):
        super().__init__(name, ref, test, ref_name, test_names)
        self.acc, self.sens, self.spec, self.npv, self.ppv, self.conf_matrix, self.fpr,\
        self.roc_thresholds, self.roc_auc, self.roc_fpr, self.roc_tpr = ([] for i in range(11))

    def get_diagnostic_accuracy(self, reference_threshold=0.8):
        self.reference_threshold = reference_threshold
        for i_test in self.test:
            actual = np.where(self.ref>reference_threshold, 0, 1)
            predicted = np.where(i_test>reference_threshold, 0, 1)
            TN_count, FP_count, FN_count, TP_count = skl.confusion_matrix(actual,
                                                                          predicted,
                                                                          labels=[0, 1]
                                                                          ).ravel()
            self.conf_matrix.append([[TP_count, FP_count], [FN_count, TN_count]])
            self.acc.append((TP_count + TN_count) / (TP_count + TN_count + FP_count + FN_count))
            self.sens.append(TP_count / (TP_count + FN_count))
            self.spec.append(TN_count / (FP_count + TN_count))
            self.ppv.append(1 if np.isnan((TP_count / (TP_count + FP_count))) else (TP_count / (TP_count + FP_count)))
            self.npv.append(TN_count / (TN_count + FN_count))
            self.fpr.append(1-self.spec[-1])

    def get_auc(self, predicted_thresholds, reference_threshold=0.8):
        if self.acc is None:
            self.get_diagnostic_accuracy(reference_threshold)
        self.roc_thresholds = predicted_thresholds
        for i_test in self.test:
            tpr, fpr = ([] for i in range(2))
            actual = np.where(self.ref>self.reference_threshold, 0, 1)
            for thresh in self.roc_thresholds:
                predicted = np.where(i_test>thresh, 0, 1)
                TN_count, FP_count, FN_count, TP_count = skl.confusion_matrix(actual,
                                                                              predicted,
                                                                              labels=[0, 1]
                                                                              ).ravel()
                spec_thresh = TN_count / (FP_count + TN_count)
                sens_thresh = TP_count / (TP_count + FN_count)
                tpr.append(sens_thresh)
                fpr.append(1 - spec_thresh)
            self.roc_fpr.append(fpr)
            self.roc_tpr.append(tpr)
            self.roc_auc.append(skl.auc(fpr, tpr))

    def print_values(self):
        print('--------Diagnostic Performance--------')
        print('Object: {}'.format(self))
        print('Comparison name: {}'.format(self.name))
        print('Reference threshold: {}'.format(self.reference_threshold))
        print(f'Confusion matrix [TP, FP, FN, TN]: {self.conf_matrix}')
        print('Accuracy: {}'.format(self.acc))
        print('Sensitivity: {}'.format(self.sens))
        print('Specificity: {}'.format(self.spec))
        print('NPV: {}'.format(self.npv))
        print('PPV: {}'.format(self.ppv))
        print('FPR: {}'.format(self.fpr))
        # print('ROC AUC: {}'.format(self.roc_auc))
        print('--------------------------------------')


if __name__ == '__main__':
    pass
