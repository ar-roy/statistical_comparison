"""
Create class used for comparing different sets.
Comparison.ref and Comparison.test can only be a lists of lists currently
Outputs are numpy arrays of lists
"""
import numpy as np
from scipy import stats


class Comparison:

    instances = []

    def __init__(self, name, ref, test, ref_name, test_names):
        self.name = name
        try:
            self.ref = np.asarray(ref)
        except TypeError:
            raise TypeError('Unsupported type ({}) for test variable'.format(type(test)))
        if isinstance(test[0], list):
            try:
                # FIXME: numpy array creation from list of lists flips dimensions
                self.test = np.asarray(test)
            except TypeError:
                raise TypeError('Unsupported type ({}) for test variable'.format(type(test)))
        else:
            try:
                self.test = np.asarray([test])
            except TypeError:
                raise TypeError('Unsupported type ({}) for test variable'.format(type(test)))
        self.ref_name = ref_name
        self.test_names = test_names
        self.__class__.instances.append(self)
        self.__class__.CI_z_relationship = {80: 1.282,
                                            85: 1.440,
                                            90: 1.645,
                                            95: 1.960,
                                            99: 2.576,
                                            99.5: 2.807,
                                            99.9: 3.291}

    def __eq__(self, other):
        return self.name == other.name

    def __lt__(self, other):
        return self.name < other.name

    @classmethod
    def printInstances(cls):
        print('----------Instances-------------')
        for instance in cls.instances:
            print(instance)
        print('--------------------------------')

    @classmethod
    def listInstances(cls):
        list_instances = []
        for instance in cls.instances:
            list_instances.append(instance)
        return list_instances

    def n_sample(self):
        try:
            n = np.ma.size(self.test, axis=1)
            return n
        except IndexError:
            raise IndexError('Cant get size of comparison dataset - self.test may not by numpy array')

    def get_mean(self):
        mean = []
        for vals in self.test:
            mean.append(vals.mean())
        return np.array(mean)

    def get_median(self):
        median = []
        for vals in self.test:
            median.append(vals.median())
        return np.array(median)

    def get_bias(self):
        bias = []
        for vals in self.test:
            bias.append(vals - self.ref)
        return np.array(bias)

    def get_percent_bias(self):
        percent_bias = []
        for vals in self.test:
            percent_bias.append(((vals - self.ref)/self.ref)*100)
        return np.array(percent_bias)

    def get_mean_bias(self):
        bias = self.get_bias()
        mean_bias = []
        for b in bias:
            mean_bias.append(b.mean())
        return np.array(mean_bias)

    def get_mean_percent_bias(self):
        percent_bias = self.get_percent_bias()
        mean_percent_bias = []
        for pb in percent_bias:
            mean_percent_bias.append(pb.mean())
        return np.array(mean_percent_bias)

    def get_error(self):
        bias = self.get_bias()
        error = abs(bias)
        return error

    def get_percent_error(self):
        percent_error = []
        for vals in self.test:
            percent_error.append((abs(vals - self.ref)/self.ref)*100)
        return np.array(percent_error)

    def get_mean_error(self):
        error = self.get_error()
        mean_error = []
        for e in error:
            mean_error.append(e.mean())
        return np.array(mean_error)

    def get_mean_percent_error(self):
        percent_error = self.get_percent_error()
        mean_percent_error = []
        for pe in percent_error:
            mean_percent_error.append(pe.mean())
        return np.array(mean_percent_error)

    def get_standard_deviation(self):
        std = []
        for vals in self.test:
            std.append(np.std(vals))
        return np.array(std)

    def get_relative_standard_deviation(self):
        std = self.get_standard_deviation()
        mean = self.get_mean()
        rsd = []
        for s, m in zip(std, mean):
            rsd.append(100*s/m)
        return np.array(rsd)

    def get_standard_error(self):
        std = self.get_standard_deviation()
        n_samples = self.n_sample()
        ste = []
        for i_std in std:
            ste.append(i_std / np.sqrt(n_samples))
        return np.array(ste)

    def get_CI(self, CI_percentage, bound_type='free', pop_type='ste', mean_type='bias'):
        if 'bias' in mean_type:
            mean = self.get_mean_bias()
        elif 'pop' in mean_type:
            mean = self.get_mean()
        elif 'percent_bias' in mean_type:
            mean = self.get_mean_percent_bias()

        if 'ste' in pop_type:
            st = self.get_standard_error()
        elif 'std' in pop_type:
            st = self.get_standard_deviation()
        elif 'rsd' in pop_type:
            st = self.get_relative_standard_deviation()

        z_value = self.CI_z_relationship.get(CI_percentage)
        if type(z_value) != float:
            raise ValueError(
                'comparison.get_CI_upper(): Percentage not acceptable, must be (80, 85, 90, 95, 99, 99.5, 99.9)')
        CI = []
        for i_mean_bias, i_st in zip(mean, st):
            if 'free' in bound_type:
                CI.append(z_value * i_st)
            elif 'upper' in bound_type:
                CI.append(i_mean_bias + (z_value * i_st))
            elif 'lower' in bound_type:
                CI.append(i_mean_bias - (z_value * i_st))
        return np.array(CI)

    def get_pvalue(self):
        p = []
        for vals in self.test:
            p.append(stats.ttest_rel(self.ref, vals).pvalue)
        return np.array(p)

    def get_pearsonr(self):
        r = []
        for vals in self.test:
            r_val, _ = stats.pearsonr(x=self.ref, y=vals)
            r.append(r_val)
        return np.array(r)

    def get_slope_intercept(self, test_data, ref_data, invert=False):
        slope = []
        intercept = []
        for vals in test_data:
            if invert:
                i_slope, i_intercept, _, _, _ = stats.linregress(x=ref_data, y=vals)
            else:
                i_slope, i_intercept, _, _, _ = stats.linregress(y=ref_data, x=vals)
            slope.append(i_slope)
            intercept.append(i_intercept)
        return np.array(slope), np.array(intercept)

    def print_names(self):
        print('----------Print Names-----------')
        print('Object: {}'.format(self))
        print('Comparison name: {}'.format(self.name))
        print('Reference name: {}'.format(self.ref_name))
        print('Test name: {}'.format(self.test_names))
        print('--------------------------------')


if __name__ == '__main__':
    pass