from unittest import TestCase
import randomstate.xorshift128 as xorshift128
import numpy as np
from numpy.testing import assert_equal, assert_allclose
import os
from os.path import join

pwd = os.path.dirname(os.path.abspath(__file__))


def uniform_from_uint64(x):
    a = x >> 37
    b = (x & 0xFFFFFFFF) >> 6
    return (a * 67108864.0 + b) / 9007199254740992.0


class TestXorshift128(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.RandomState = xorshift128.RandomState
        cls.bits = 64
        cls.dtype = np.uint64
        cls.u = uniform_from_uint64
        cls.data1 = cls._read_csv(join(pwd, './data/xorshift128-testset-1.csv'))
        cls.data2 = cls._read_csv(join(pwd, './data/xorshift128-testset-2.csv'))

    @classmethod
    def _read_csv(cls, filename):
        with open(filename) as csv:
            seed = csv.readline()
            seed = seed.split(',')
            seed = [long(s) for s in seed[1:]]
            data = []
            for line in csv:
                data.append(long(line.split(',')[-1]))
            return {'seed': seed, 'data': np.array(data, dtype=cls.dtype)}

    def test_raw(self):
        rs = self.RandomState(*self.data1['seed'])
        uints = rs.random_uintegers(1000, bits=self.bits)
        assert_equal(uints, self.data1['data'])

        rs = self.RandomState(*self.data2['seed'])
        uints = rs.random_uintegers(1000, bits=self.bits)
        assert_equal(uints, self.data2['data'])

    def test_double(self):
        rs = self.RandomState(*self.data1['seed'])
        uniforms = rs.random_sample(1000)
        assert_allclose(uniforms, uniform_from_uint64(self.data1['data']))

        rs = self.RandomState(*self.data2['seed'])
        uniforms = rs.random_sample(1000)
        assert_allclose(uniforms, uniform_from_uint64(self.data2['data']))
