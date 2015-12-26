import sys
import unittest
import numpy
import numpy.random
import randomstate.mlfg_1279_861 as mlfg_1279_861
import randomstate.mrg32k3a as mrg32k3a
import randomstate.mt19937 as mt19937
import randomstate.pcg32 as pcg32
import randomstate.pcg64 as pcg64
import randomstate.xorshift1024 as xorshift1024
import randomstate.xorshift128 as xorshift128

from nose import SkipTest

def comp_state(state1, state2):
    identical = True
    if isinstance(state1, dict):
        for key in state1:
            identical &= comp_state(state1[key], state2[key])
    else:
        if isinstance(state1, (list, tuple, numpy.ndarray)):
            for s1, s2 in zip(state1, state2):
                identical &= comp_state(s1, s2)
        else:
            identical &= state1 == state2
    return identical


class RNG(object):
    def _reset_state(self):
        self.rs.set_state(self.initial_state)

    def test_init(self):
        rs = self.mod.RandomState()
        state = rs.get_state()
        rs.random_uintegers(1)
        rs.set_state(state)
        new_state = rs.get_state()
        assert comp_state(state, new_state)

    def test_advance(self):
        state = self.rs.get_state()
        if hasattr(self.rs, 'advance'):
            self.rs.advance(self.advance)
            assert not comp_state(state, self.rs.get_state())
        else:
            raise SkipTest

    def test_jump(self):
        state = self.rs.get_state()
        if hasattr(self.rs, 'jump'):
            self.rs.jump()
            assert not comp_state(state, self.rs.get_state())
        else:
            raise SkipTest

    def test_random_uintegers(self):
        assert len(self.rs.random_uintegers(10)) == 10

    def test_uniform(self):
        r = self.rs.uniform(-1.0, 0.0, size=10)
        assert len(r) == 10
        print(r)
        assert (r > -1).all()
        assert (r <= 0).all()

    def test_random_sample(self):
        assert len(self.rs.random_sample(10)) == 10

    def test_standard_normal_zig(self):
        assert len(self.rs.standard_normal(10, method='zig')) == 10

    def test_standard_normal(self):
        assert len(self.rs.standard_normal(10)) == 10

    def test_standard_gamma(self):
        assert len(self.rs.standard_gamma(10, 10)) == 10

    def test_standard_exponential(self):
        assert len(self.rs.standard_exponential(10)) == 10

    def test_standard_cauchy(self):
        assert len(self.rs.standard_cauchy(10)) == 10

    def test_binomial(self):
        assert self.rs.binomial(10, .5) >= 0
        assert self.rs.binomial(1000, .5) >= 0

    def test_bounded_uint(self):
        assert len(self.rs.random_bounded_uintegers(2 ** 24 + 1, 10)) == 10
        assert len(self.rs.random_bounded_uintegers(2 ** 48 + 1, 10)) == 10

    def test_bounded_int(self):
        assert len(self.rs.random_bounded_integers(2 ** 24 + 1, size=10)) == 10
        assert len(self.rs.random_bounded_integers(2 ** 48 + 1, size=10)) == 10
        assert len(self.rs.random_bounded_integers(-2 ** 24, 2 ** 24 + 1, size=10)) == 10
        assert len(self.rs.random_bounded_integers(-2 ** 48, 2 ** 48 + 1, size=10)) == 10

    def test_reset_state(self):
        state = self.rs.get_state()
        int_1 = self.rs.random_uintegers(1)
        self.rs.set_state(state)
        int_2 = self.rs.random_uintegers(1)
        assert int_1 == int_2

    def test_entropy_init(self):
        rs = self.mod.RandomState()
        rs2 = self.mod.RandomState()
        print('\n'*10)
        print('*'*80)
        s1 = rs.get_state()
        s2 = rs2.get_state()
        print('\n'*10)
        assert not comp_state(rs.get_state(), rs2.get_state())

    def test_seed(self):
        rs = self.mod.RandomState(*self.seed)
        rs2 = self.mod.RandomState(*self.seed)
        assert comp_state(rs.get_state(), rs2.get_state())

    def test_reset_state_gauss(self):
        rs = self.mod.RandomState(*self.seed)
        rs.standard_normal()
        state = rs.get_state()
        n1 = rs.standard_normal(size=10)
        rs2 = self.mod.RandomState()
        rs2.set_state(state)
        n2 = rs2.standard_normal(size=10)
        assert (n1 == n2).all()

    def test_reset_state_uint32(self):
        rs = self.mod.RandomState(*self.seed)
        rs.random_uintegers(bits=32)
        state = rs.get_state()
        n1 = rs.random_uintegers(bits=32, size=10)
        rs2 = self.mod.RandomState()
        rs2.set_state(state)
        n2 = rs2.random_uintegers(bits=32, size=10)
        assert (n1 == n2).all()

    def test_shuffle(self):
        original = numpy.arange(200,0,-1)
        permuted = self.rs.permutation(original)
        assert (original != permuted).any()

    def test_permutation(self):
        original = numpy.arange(200,0,-1)
        permuted = self.rs.permutation(original)
        assert (original != permuted).any()

    def test_tomaxint(self):
        vals = self.rs.tomaxint(size=100000)
        if sys.maxsize < 2**32:
            assert (vals < sys.maxsize).all()
        else:
            assert (vals >= 2 ** 32).any()

class TestMT19937(RNG):
    @classmethod
    def setup_class(cls):
        cls.mod = mt19937
        cls.advance = None
        cls.seed = [2 ** 21 + 2 ** 16 + 2 ** 5 + 1]
        cls.rs = cls.mod.RandomState(*cls.seed)
        cls.initial_state = cls.rs.get_state()

    def test_numpy_state(self):

        nprs = numpy.random.RandomState()
        nprs.standard_normal(99)
        state = nprs.get_state()
        self.rs.set_state(state)
        state2 = self.rs.get_state()
        assert (state[1] == state2['state'][0]).all()
        assert (state[2] == state2['state'][1])
        assert (state[3] == state2['gauss']['has_gauss'])
        assert (state[4] == state2['gauss']['gauss'])


class TestPCG32(RNG, unittest.TestCase):
    @classmethod
    def setup_class(cls):
        cls.mod = pcg32
        cls.advance = 2 ** 48 + 2 ** 21 + 2 ** 16 + 2 ** 5 + 1
        cls.seed = [2 ** 48 + 2 ** 21 + 2 ** 16 + 2 ** 5 + 1, 2 ** 21 + 2 ** 16 + 2 ** 5 + 1]
        cls.rs = cls.mod.RandomState(*cls.seed)
        cls.initial_state = cls.rs.get_state()


class TestPCG64(RNG, unittest.TestCase):
    @classmethod
    def setup_class(cls):
        cls.mod = pcg64
        cls.advance = 2 ** 96 + 2 ** 48 + 2 ** 21 + 2 ** 16 + 2 ** 5 + 1
        cls.seed = [2 ** 96 + 2 ** 48 + 2 ** 21 + 2 ** 16 + 2 ** 5 + 1,
                    2 ** 21 + 2 ** 16 + 2 ** 5 + 1]
        cls.rs = cls.mod.RandomState(*cls.seed)
        cls.initial_state = cls.rs.get_state()


class TestXorShift128(RNG, unittest.TestCase):
    @classmethod
    def setup_class(cls):
        cls.mod = xorshift128
        cls.advance = None
        cls.seed = [12345]
        cls.rs = cls.mod.RandomState(*cls.seed)
        cls.initial_state = cls.rs.get_state()


class TestXorShift1024(RNG, unittest.TestCase):
    @classmethod
    def setup_class(cls):
        cls.mod = xorshift1024
        cls.advance = None
        cls.seed = [12345]
        cls.rs = cls.mod.RandomState(*cls.seed)
        cls.initial_state = cls.rs.get_state()


class TestMLFG(RNG, unittest.TestCase):
    @classmethod
    def setup_class(cls):
        cls.mod = mlfg_1279_861
        cls.advance = None
        cls.seed = [12345]
        cls.rs = cls.mod.RandomState(*cls.seed)
        cls.initial_state = cls.rs.get_state()


class TestMRG32k3A(RNG, unittest.TestCase):
    @classmethod
    def setup_class(cls):
        cls.mod = mrg32k3a
        cls.advance = None
        cls.seed = [12345]
        cls.rs = cls.mod.RandomState(*cls.seed)
        cls.initial_state = cls.rs.get_state()
