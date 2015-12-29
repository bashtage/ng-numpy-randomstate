import sys
import sys
import unittest
import numpy as np
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
        if isinstance(state1, (list, tuple, np.ndarray)):
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

    def test_uniform_array(self):
        r = self.rs.uniform(np.array([-1.0]*10), 0.0, size=10)
        assert len(r) == 10
        assert (r > -1).all()
        assert (r <= 0).all()
        r = self.rs.uniform(np.array([-1.0]*10), np.array([0.0]*10), size=10)
        assert len(r) == 10
        assert (r > -1).all()
        assert (r <= 0).all()
        r = self.rs.uniform(-1.0, np.array([0.0]*10), size=10)
        assert len(r) == 10
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
        assert len(self.rs.standard_gamma(np.array([10]*10), 10)) == 10

    def test_standard_gamma_array(self):
        assert len(self.rs.standard_gamma(np.array([10]*10), 10)) == 10

    def test_standard_exponential(self):
        assert len(self.rs.standard_exponential(10)) == 10

    def test_standard_cauchy(self):
        assert len(self.rs.standard_cauchy(10)) == 10

    def test_standard_t(self):
        assert len(self.rs.standard_t(10, 10)) == 10

    def test_standard_array(self):
        assert len(self.rs.standard_t(np.arange(1,11.0), 10)) == 10

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
        original = np.arange(200,0,-1)
        permuted = self.rs.permutation(original)
        assert (original != permuted).any()

    def test_permutation(self):
        original = np.arange(200,0,-1)
        permuted = self.rs.permutation(original)
        assert (original != permuted).any()

    def test_tomaxint(self):
        vals = self.rs.tomaxint(size=100000)
        if sys.maxsize < 2**32:
            assert (vals < sys.maxsize).all()
        else:
            assert (vals >= 2 ** 32).any()

    def test_beta(self):
        vals = self.rs.beta(2.0,2.0, 10)
        assert len(vals) == 10
        vals = self.rs.beta(np.array([2.0]*10),2.0)
        assert len(vals) == 10
        vals = self.rs.beta(2.0, np.array([2.0]*10))
        assert len(vals) == 10
        vals = self.rs.beta(np.array([2.0]*10), np.array([2.0]*10))
        assert len(vals) == 10

    def test_bytes(self):
        vals = self.rs.bytes(10)
        assert len(vals) == 10

    def test_chisquare(self):
        vals = self.rs.chisquare(2.0, 10)
        assert len(vals) == 10

    def test_exponential(self):
        vals = self.rs.exponential(2.0, 10)
        assert len(vals) == 10

    def test_f(self):
        vals = self.rs.f(3, 1000, 10)
        assert len(vals) == 10

    def test_gamma(self):
        vals = self.rs.gamma(3, 2, 10)
        assert len(vals) == 10

    def test_geometric(self):
        vals = self.rs.geometric(0.5, 10)
        assert len(vals) == 10

    def test_gumbel(self):
        vals = self.rs.gumbel(2.0, 2.0, 10)
        assert len(vals) == 10

    def test_laplace(self):
        vals = self.rs.laplace(2.0, 2.0, 10)
        assert len(vals) == 10

    def test_logitic(self):
        vals = self.rs.logistic(2.0, 2.0, 10)
        assert len(vals) == 10

    def test_logseries(self):
        vals = self.rs.logseries(0.5, 10)
        assert len(vals) == 10

    def test_negative_binomial(self):
        vals = self.rs.negative_binomial(10, 0.2, 10)
        assert len(vals) == 10

    def test_rand(self):
        state = self.rs.get_state()
        vals = self.rs.rand(10,10,10)
        self.rs.set_state(state)
        assert (vals == self.rs.random_sample((10,10,10))).all()
        assert vals.shape == (10,10,10)

    def test_randn(self):
        state = self.rs.get_state()
        vals = self.rs.randn(10,10,10)
        self.rs.set_state(state)
        assert (vals == self.rs.standard_normal((10,10,10))).all()
        assert vals.shape == (10,10,10)

    def test_noncentral_chisquare(self):
        vals = self.rs.noncentral_chisquare(10, 2, 10)
        assert len(vals) == 10

    def test_noncentral_f(self):
        vals = self.rs.noncentral_f(3, 1000, 2, 10)
        assert len(vals) == 10
        vals = self.rs.noncentral_f(np.array([3]*10), 1000, 2)
        assert len(vals) == 10
        vals = self.rs.noncentral_f(3, np.array([1000]*10), 2)
        assert len(vals) == 10
        vals = self.rs.noncentral_f(3, 1000, np.array([2]*10))
        assert len(vals) == 10

    def test_normal(self):
        vals = self.rs.normal(10, 0.2, 10)
        assert len(vals) == 10

    def test_pareto(self):
        vals = self.rs.pareto(3.0, 10)
        assert len(vals) == 10

    def test_poisson(self):
        vals = self.rs.poisson(10, 10)
        assert len(vals) == 10
        vals = self.rs.poisson(np.array([10]*10))
        assert len(vals) == 10

    def test_poisson_lam_max(self):
        vals = self.rs.poisson_lam_max
        assert np.abs(vals - (np.iinfo('l').max - np.sqrt(np.iinfo('l').max)*10)) < (self.rs.poisson_lam_max * np.finfo('d').eps)

    def test_power(self):
        vals = self.rs.power(0.2, 10)
        assert len(vals) == 10

    def test_randint(self):
        vals = self.rs.randint(10, 0.2, 10)
        assert len(vals) == 10

    def test_random_integers(self):
        vals = self.rs.random_integers(10, 20, 10)
        assert len(vals) == 10

    def test_rayleigh(self):
        vals = self.rs.rayleigh(0.2, 10)
        assert len(vals) == 10

    def test_vonmises(self):
        vals = self.rs.vonmises(10, 0.2, 10)
        assert len(vals) == 10

    def test_wald(self):
        vals = self.rs.wald(1.0, 1.0, 10)
        assert len(vals) == 10

    def test_weibull(self):
        vals = self.rs.weibull(1.0, 10)
        assert len(vals) == 10

    def test_zipf(self):
        vals = self.rs.zipf(10, 10)
        assert len(vals) == 10


class TestMT19937(RNG):
    @classmethod
    def setup_class(cls):
        cls.mod = mt19937
        cls.advance = None
        cls.seed = [2 ** 21 + 2 ** 16 + 2 ** 5 + 1]
        cls.rs = cls.mod.RandomState(*cls.seed)
        cls.initial_state = cls.rs.get_state()

    def test_numpy_state(self):

        nprs = np.random.RandomState()
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
