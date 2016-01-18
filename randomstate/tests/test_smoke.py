import pickle
import time

try:
    import cPickle
except ImportError:
    cPickle = pickle
import sys
import os
import unittest
import numpy as np
import randomstate.entropy as entropy
from randomstate.prng.mlfg_1279_861 import mlfg_1279_861
from randomstate.prng.mrg32k3a import mrg32k3a
from randomstate.prng.mt19937 import mt19937
from randomstate.prng.pcg32 import pcg32
from randomstate.prng.pcg64 import pcg64
from randomstate.prng.xorshift1024 import xorshift1024
from randomstate.prng.xorshift128 import xorshift128
from randomstate.prng.dsfmt import dsfmt
from numpy.testing import assert_almost_equal, assert_equal

from nose import SkipTest


def params_0(f):
    val = f()
    assert np.isscalar(val)
    val = f(10)
    assert val.shape == (10,)
    val = f((10, 10))
    assert val.shape == (10, 10)
    val = f((10, 10, 10))
    assert val.shape == (10, 10, 10)
    val = f(size=(5, 5))
    assert val.shape == (5, 5)


def params_1(f, bounded=False):
    a = 5.0
    b = np.arange(2.0, 12.0)
    c = np.arange(2.0, 102.0).reshape(10, 10)
    d = np.arange(2.0, 1002.0).reshape(10, 10, 10)
    e = np.array([2.0, 3.0])
    g = np.arange(2.0, 12.0).reshape(1, 10, 1)
    if bounded:
        a = 0.5
        b = b / (1.5 * b.max())
        c = c / (1.5 * c.max())
        d = d / (1.5 * d.max())
        e = e / (1.5 * e.max())
        g = g / (1.5 * g.max())

    # Scalar
    f(a)
    # Scalar - size
    f(a, size=(10, 10))
    # 1d
    f(b)
    # 2d
    f(c)
    # 3d
    f(d)
    # 1d size
    f(b, size=10)
    # 2d - size - broadcast
    f(e, size=(10, 2))
    # 3d - size
    f(g, size=(10, 10, 10))


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
    @classmethod
    def _extra_setup(cls):
        cls.vec_1d = np.arange(2.0, 102.0)
        cls.vec_2d = np.arange(2.0, 102.0)[None, :]
        cls.mat = np.arange(2.0, 102.0, 0.01).reshape((100, 100))

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
        assert (r > -1).all()
        assert (r <= 0).all()

    def test_uniform_array(self):
        r = self.rs.uniform(np.array([-1.0] * 10), 0.0, size=10)
        assert len(r) == 10
        assert (r > -1).all()
        assert (r <= 0).all()
        r = self.rs.uniform(np.array([-1.0] * 10), np.array([0.0] * 10), size=10)
        assert len(r) == 10
        assert (r > -1).all()
        assert (r <= 0).all()
        r = self.rs.uniform(-1.0, np.array([0.0] * 10), size=10)
        assert len(r) == 10
        assert (r > -1).all()
        assert (r <= 0).all()

    def test_random_sample(self):
        assert len(self.rs.random_sample(10)) == 10
        params_0(self.rs.random_sample)

    def test_standard_normal_zig(self):
        assert len(self.rs.standard_normal(10, method='zig')) == 10

    def test_standard_normal(self):
        assert len(self.rs.standard_normal(10)) == 10
        params_0(self.rs.standard_normal)

    def test_standard_gamma(self):
        assert len(self.rs.standard_gamma(10, 10)) == 10
        assert len(self.rs.standard_gamma(np.array([10] * 10), 10)) == 10
        params_1(self.rs.standard_gamma)

    def test_standard_exponential(self):
        assert len(self.rs.standard_exponential(10)) == 10
        params_0(self.rs.standard_exponential)

    def test_standard_cauchy(self):
        assert len(self.rs.standard_cauchy(10)) == 10
        params_0(self.rs.standard_cauchy)

    def test_standard_t(self):
        assert len(self.rs.standard_t(10, 10)) == 10
        params_1(self.rs.standard_t)

    def test_binomial(self):
        assert self.rs.binomial(10, .5) >= 0
        assert self.rs.binomial(1000, .5) >= 0

    def test_reset_state(self):
        state = self.rs.get_state()
        int_1 = self.rs.random_uintegers(1)
        self.rs.set_state(state)
        int_2 = self.rs.random_uintegers(1)
        assert int_1 == int_2

    def test_entropy_init(self):
        rs = self.mod.RandomState()
        rs2 = self.mod.RandomState()
        s1 = rs.get_state()
        s2 = rs2.get_state()
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
        original = np.arange(200, 0, -1)
        permuted = self.rs.permutation(original)
        assert (original != permuted).any()

    def test_permutation(self):
        original = np.arange(200, 0, -1)
        permuted = self.rs.permutation(original)
        assert (original != permuted).any()

    def test_tomaxint(self):
        vals = self.rs.tomaxint(size=100000)
        maxsize = 0
        if os.name == 'nt':
            maxsize = 2 ** 31 - 1
        else:
            try:
                maxsize = sys.maxint
            except:
                maxsize = sys.maxsize
        if maxsize < 2 ** 32:
            assert (vals < sys.maxsize).all()
        else:
            assert (vals >= 2 ** 32).any()

    def test_beta(self):
        vals = self.rs.beta(2.0, 2.0, 10)
        assert len(vals) == 10
        vals = self.rs.beta(np.array([2.0] * 10), 2.0)
        assert len(vals) == 10
        vals = self.rs.beta(2.0, np.array([2.0] * 10))
        assert len(vals) == 10
        vals = self.rs.beta(np.array([2.0] * 10), np.array([2.0] * 10))
        assert len(vals) == 10
        vals = self.rs.beta(np.array([2.0] * 10), np.array([[2.0]] * 10))
        assert vals.shape == (10, 10)

    def test_bytes(self):
        vals = self.rs.bytes(10)
        assert len(vals) == 10

    def test_chisquare(self):
        vals = self.rs.chisquare(2.0, 10)
        assert len(vals) == 10
        params_1(self.rs.chisquare)

    def test_exponential(self):
        vals = self.rs.exponential(2.0, 10)
        assert len(vals) == 10
        params_1(self.rs.exponential)

    def test_f(self):
        vals = self.rs.f(3, 1000, 10)
        assert len(vals) == 10

    def test_gamma(self):
        vals = self.rs.gamma(3, 2, 10)
        assert len(vals) == 10

    def test_geometric(self):
        vals = self.rs.geometric(0.5, 10)
        assert len(vals) == 10
        params_1(self.rs.exponential, bounded=True)

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
        vals = self.rs.rand(10, 10, 10)
        self.rs.set_state(state)
        assert (vals == self.rs.random_sample((10, 10, 10))).all()
        assert vals.shape == (10, 10, 10)

    def test_randn(self):
        state = self.rs.get_state()
        vals = self.rs.randn(10, 10, 10)
        self.rs.set_state(state)
        assert_equal(vals, self.rs.standard_normal((10, 10, 10)))
        assert_equal(vals.shape, (10, 10, 10))

        state = self.rs.get_state()
        vals = self.rs.randn(10, 10, 10, method='inv')
        self.rs.set_state(state)
        assert_equal(vals, self.rs.standard_normal((10, 10, 10), method='inv'))

        state = self.rs.get_state()
        vals_inv = self.rs.randn(10, 10, 10, method='inv')
        self.rs.set_state(state)
        vals_zig = self.rs.randn(10, 10, 10, method='zig')
        assert (vals_zig != vals_inv).any()

    def test_noncentral_chisquare(self):
        vals = self.rs.noncentral_chisquare(10, 2, 10)
        assert len(vals) == 10

    def test_noncentral_f(self):
        vals = self.rs.noncentral_f(3, 1000, 2, 10)
        assert len(vals) == 10
        vals = self.rs.noncentral_f(np.array([3] * 10), 1000, 2)
        assert len(vals) == 10
        vals = self.rs.noncentral_f(3, np.array([1000] * 10), 2)
        assert len(vals) == 10
        vals = self.rs.noncentral_f(3, 1000, np.array([2] * 10))
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
        vals = self.rs.poisson(np.array([10] * 10))
        assert len(vals) == 10
        params_1(self.rs.poisson)

    def test_poisson_lam_max(self):
        vals = self.rs.poisson_lam_max
        assert_almost_equal(vals, np.iinfo('l').max - np.sqrt(np.iinfo('l').max) * 10)

    def test_power(self):
        vals = self.rs.power(0.2, 10)
        assert len(vals) == 10

    def test_randint(self):
        vals = self.rs.randint(10, 20, 10)
        assert len(vals) == 10

    def test_random_integers(self):
        vals = self.rs.random_integers(10, 20, 10)
        assert len(vals) == 10

    def test_rayleigh(self):
        vals = self.rs.rayleigh(0.2, 10)
        assert len(vals) == 10
        params_1(self.rs.rayleigh, bounded=True)

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
        vals = self.rs.zipf(self.vec_1d)
        assert len(vals) == 100
        vals = self.rs.zipf(self.vec_2d)
        assert vals.shape == (1, 100)
        vals = self.rs.zipf(self.mat)
        assert vals.shape == (100, 100)

    def test_hypergeometric(self):
        vals = self.rs.hypergeometric(25, 25, 20)
        assert np.isscalar(vals)
        vals = self.rs.hypergeometric(np.array([25] * 10), 25, 20)
        assert vals.shape == (10,)

    def test_triangular(self):
        vals = self.rs.triangular(-5, 0, 5)
        assert np.isscalar(vals)
        vals = self.rs.triangular(-5, np.array([0] * 10), 5)
        assert vals.shape == (10,)

    def test_multivariate_normal(self):
        mean = [0, 0]
        cov = [[1, 0], [0, 100]]  # diagonal covariance
        x = self.rs.multivariate_normal(mean, cov, 5000)
        assert x.shape == (5000, 2)
        x_zig = self.rs.multivariate_normal(mean, cov, 5000, method='zig')
        assert x.shape == (5000, 2)
        x_inv = self.rs.multivariate_normal(mean, cov, 5000, method='inv')
        assert x.shape == (5000, 2)
        assert (x_zig != x_inv).any()

    def test_multinomial(self):
        vals = self.rs.multinomial(100, [1.0 / 3, 2.0 / 3])
        assert vals.shape == (2,)
        vals = self.rs.multinomial(100, [1.0 / 3, 2.0 / 3], size=10)
        assert vals.shape == (10, 2)

    def test_dirichlet(self):
        s = self.rs.dirichlet((10, 5, 3), 20)
        assert s.shape == (20, 3)

    def test_pickle(self):
        pick = pickle.dumps(self.rs)
        unpick = pickle.loads(pick)
        assert (type(self.rs) == type(unpick))
        assert comp_state(self.rs.get_state(), unpick.get_state())

        pick = cPickle.dumps(self.rs)
        unpick = cPickle.loads(pick)
        assert (type(self.rs) == type(unpick))
        assert comp_state(self.rs.get_state(), unpick.get_state())


class TestMT19937(RNG):
    @classmethod
    def setup_class(cls):
        cls.mod = mt19937
        cls.advance = None
        cls.seed = [2 ** 21 + 2 ** 16 + 2 ** 5 + 1]
        cls.rs = cls.mod.RandomState(*cls.seed)
        cls.initial_state = cls.rs.get_state()
        cls._extra_setup()

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
        cls._extra_setup()


class TestPCG64(RNG, unittest.TestCase):
    @classmethod
    def setup_class(cls):
        cls.mod = pcg64
        cls.advance = 2 ** 96 + 2 ** 48 + 2 ** 21 + 2 ** 16 + 2 ** 5 + 1
        cls.seed = [2 ** 96 + 2 ** 48 + 2 ** 21 + 2 ** 16 + 2 ** 5 + 1,
                    2 ** 21 + 2 ** 16 + 2 ** 5 + 1]
        cls.rs = cls.mod.RandomState(*cls.seed)
        cls.initial_state = cls.rs.get_state()
        cls._extra_setup()


class TestXorShift128(RNG, unittest.TestCase):
    @classmethod
    def setup_class(cls):
        cls.mod = xorshift128
        cls.advance = None
        cls.seed = [12345]
        cls.rs = cls.mod.RandomState(*cls.seed)
        cls.initial_state = cls.rs.get_state()
        cls._extra_setup()


class TestXorShift1024(RNG, unittest.TestCase):
    @classmethod
    def setup_class(cls):
        cls.mod = xorshift1024
        cls.advance = None
        cls.seed = [12345]
        cls.rs = cls.mod.RandomState(*cls.seed)
        cls.initial_state = cls.rs.get_state()
        cls._extra_setup()


class TestMLFG(RNG, unittest.TestCase):
    @classmethod
    def setup_class(cls):
        cls.mod = mlfg_1279_861
        cls.advance = None
        cls.seed = [12345]
        cls.rs = cls.mod.RandomState(*cls.seed)
        cls.initial_state = cls.rs.get_state()
        cls._extra_setup()


class TestMRG32k3A(RNG, unittest.TestCase):
    @classmethod
    def setup_class(cls):
        cls.mod = mrg32k3a
        cls.advance = None
        cls.seed = [12345]
        cls.rs = cls.mod.RandomState(*cls.seed)
        cls.initial_state = cls.rs.get_state()
        cls._extra_setup()


class TestDSFMT(RNG, unittest.TestCase):
    @classmethod
    def setup_class(cls):
        cls.mod = dsfmt
        cls.advance = None
        cls.seed = [12345]
        cls.rs = cls.mod.RandomState(*cls.seed)
        cls.initial_state = cls.rs.get_state()
        cls._extra_setup()


class TestEntropy(unittest.TestCase):
    def test_entropy(self):
        e1 = entropy.random_entropy()
        e2 = entropy.random_entropy()
        assert (e1 != e2)
        e1 = entropy.random_entropy(10)
        e2 = entropy.random_entropy(10)
        assert (e1 != e2).all()
        e1 = entropy.random_entropy(10, source='system')
        e2 = entropy.random_entropy(10, source='system')
        assert (e1 != e2).all()

    def test_fallback(self):
        e1 = entropy.random_entropy(source='fallback')
        time.sleep(0.1)
        e2 = entropy.random_entropy(source='fallback')
        assert (e1 != e2)


if __name__ == '__main__':
    import nose

    nose.run(argv=[__file__, '-vv'])
