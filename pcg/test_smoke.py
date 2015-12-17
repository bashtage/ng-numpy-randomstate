import dummy
import pcg32
import pcg64
import randomkit
from nose import SkipTest
import xorshift128

def comp_state(state1, state2):
    identical = True
    try:
        iter(state1)
        for s1, s2 in zip(state1, state2):
            identical &= comp_state(s1, s2)
    except:
        identical &= s1 == s2
    return identical


class RNG(object):
    def _reset_state(self):
        self.rs.set_state(self.initial_state)

    def test_init(self):
        rs = self.mod.RandomState()
        state = rs.get_state()
        rs.random_integers(1)
        rs.set_state(state)
        new_state = rs.get_state()
        identical = True
        comp_state(state, new_state)
        assert identical

    def test_advance(self):
        state = self.rs.get_state()
        if hasattr(self.rs, 'advance'):
            self.rs.advance(self.advance)
            assert self.rs.get_state() != state
        else:
            raise SkipTest

    def test_random_integers(self):
        assert len(self.rs.random_integers(10)) == 10

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

    def test_reset_state(self):
        state = self.rs.get_state()
        int_1 = self.rs.random_integers(1)
        self.rs.set_state(state)
        int_2 = self.rs.random_integers(1)
        assert int_1 == int_2

    def test_entropy_init(self):
        rs = self.mod.RandomState()
        rs2 = self.mod.RandomState()
        assert not comp_state(rs.get_state(), rs2.get_state())

    def test_pareto(self):
        assert len(self.rs.pareto(2.0, 10)) == 10

    def test_weibull(self):
        assert len(self.rs.weibull(1, 10)) == 10

    def test_power(self):
        assert len(self.rs.power(2.0, 10)) == 10

    def test_rayleigh(self):
        assert len(self.rs.rayleigh(0.5, 10)) == 10

    def test_standard_t(self):
        assert len(self.rs.standard_t(4, 10)) == 10

    def test_chisquare(self):
        assert len(self.rs.chisquare(1, 10)) == 10

    def test_normal(self):
        assert len(self.rs.normal(0, 1, 10)) == 10

    def test_uniform(self):
        assert len(self.rs.uniform(1, 2, 10)) == 10

    def test_gamma(self):
        assert len(self.rs.gamma(2, 5, 10)) == 10

    def test_beta(self):
        assert len(self.rs.beta(0.5, 1.5, 10)) == 10

    def test_f(self):
        assert len(self.rs.beta(3, 373, 10)) == 10

    def test_laplace(self):
        assert len(self.rs.laplace(2, 4, 10)) == 10

    def test_gumbel(self):
        assert len(self.rs.gumbel(3.0, 5.0, 10)) == 10

    def test_lognormal(self):
        assert len(self.rs.lognormal(3.0, 5.0, 10)) == 10


class TestRandomKit(RNG):
    @classmethod
    def setup_class(cls):
        cls.mod = randomkit
        cls.advance = None
        cls.seed = [2 ** 21 + 2 ** 16 + 2 ** 5 + 1]
        cls.rs = cls.mod.RandomState(*cls.seed)
        cls.initial_state = cls.rs.get_state()


class TestDummy(RNG):
    @classmethod
    def setup_class(cls):
        cls.mod = dummy
        cls.advance = 2 ** 21 + 2 ** 16 + 2 ** 5 + 1
        cls.seed = [2 ** 21 + 2 ** 16 + 2 ** 5 + 1]
        cls.rs = cls.mod.RandomState(*cls.seed)
        cls.initial_state = cls.rs.get_state()


class TestPCG32(RNG):
    @classmethod
    def setup_class(cls):
        cls.mod = pcg32
        cls.advance = 2 ** 48 + 2 ** 21 + 2 ** 16 + 2 ** 5 + 1
        cls.seed = [2 ** 48 + 2 ** 21 + 2 ** 16 + 2 ** 5 + 1, 2 ** 21 + 2 ** 16 + 2 ** 5 + 1]
        cls.rs = cls.mod.RandomState(*cls.seed)
        cls.initial_state = cls.rs.get_state()


class TestPCG64(RNG):
    @classmethod
    def setup_class(cls):
        cls.mod = pcg64
        cls.advance = 2 ** 96 + 2 ** 48 + 2 ** 21 + 2 ** 16 + 2 ** 5 + 1
        cls.seed = [2 ** 96 + 2 ** 48 + 2 ** 21 + 2 ** 16 + 2 ** 5 + 1,
                    2 ** 48 + 2 ** 21 + 2 ** 16 + 2 ** 5 + 1]
        cls.rs = cls.mod.RandomState(*cls.seed)
        cls.initial_state = cls.rs.get_state()


class TestXorShift128(RNG):
    @classmethod
    def setup_class(cls):
        cls.mod = xorshift128
        cls.advance = None
        cls.seed = [2 ** 48 + 2 ** 21 + 2 ** 16 + 2 ** 5 + 1,
                    2 ** 21 + 2 ** 16 + 2 ** 5 + 1]
        cls.rs = cls.mod.RandomState(*cls.seed)
        cls.initial_state = cls.rs.get_state()
