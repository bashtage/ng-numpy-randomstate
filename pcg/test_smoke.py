import mt19937
import pcg32
import xorshift1024
import xorshift128
from nose import SkipTest

import pcg64


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

    def test_seed(self):
        rs = self.mod.RandomState(*self.seed)
        rs2 = self.mod.RandomState(*self.seed)
        assert comp_state(rs.get_state(), rs2.get_state())



class TestMT19937(RNG):
    @classmethod
    def setup_class(cls):
        cls.mod = mt19937
        cls.advance = None
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
                    2 ** 21 + 2 ** 16 + 2 ** 5 + 1]
        cls.rs = cls.mod.RandomState(*cls.seed)
        cls.initial_state = cls.rs.get_state()


class TestXorShift128(RNG):
    @classmethod
    def setup_class(cls):
        cls.mod = xorshift128
        cls.advance = None
        cls.seed = [12345]
        cls.rs = cls.mod.RandomState(*cls.seed)
        cls.initial_state = cls.rs.get_state()


class TestXorShift1024(RNG):
    @classmethod
    def setup_class(cls):
        cls.mod = xorshift1024
        cls.advance = None
        cls.seed = [12345]
        cls.rs = cls.mod.RandomState(*cls.seed)
        cls.initial_state = cls.rs.get_state()
