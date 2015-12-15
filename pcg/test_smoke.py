import dummy
import pcg32
import pcg64
import randomkit
from nose import SkipTest


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



class TestRandomKit(RNG):
    @classmethod
    def setup_class(cls):
        cls.mod = randomkit
        cls.advance = None
        cls.seed = [2 ** 21 + 2 ** 16 + 2 ** 5 + 1]
        cls.rs = cls.mod.RandomState(*cls.seed)
        cls.initial_state = cls.rs.get_state()

    @classmethod
    def teardown_class(cls):
        pass


class TestDummy(RNG):
    @classmethod
    def setup_class(cls):
        cls.mod = dummy
        cls.advance = 2 ** 21 + 2 ** 16 + 2 ** 5 + 1
        cls.seed = [2 ** 21 + 2 ** 16 + 2 ** 5 + 1]
        cls.rs = cls.mod.RandomState(*cls.seed)
        cls.initial_state = cls.rs.get_state()

    @classmethod
    def teardown_class(cls):
        pass


class TestPCG32(RNG):
    @classmethod
    def setup_class(cls):
        cls.mod = pcg32
        cls.advance = 2 ** 48 + 2 ** 21 + 2 ** 16 + 2 ** 5 + 1
        cls.seed = [2 ** 48 + 2 ** 21 + 2 ** 16 + 2 ** 5 + 1, 2 ** 21 + 2 ** 16 + 2 ** 5 + 1]
        cls.rs = cls.mod.RandomState(*cls.seed)
        cls.initial_state = cls.rs.get_state()

    @classmethod
    def teardown_class(cls):
        pass


class TestPCG64(RNG):
    @classmethod
    def setup_class(cls):
        cls.mod = pcg64
        cls.advance = 2 ** 96 + 2 ** 48 + 2 ** 21 + 2 ** 16 + 2 ** 5 + 1
        cls.seed = [2 ** 96 + 2 ** 48 + 2 ** 21 + 2 ** 16 + 2 ** 5 + 1,
                    2 ** 48 + 2 ** 21 + 2 ** 16 + 2 ** 5 + 1]
        cls.rs = cls.mod.RandomState(*cls.seed)
        cls.initial_state = cls.rs.get_state()

    @classmethod
    def teardown_class(cls):
        pass
