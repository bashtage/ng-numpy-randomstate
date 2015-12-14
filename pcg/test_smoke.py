import unittest

import pcg

class TestSmoke(unittest.TestCase):

    @classmethod
    def setup_class(cls):
        cls.rs = pcg.PCGRandomState()
        cls.initial_state = cls.rs.get_state()

    @classmethod
    def teardown_class(cls):
        pass

    def _reset_state(self):
        self.rs.set_state(self.initial_state)

    def test_init(self):
        rs = pcg.PCGRandomState(2**128-1, 2**96-1)
        state = rs.get_state()
        rs.random_integers(1)
        rs.set_state(state)
        new_state = rs.get_state()
        assert state == new_state

    def test_advance(self):
        state = self.rs.get_state()
        self.rs.advance(2**96)
        assert self.rs.get_state() != state

    def test_random_integers(self):
        assert len(self.rs.random_integers(10)) == 10

    def test_random_sample(self):
        assert len(self.rs.random_sample(10)) == 10

    def test_standard_normal_zig(self):
        assert len(self.rs.random_sample(10)) == 10

    def test_standard_normal(self):
        assert len(self.rs.random_sample(10)) == 10

    def test_standard_gamma(self):
        assert len(self.rs.standard_gamma(10, 10)) == 10

    def test_standard_exponential(self):
        assert len(self.rs.standard_exponential(10)) == 10
