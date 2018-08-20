from pysb import Monomer, Parameter, Initial, Observable, Rule
from pysb.bng import generate_equations
from pysb.testing import *
from opt2q.simulator import Simulator
from opt2q.utils import UnsupportedSimulatorError
from nose.tools import *
import unittest


class TestSolverModel(object):
    @with_model
    def setUp(self):
        Monomer('A', ['a'])
        Monomer('B', ['b'])

        Parameter('ksynthA', 100)
        Parameter('ksynthB', 100)
        Parameter('kbindAB', 100)

        Parameter('A_init', 0)
        Parameter('B_init', 0)

        Initial(A(a=None), A_init)
        Initial(B(b=None), B_init)

        Observable("A_free", A(a=None))
        Observable("B_free", B(b=None))
        Observable("AB_complex", A(a=1) % B(b=1))

        Rule('A_synth', None >> A(a=None), ksynthA)
        Rule('B_synth', None >> B(b=None), ksynthB)
        Rule('AB_bind', A(a=None) + B(b=None) >> A(a=1) % B(b=1), kbindAB)

        self.model = model

        # Convenience shortcut for accessing model monomer objects
        self.mon = lambda m: self.model.monomers[m]
        generate_equations(self.model)

        # Hack to prevent weird fails after assertDictEqual is called
        self.test_non_opt2q_params = None
        self.test_non_opt2q_params_df = None

    def tearDown(self):
        self.model=None
        self.mon=None
        self.test_non_opt2q_params = None
        self.test_non_opt2q_params_df = None


class TestSolver(TestSolverModel, unittest.TestCase):
    """test solver"""
    @raises(UnsupportedSimulatorError)
    def test_solver_exception(self):
        Simulator(self.model, solver='unsupported_solver')

    def test_get_solver_kwargs(self):
        sim = Simulator(self.model)
        test = sim._get_solver_kwargs({'a': 2})
        self.assertDictEqual({'a': 2}, test)

    def test_add_integrator_options_dict_none(self):
        sim = Simulator(self.model)
        self.assertDictEqual(sim.solver_kwargs, {'integrator_options': {}})  # when None return empty dict

    def test_add_integrator_options_dict_dict(self):
        sim = Simulator(self.model, integrator_options={'test_option':42})
        self.assertDictEqual(sim.solver_kwargs, {'integrator_options': {'test_option':42}})

    def test_custom_solver_options(self):
        sim = Simulator(self.model, solver_options={'integrator':'lsoda'}, integrator_options={'mxstep':2**10})
        assert sim.sim.opts == {'mxstep': 1024}
        assert sim.sim.integrator == 'lsoda'




