"""Run with python -m unittest discover -s tests -v."""
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import OptimizeResult

from src.estimate_parameters import estimate_parameters
from src.simulate_tga_sv import simulate_tga_sv


class EstimationTests(unittest.TestCase):
    def test_optimizer_recovers_known_maximum_and_uses_fixed_randomness(self):
        target = np.array([.3, .7, .1, -.2, .15, .08])
        calls = []

        def likelihood(y, theta, **kwargs):
            calls.append(kwargs)
            return {'loglik': -len(y) * np.sum((theta - target) ** 2)}

        with patch('src.estimate_parameters.abc_filter', side_effect=likelihood):
            fit = estimate_parameters([-.5, 0., .5, 1.], maxfev=5000)
        self.assertTrue(fit['success'], fit['message'])
        np.testing.assert_allclose(fit['theta'], target, atol=.005)
        self.assertGreater(fit['loglik'], fit['initial_loglik'])
        self.assertTrue(all(c['seed'] == 123 and c['resampling'] == 'continuous'
                            and c['nu'] is None for c in calls))

    def test_ged_shape_is_estimated(self):
        def likelihood(y, theta, **kwargs):
            return {'loglik': -sum(np.asarray(theta) ** 2) - (kwargs['nu'] - 2.3) ** 2}
        with patch('src.estimate_parameters.abc_filter', side_effect=likelihood):
            fit = estimate_parameters([-.5, .5], dist='ged', estimate_nu=True,
                                      maxfev=3000)
        self.assertTrue(fit['success'], fit['message'])
        self.assertAlmostEqual(fit['nu'], 2.3, places=3)

    def test_invalid_inputs(self):
        for values, kwargs in [([1., np.nan], {}), ([1., 1.], {}),
                               ([1., 2.], {'estimate_nu': True}),
                               ([1., 2.], {'init_burn_in': -1}),
                               ([1., 2.], {'seed': None}),
                               ([1., 2.], {'n_particles': 1})]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                estimate_parameters(values, **kwargs)

    def test_budget_exhaustion_is_reported(self):
        with patch('src.estimate_parameters.abc_filter',
                   return_value={'loglik': -10.}):
            fit = estimate_parameters([-.5, .5], maxfev=2)
        self.assertFalse(fit['success'])
        self.assertLessEqual(fit['nfev'], 2)

    def test_worse_optimizer_endpoint_does_not_replace_start(self):
        def stop_at_worse_point(fun, x0, **kwargs):
            x = x0.copy()
            x[2] = .9
            return OptimizeResult(x=x, fun=fun(x), success=True,
                                  message='Stopped', nfev=1, nit=1)

        def likelihood(y, theta, **kwargs):
            return {'loglik': -float(theta[2] ** 2)}

        with patch('src.estimate_parameters.minimize', side_effect=stop_at_worse_point), \
             patch('src.estimate_parameters.abc_filter', side_effect=likelihood):
            fit = estimate_parameters([-.5, .5])
        self.assertFalse(fit['success'])
        self.assertEqual(fit['theta'][2], 0.)
        self.assertEqual(fit['loglik'], fit['initial_loglik'])

    def test_real_filter_is_reproducible_for_normal_and_ged(self):
        for dist, shape in [('normal', None), ('ged', 1.5)]:
            y = simulate_tga_sv(30, dist=dist, nu=shape, burn_in=100)['Return']
            kwargs = dict(dist=dist, nu=1.5, estimate_nu=dist == 'ged',
                          n_particles=40, init_burn_in=20, maxfev=40)
            a = estimate_parameters(y, **kwargs)
            b = estimate_parameters(y, **kwargs)
            np.testing.assert_array_equal(a['theta'], b['theta'])
            self.assertEqual(a['loglik'], b['loglik'])
            self.assertTrue(np.isfinite(a['loglik']))
            self.assertLess(abs(a['theta'][1]), 1.)
            self.assertGreater(a['theta'][5], 0.)


if __name__ == '__main__':
    unittest.main()
