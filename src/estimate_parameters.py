#!/usr/bin/env python3
"""Approximate maximum likelihood estimation for the TGA-SV model.

Maximize the ABC log-likelihood of Mao et al., Section 4.1, using the
existing abc_filter and continuous Malik-Pitt resampling (Appendix C).
Optimize c = mu * (1 - phi), as in Section 4.2, then report mu.

A fixed seed supplies common random numbers throughout each optimization.
Powell's bounded, derivative-free method avoids relying on numerical
gradients: continuous resampling helps, but sorting and scipy's GED sampler
can still make the simulated objective nonsmooth. The optimizer and bounds
below are practical choices, not specifications claimed from the paper.

Examples (run from the repository root)::

    python src/estimate_parameters.py --csv data/sp500_in_sample.csv
    python src/estimate_parameters.py --csv data/sp500_in_sample.csv \
        --dist ged --estimate-nu --repeats 10

Finite-particle estimates have approximation bias and Monte Carlo error.
Repeat optimization across seeds and increase the particle count to assess
stability. Across-seed standard deviations are NOT sampling standard errors.
No global optimum or exact reproduction of the paper is guaranteed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

if __package__:
    from .abc_filter import abc_filter
else:
    from abc_filter import abc_filter


PARAMETER_NAMES = ("mu", "phi", "alpha", "gamma1", "gamma2", "sigma_eta2")
# Bounds are in optimizer coordinates: c, phi, alpha, gamma1, gamma2, variance.
# They are deliberately finite to avoid explosive trial simulations. Inspect
# boundary hits and widen scientifically plausible bounds when needed.
DEFAULT_BOUNDS = ((-2., 2.), (-.995, .995), (-1., 1.),
                  (-1., 1.), (-1., 1.), (1e-5, 2.))
NU_BOUNDS = (.5, 10.)


def estimate_parameters(
    y: Sequence[float],
    initial_theta: Sequence[float] | None = None,
    *,
    n_particles: int = 5000,
    dist: str = "normal",
    nu: float = 1.5,
    estimate_nu: bool = False,
    seed: int = 123,
    init_burn_in: int = 1000,
    bounds: Sequence[tuple[float, float]] | None = None,
    maxiter: int = 100,
    maxfev: int = 2000,
    xtol: float = 1e-3,
    ftol: float = 1e-5,
) -> dict:
    """Fit observed returns only; return estimates and optimizer diagnostics.

    ``initial_theta`` uses (mu, phi, alpha, gamma1, gamma2, sigma_eta2).
    ``bounds`` instead uses (c, phi, alpha, gamma1, gamma2, sigma_eta2),
    followed by nu when estimate_nu=True. GED nu is otherwise fixed.
    Defaults start at a symmetric model with a data-based log-variance level.
    Multiple initial values are recommended to check for local maxima.

    The objective is minus the average ABC log-likelihood (equation 19).
    Non-finite trial likelihoods receive an infinite penalty; invalid input
    is rejected before optimization. Success describes numerical convergence
    only, not statistical accuracy. History records every objective evaluation.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or y.size < 2 or not np.isfinite(y).all():
        raise ValueError("y must contain at least two finite observed returns")
    if np.var(y) <= 0:
        raise ValueError("y must have positive variance")
    if dist not in {"normal", "ged"}:
        raise ValueError("dist must be 'normal' or 'ged'")
    if estimate_nu and dist != "ged":
        raise ValueError("estimate_nu requires dist='ged'")
    if not np.isfinite(nu) or nu <= 0:
        raise ValueError("nu must be finite and positive")
    for name, value, minimum in [("n_particles", n_particles, 2),
                                  ("init_burn_in", init_burn_in, 0),
                                  ("seed", seed, 0), ("maxiter", maxiter, 1),
                                  ("maxfev", maxfev, 1)]:
        if not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not np.isfinite([xtol, ftol]).all() or min(xtol, ftol) <= 0:
        raise ValueError("xtol and ftol must be finite and positive")

    if initial_theta is None:
        initial_theta = [float(np.log(np.var(y))), .95, 0., 0., 0., .05]
    theta0 = np.asarray(initial_theta, dtype=float)
    if theta0.shape != (6,) or not np.isfinite(theta0).all():
        raise ValueError("initial_theta must contain six finite parameters")
    if abs(theta0[1]) >= 1 or theta0[5] <= 0:
        raise ValueError("initial_theta requires |phi| < 1 and sigma_eta2 > 0")
    x0 = theta0.copy()
    x0[0] *= 1 - x0[1]
    if estimate_nu:
        x0 = np.append(x0, nu)
    limits = np.asarray(bounds if bounds is not None else
                        DEFAULT_BOUNDS + ((NU_BOUNDS,) if estimate_nu else ()),
                        dtype=float)
    if (limits.shape != (len(x0), 2) or not np.isfinite(limits).all()
            or np.any(limits[:, 0] >= limits[:, 1])):
        raise ValueError("bounds must be finite ordered pairs for every optimizer parameter")
    if limits[1, 0] <= -1 or limits[1, 1] >= 1 or limits[5, 0] <= 0:
        raise ValueError("bounds must enforce |phi| < 1 and positive variance")
    if estimate_nu and limits[6, 0] <= 0:
        raise ValueError("nu bounds must be positive")
    if np.any(x0 < limits[:, 0]) or np.any(x0 > limits[:, 1]):
        raise ValueError("initial parameters are outside bounds (first bound is on c)")

    def unpack(x):
        theta = np.asarray(x[:6], dtype=float).copy()
        theta[0] /= 1 - theta[1]
        return theta, float(x[6]) if estimate_nu else nu

    history = []
    best_x, best_value = x0.copy(), np.inf

    def objective(x):
        nonlocal best_x, best_value
        theta, shape = unpack(x)
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                fit = abc_filter(y, theta, n_particles=n_particles, dist=dist,
                                 nu=shape if dist == "ged" else None, seed=seed,
                                 init_burn_in=init_burn_in, resampling="continuous")
            loglik = float(fit["loglik"])
        except (FloatingPointError, OverflowError):
            loglik = -np.inf
        if not np.isfinite(loglik):
            loglik = -np.inf

        print(f"Évaluation -> loglik: {loglik:.2f} | params: {theta}", flush=True)
        history.append(dict(zip(PARAMETER_NAMES, theta),
                            nu=shape if dist == "ged" else np.nan,
                            loglik=loglik))
        value = -loglik / len(y)
        if value < best_value:
            best_x, best_value = np.array(x, copy=True), value
        return value

    initial_objective = objective(x0)
    if not np.isfinite(initial_objective):
        raise ValueError("initial parameters give a non-finite likelihood; choose another start")
    opt = minimize(objective, x0, method="Powell", bounds=limits,
                   options=dict(maxiter=maxiter, maxfev=maxfev, xtol=xtol, ftol=ftol))
    if not np.isfinite(opt.fun):
        raise RuntimeError("optimization ended with a non-finite likelihood")
    # A bounded line search need not be unimodal for a particle likelihood.
    # Keep the best evaluated point, even if Powell ends at a worse point.
    worse_endpoint = opt.fun - best_value > ftol * max(1., abs(best_value))
    message = str(opt.message)
    if worse_endpoint:
        message += " Returned best evaluation; optimizer ended at a worse point. Retry another start or more particles."
    theta_hat, nu_hat = unpack(best_x)
    at_bounds = np.isclose(best_x, limits[:, 0], atol=xtol, rtol=0) | np.isclose(
        best_x, limits[:, 1], atol=xtol, rtol=0)
    names = ("c",) + PARAMETER_NAMES[1:] + (("nu",) if estimate_nu else ())
    return dict(theta=theta_hat, nu=nu_hat if dist == "ged" else None,
                loglik=float(-best_value * len(y)), initial_loglik=-initial_objective * len(y),
                success=bool(opt.success and not worse_endpoint), message=message, seed=int(seed),
                nfev=int(opt.nfev), nit=int(opt.nit),
                boundary_parameters=[name for name, hit in zip(names, at_bounds) if hit],
                history=pd.DataFrame(history), optimizer_result=opt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True, help="CSV with observed Return column")
    parser.add_argument("--dist", choices=["normal", "ged"], default="normal")
    parser.add_argument("--nu", type=float, default=1.5)
    parser.add_argument("--estimate-nu", action="store_true")
    parser.add_argument("--initial-theta", nargs=6, type=float, metavar=PARAMETER_NAMES)
    parser.add_argument("--n-particles", type=int, default=5000)
    parser.add_argument("--init-burn-in", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--repeats", type=int, default=1, help="Refit with consecutive filter seeds")
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--maxfev", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=Path("results/parameter_estimates.csv"))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    data = pd.read_csv(args.csv)
    if "Return" not in data:
        parser.error("CSV must contain a Return column")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for run in range(args.repeats):
        fit = estimate_parameters(
            data["Return"].to_numpy(), args.initial_theta,
            dist=args.dist, nu=args.nu, estimate_nu=args.estimate_nu,
            n_particles=args.n_particles, init_burn_in=args.init_burn_in,
            seed=args.seed + run, maxiter=args.maxiter, maxfev=args.maxfev)
        record = dict(zip(PARAMETER_NAMES, fit["theta"]),
                      nu=fit["nu"], dist=args.dist, estimate_nu=args.estimate_nu,
                      seed=fit["seed"], n_observations=len(data),
                      n_particles=args.n_particles, init_burn_in=args.init_burn_in,
                      loglik=fit["loglik"], initial_loglik=fit["initial_loglik"],
                      success=fit["success"], message=fit["message"],
                      nfev=fit["nfev"], nit=fit["nit"],
                      boundary_parameters=",".join(fit["boundary_parameters"]))
        records.append(record)
        # Save completed runs immediately, including non-converged attempts.
        pd.DataFrame(records).to_csv(args.output, index=False)
        trace = args.output.with_name(f"{args.output.stem}_seed{fit['seed']}_history.csv")
        fit["history"].to_csv(trace, index=False)
        print(f"Run {run + 1}/{args.repeats}: {record}", flush=True)
    estimates = pd.DataFrame(records)
    converged = estimates.loc[estimates.success]
    print(f"\nConverged: {len(converged)}/{len(estimates)}; saved to {args.output}")
    if len(converged) > 1:
        columns = list(PARAMETER_NAMES) + (["nu"] if args.estimate_nu else [])
        print("Across-seed variability (converged fits only; not sampling uncertainty):")
        print(converged[columns].agg(["mean", "std"]).to_string())
    if not estimates.success.all():
        parser.exit(1, "Some fits did not converge; inspect saved diagnostics before using estimates.\n")


if __name__ == "__main__":
    main()
