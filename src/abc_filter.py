#!/usr/bin/env python3
"""ABC particle filter for the TGA-SV model of Mao et al. (2016).

This file implements the filter for a FIXED parameter vector

    theta = (mu, phi, alpha, gamma1, gamma2, sigma_eta2).

It returns an ABC approximation of the log-likelihood. Parameter
optimization is intentionally left for the next step.

The paper's state is x_t = (h_{t+1}, h_t). Only h_{t+1} needs to be
carried after resampling, so this implementation stores one-dimensional
particles h_t.

The filter supports both multinomial resampling (basic ABC filter) and
continuous Malik-Pitt resampling, which the paper uses to smooth the
likelihood during maximum-likelihood optimization.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.special import gamma
from scipy.stats import gennorm


# ---------------------------------------------------------------------
# Innovation distribution
# ---------------------------------------------------------------------

def ged_scale_unit_variance(nu: float) -> float:
    """Scale for scipy.stats.gennorm so that Var(epsilon)=1."""
    if nu <= 0:
        raise ValueError("nu must be strictly positive")
    return float(np.sqrt(gamma(1.0 / nu) / gamma(3.0 / nu)))


def expected_abs_epsilon(dist: str, nu: float | None = None) -> float:
    """Return E|epsilon| for standardized innovations."""
    dist = dist.lower()
    if dist == "normal":
        return float(np.sqrt(2.0 / np.pi))
    if dist == "ged":
        if nu is None:
            raise ValueError("nu is required for GED innovations")
        s = ged_scale_unit_variance(nu)
        return float(s * gamma(2.0 / nu) / gamma(1.0 / nu))
    raise ValueError("dist must be 'normal' or 'ged'")


def draw_epsilon(
    rng: np.random.Generator,
    size: int,
    dist: str,
    nu: float | None = None,
) -> np.ndarray:
    """Draw standardized epsilon innovations."""
    dist = dist.lower()
    if dist == "normal":
        return rng.normal(0.0, 1.0, size=size)
    if dist == "ged":
        if nu is None:
            raise ValueError("nu is required for GED innovations")
        return gennorm.rvs(
            beta=nu,
            loc=0.0,
            scale=ged_scale_unit_variance(nu),
            size=size,
            random_state=rng,
        )
    raise ValueError("dist must be 'normal' or 'ged'")


# ---------------------------------------------------------------------
# TGA-SV helpers
# ---------------------------------------------------------------------

def unpack_theta(theta: Sequence[float]) -> tuple[float, ...]:
    """Return (mu, phi, alpha, gamma1, gamma2, sigma_eta2)."""
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (6,):
        raise ValueError(
            "theta must be (mu, phi, alpha, gamma1, gamma2, sigma_eta2)"
        )
    mu, phi, alpha, gamma1, gamma2, sigma_eta2 = theta
    if abs(phi) >= 1:
        raise ValueError("|phi| must be < 1")
    if sigma_eta2 <= 0:
        raise ValueError("sigma_eta2 must be > 0")
    return tuple(float(x) for x in theta)


def tga_response(
    epsilon: np.ndarray,
    alpha: float,
    gamma1: float,
    gamma2: float,
    e_abs_epsilon: float,
) -> np.ndarray:
    """f(epsilon) in the TGA-SV state equation."""
    return (
        alpha * (epsilon < 0.0).astype(float)
        + gamma1 * epsilon
        + gamma2 * (np.abs(epsilon) - e_abs_epsilon)
    )


# ---------------------------------------------------------------------
# ABC kernel
# ---------------------------------------------------------------------

def quasi_cauchy_kernel(u: np.ndarray) -> np.ndarray:
    """Univariate quasi-Cauchy kernel K(u)=(1+pi^2*u^2/4)^(-2)."""
    return (1.0 + (np.pi**2 / 4.0) * u**2) ** (-2.0)


def plugin_bandwidth(
    pseudo_y: np.ndarray,
    n_particles: int,
    min_bandwidth: float = 1e-8,
) -> float:
    """Plug-in bandwidth from equation (20) of the paper."""
    sigma_hat = float(np.std(pseudo_y, ddof=1))
    c = (5.0 * np.pi ** (9.0 / 2.0) / (48.0 * n_particles)) ** (1.0 / 5.0)
    return max(sigma_hat * c, min_bandwidth)


def abc_weights(
    observed_y: float,
    pseudo_y: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    """w_n = K_h(y - y_tilde_n) = K((y-y_tilde_n)/h)/h."""
    u = (observed_y - pseudo_y) / bandwidth
    return quasi_cauchy_kernel(u) / bandwidth


# ---------------------------------------------------------------------
# Initial state distribution
# ---------------------------------------------------------------------

def initialize_stationary_h_particles(
    n_particles: int,
    theta: Sequence[float],
    rng: np.random.Generator,
    dist: str,
    nu: float | None,
    burn_in: int = 1000,
) -> np.ndarray:
    """Approximate draws from the stationary distribution of h_t.

    The paper denotes the initial law by lambda_0 but does not specify it
    in the algorithm description. For validation we approximate it by a
    vectorized burn-in under the candidate theta.
    """
    mu, phi, alpha, gamma1, gamma2, sigma_eta2 = unpack_theta(theta)
    e_abs = expected_abs_epsilon(dist, nu)
    h = np.full(n_particles, mu, dtype=float)
    sigma_eta = np.sqrt(sigma_eta2)

    for _ in range(burn_in):
        eps = draw_epsilon(rng, n_particles, dist, nu)
        eta = rng.normal(0.0, sigma_eta, size=n_particles)
        f_eps = tga_response(eps, alpha, gamma1, gamma2, e_abs)
        h = mu + phi * (h - mu) + f_eps + eta

    return h


# ---------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------

def continuous_resampling(
    particles: np.ndarray,
    probabilities: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Continuous one-dimensional resampling of Malik & Pitt (2011).

    This implements Appendix C of Mao et al. (2016). The particles are
    first sorted. Instead of selecting an existing particle exactly, the
    algorithm samples a region and linearly interpolates between two
    adjacent sorted particles.

    Parameters
    ----------
    particles
        One-dimensional particle values to resample. In the TGA-SV
        application these are the proposed h_{t+1} particles.
    probabilities
        Normalized ABC importance weights, summing to one.
    rng
        NumPy random generator.

    Returns
    -------
    np.ndarray
        N continuously resampled particles.
    """
    particles = np.asarray(particles, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)

    if particles.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("particles and probabilities must be 1D")
    if len(particles) != len(probabilities):
        raise ValueError("particles and probabilities must have same length")

    N = len(particles)
    if N < 2:
        raise ValueError("continuous resampling requires at least 2 particles")

    # Step (i): sort particles and carry their weights along.
    order = np.argsort(particles)
    x = particles[order]
    p = probabilities[order]

    # Numerical normalization for safety.
    p = p / p.sum()

    # Step (ii): Malmquist ordered uniforms.
    # Draw v_i ~ U(0,1), then construct u_1 < ... < u_N.
    v = rng.uniform(0.0, 1.0, size=N)
    u = np.empty(N, dtype=float)

    # Paper notation: u_N = v_N^(1/N).
    u[-1] = v[-1] ** (1.0 / N)

    # For n = N-1, ..., 1: u_n = u_{n+1} v_n^(1/n).
    # Python index k corresponds to paper n=k+1.
    for k in range(N - 2, -1, -1):
        n = k + 1
        u[k] = u[k + 1] * (v[k] ** (1.0 / n))

    # Step (iii): region probabilities pi^(0), ..., pi^(N).
    pi = np.empty(N + 1, dtype=float)
    pi[0] = p[0] / 2.0
    pi[N] = p[-1] / 2.0
    pi[1:N] = (p[:-1] + p[1:]) / 2.0

    # Cumulative region probabilities. The last value should be 1.
    cdf = np.cumsum(pi)
    cdf[-1] = 1.0

    # Each ordered uniform selects one region r_j in {0,...,N}.
    regions = np.searchsorted(cdf, u, side="left")

    # Local coordinate u*_j within the selected region.
    cdf_left = np.zeros_like(u)
    mask = regions > 0
    cdf_left[mask] = cdf[regions[mask] - 1]

    local_u = (u - cdf_left) / pi[regions]
    local_u = np.clip(local_u, 0.0, 1.0)

    # Step (iv): particle drawing / interpolation.
    out = np.empty(N, dtype=float)

    left_edge = regions == 0
    right_edge = regions == N
    interior = ~(left_edge | right_edge)

    out[left_edge] = x[0]
    out[right_edge] = x[-1]

    # Region k=1,...,N-1 interpolates between x[k-1] and x[k].
    k = regions[interior]
    out[interior] = (
        x[k - 1]
        + local_u[interior] * (x[k] - x[k - 1])
    )

    return out


# ---------------------------------------------------------------------
# ABC particle filter
# ---------------------------------------------------------------------

def abc_filter(
    y: Sequence[float],
    theta: Sequence[float],
    *,
    n_particles: int = 5000,
    dist: str = "normal",
    nu: float | None = None,
    seed: int | None = 123,
    init_burn_in: int = 1000,
    store_particles: bool = False,
    resampling: str = "multinomial",
) -> dict:
    """Run the ABC particle filter for a fixed theta.

    Only observed returns y are passed to the filter. Hidden simulated
    columns such as h, epsilon, eta, sigma, etc. must NOT be supplied.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or len(y) == 0:
        raise ValueError("y must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(y)):
        raise ValueError("y contains NaN or infinite values")
    if n_particles < 2:
        raise ValueError("n_particles must be >= 2")

    resampling = resampling.lower()
    if resampling not in {"multinomial", "continuous"}:
        raise ValueError("resampling must be 'multinomial' or 'continuous'")

    mu, phi, alpha, gamma1, gamma2, sigma_eta2 = unpack_theta(theta)
    if dist == "ged" and nu is None:
        raise ValueError("nu is required when dist='ged'")

    rng = np.random.default_rng(seed)
    e_abs = expected_abs_epsilon(dist, nu)
    sigma_eta = np.sqrt(sigma_eta2)

    # Particles represent h_1 before observing y_1.
    h_particles = initialize_stationary_h_particles(
        n_particles=n_particles,
        theta=theta,
        rng=rng,
        dist=dist,
        nu=nu,
        burn_in=init_burn_in,
    )

    T = len(y)
    diagnostics = {
        "t": np.arange(1, T + 1),
        "y": y.copy(),
        "predictive_density": np.empty(T),
        "loglik_increment": np.empty(T),
        "bandwidth": np.empty(T),
        "ess": np.empty(T),
        "filtered_h_mean": np.empty(T),
        "filtered_h_sd": np.empty(T),
        "filtered_h_next_mean": np.empty(T),
        "filtered_h_next_sd": np.empty(T),
    }

    particle_history = (
        np.empty((T, n_particles), dtype=float)
        if store_particles
        else None
    )

    for t in range(T):
        # ---------------------------------------------------------
        # STEP 1: state-observation sampling
        # ---------------------------------------------------------
        eps = draw_epsilon(rng, n_particles, dist, nu)
        eta = rng.normal(0.0, sigma_eta, size=n_particles)

        # pseudo observation: y_tilde_t = exp(h_t/2) * epsilon_t
        pseudo_y = np.exp(h_particles / 2.0) * eps

        # next latent state, using the SAME epsilon_t
        f_eps = tga_response(eps, alpha, gamma1, gamma2, e_abs)
        h_next_tilde = mu + phi * (h_particles - mu) + f_eps + eta

        # ---------------------------------------------------------
        # STEP 2: correction
        # ---------------------------------------------------------
        bandwidth = plugin_bandwidth(pseudo_y, n_particles)
        weights = abc_weights(float(y[t]), pseudo_y, bandwidth)
        sum_w = float(np.sum(weights))
        if not np.isfinite(sum_w) or sum_w <= 0.0:
            raise FloatingPointError(f"Invalid weights at t={t+1}")

        p = weights / sum_w
        pred_density = sum_w / n_particles

        diagnostics["predictive_density"][t] = pred_density
        diagnostics["loglik_increment"][t] = np.log(pred_density)
        diagnostics["bandwidth"][t] = bandwidth
        diagnostics["ess"][t] = 1.0 / np.sum(p**2)

        # Posterior summary for h_t after observing y_t.
        h_mean = float(np.sum(p * h_particles))
        h_var = float(np.sum(p * (h_particles - h_mean) ** 2))
        diagnostics["filtered_h_mean"][t] = h_mean
        diagnostics["filtered_h_sd"][t] = np.sqrt(max(h_var, 0.0))

        # Posterior summary for h_{t+1} after observing y_t.
        h_next_mean = float(np.sum(p * h_next_tilde))
        h_next_var = float(np.sum(p * (h_next_tilde - h_next_mean) ** 2))
        diagnostics["filtered_h_next_mean"][t] = h_next_mean
        diagnostics["filtered_h_next_sd"][t] = np.sqrt(max(h_next_var, 0.0))

        # ---------------------------------------------------------
        # STEP 3: resampling
        # ---------------------------------------------------------
        if resampling == "multinomial":
            idx = rng.choice(
                n_particles,
                size=n_particles,
                replace=True,
                p=p,
            )
            h_particles = h_next_tilde[idx]

        else:  # continuous Malik-Pitt resampling
            h_particles = continuous_resampling(
                particles=h_next_tilde,
                probabilities=p,
                rng=rng,
            )

        if store_particles:
            particle_history[t] = h_particles

    diagnostics_df = pd.DataFrame(diagnostics)
    loglik = float(diagnostics_df["loglik_increment"].sum())

    return {
        "theta": np.asarray(theta, dtype=float),
        "dist": dist,
        "nu": nu,
        "n_particles": n_particles,
        "resampling": resampling,
        "loglik": loglik,
        "average_loglik": loglik / T,
        "diagnostics": diagnostics_df,
        "final_particles": h_particles.copy(),
        "particle_history": particle_history,
    }


# ---------------------------------------------------------------------
# Optional command-line use
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--n-particles", type=int, default=5000)
    parser.add_argument("--dist", choices=["normal", "ged"], default="normal")
    parser.add_argument("--nu", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--init-burn-in", type=int, default=1000)
    parser.add_argument(
        "--resampling",
        choices=["multinomial", "continuous"],
        default="multinomial",
    )
    parser.add_argument("--output", default="results/abc_filter_validation.csv")

    # Monte-Carlo parameter values from the paper.
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--phi", type=float, default=0.98)
    parser.add_argument("--alpha", type=float, default=0.07)
    parser.add_argument("--gamma1", type=float, default=-0.08)
    parser.add_argument("--gamma2", type=float, default=0.10)
    parser.add_argument("--sigma-eta2", type=float, default=0.05)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if "Return" not in df.columns:
        raise ValueError("CSV must contain a 'Return' column")

    # Only observed returns enter the filter.
    y = df["Return"].to_numpy(dtype=float)
    theta = np.array([
        args.mu,
        args.phi,
        args.alpha,
        args.gamma1,
        args.gamma2,
        args.sigma_eta2,
    ])

    result = abc_filter(
        y,
        theta,
        n_particles=args.n_particles,
        dist=args.dist,
        nu=args.nu if args.dist == "ged" else None,
        seed=args.seed,
        init_burn_in=args.init_burn_in,
        resampling=args.resampling,
    )

    print("\nABC particle filter")
    print("=" * 60)
    print(f"T                : {len(y)}")
    print(f"N particles      : {args.n_particles}")
    print(f"resampling       : {args.resampling}")
    print(f"log L_hat(theta) : {result['loglik']:.6f}")
    print(f"average log L    : {result['average_loglik']:.6f}")
    print(f"mean ESS         : {result['diagnostics']['ess'].mean():.1f}")
    print(f"min ESS          : {result['diagnostics']['ess'].min():.1f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result["diagnostics"].to_csv(output, index=False)
    print(f"Diagnostics saved to: {output}")


if __name__ == "__main__":
    main()
