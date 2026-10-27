#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
simulate_tga_sv.py

Simulation du modèle TGA-SV de Mao, Czellar, Ruiz & Veiga.

Modèle
------
    y_t = exp(h_t / 2) * epsilon_t

    h_{t+1} = mu + phi * (h_t - mu)
              + alpha * I(epsilon_t < 0)
              + gamma1 * epsilon_t
              + gamma2 * (|epsilon_t| - E|epsilon_t|)
              + eta_t

avec
    eta_t ~ N(0, sigma_eta2)

et epsilon_t soit :
    - N(0,1)
    - GED(nu), centrée et standardisée à variance 1.

Paramètres Monte-Carlo du papier (cas de base)
-----------------------------------------------
    mu         = 0
    phi        = 0.98
    alpha      = 0.07
    gamma1     = -0.08
    gamma2     = 0.10
    sigma_eta2 = 0.05

Le papier considère :
    - erreurs normales ;
    - erreurs GED avec nu = 1.5.

Remarque
--------
Le papier ne précise pas explicitement, dans la description de l'expérience,
une longueur de burn-in. Ici on utilise un burn-in par défaut de 2000 pas afin
de réduire la dépendance à l'initialisation h_0 = mu. Ce choix peut être changé
avec --burn-in.

Installation
------------
    pip install numpy pandas scipy

Exemples
--------
Cas gaussien, T=500 :
    python simulate_tga_sv.py --dist normal --T 500 --seed 123

Cas GED, nu=1.5, T=500 :
    python simulate_tga_sv.py --dist ged --nu 1.5 --T 500 --seed 123
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gamma
from scipy.stats import gennorm


def ged_scale_unit_variance(nu: float) -> float:
    """
    Echelle de scipy.stats.gennorm permettant d'obtenir Var(epsilon)=1.

    Pour une generalized normal / GED de paramètre de forme nu :
        Var(X) = scale^2 * Gamma(3/nu) / Gamma(1/nu)

    Donc on choisit :
        scale = sqrt(Gamma(1/nu) / Gamma(3/nu)).
    """
    if nu <= 0:
        raise ValueError("nu doit être strictement positif.")

    return float(np.sqrt(gamma(1.0 / nu) / gamma(3.0 / nu)))


def expected_abs_epsilon(dist: str, nu: float | None = None) -> float:
    """
    Calcule E|epsilon| pour une distribution standardisée à variance 1.
    """
    dist = dist.lower()

    if dist == "normal":
        # E|Z| pour Z ~ N(0,1)
        return float(np.sqrt(2.0 / np.pi))

    if dist == "ged":
        if nu is None:
            raise ValueError("nu doit être fourni pour la GED.")

        scale = ged_scale_unit_variance(nu)

        # Pour la generalized normal :
        # E|X| = scale * Gamma(2/nu) / Gamma(1/nu)
        return float(scale * gamma(2.0 / nu) / gamma(1.0 / nu))

    raise ValueError("dist doit être 'normal' ou 'ged'.")


def draw_epsilon(
    rng: np.random.Generator,
    size: int,
    dist: str,
    nu: float | None = None,
) -> np.ndarray:
    """
    Génère epsilon_t avec moyenne 0 et variance 1.
    """
    dist = dist.lower()

    if dist == "normal":
        return rng.normal(loc=0.0, scale=1.0, size=size)

    if dist == "ged":
        if nu is None:
            raise ValueError("nu doit être fourni pour la GED.")

        scale = ged_scale_unit_variance(nu)

        return gennorm.rvs(
            beta=nu,
            loc=0.0,
            scale=scale,
            size=size,
            random_state=rng,
        )

    raise ValueError("dist doit être 'normal' ou 'ged'.")


def simulate_tga_sv(
    T: int,
    mu: float = 0.0,
    phi: float = 0.98,
    alpha: float = 0.07,
    gamma1: float = -0.08,
    gamma2: float = 0.10,
    sigma_eta2: float = 0.05,
    dist: str = "normal",
    nu: float | None = None,
    burn_in: int = 2000,
    seed: int | None = 123,
) -> pd.DataFrame:
    """
    Simule T observations d'un modèle TGA-SV.

    Parameters
    ----------
    T : int
        Nombre d'observations conservées après burn-in.
    mu : float
        Paramètre de localisation / échelle de la log-volatilité.
    phi : float
        Persistance de la log-volatilité.
    alpha : float
        Effet threshold supplémentaire lorsque epsilon_t < 0.
    gamma1 : float
        Effet signé de epsilon_t (leverage).
    gamma2 : float
        Effet de la magnitude |epsilon_t|.
    sigma_eta2 : float
        VARIANCE de eta_t, pas son écart-type.
    dist : {"normal", "ged"}
        Distribution des innovations epsilon_t.
    nu : float, optional
        Paramètre de forme de la GED. nu=2 correspond au cas gaussien.
    burn_in : int
        Nombre d'observations initiales supprimées.
    seed : int, optional
        Seed pour reproductibilité.

    Returns
    -------
    pd.DataFrame
        Colonnes :
        t, h, sigma2, sigma, epsilon, eta, f_epsilon, Return
    """
    if T <= 0:
        raise ValueError("T doit être strictement positif.")

    if burn_in < 0:
        raise ValueError("burn_in doit être positif ou nul.")

    if abs(phi) >= 1:
        raise ValueError("|phi| doit être < 1 pour le cas stationnaire.")

    if sigma_eta2 < 0:
        raise ValueError("sigma_eta2 doit être >= 0.")

    dist = dist.lower()

    if dist == "ged" and nu is None:
        raise ValueError("Pour dist='ged', fournir nu, par exemple nu=1.5.")

    rng = np.random.default_rng(seed)

    total_T = T + burn_in

    # epsilon_t est généré directement pour tous les temps.
    epsilon = draw_epsilon(
        rng=rng,
        size=total_T,
        dist=dist,
        nu=nu,
    )

    # ATTENTION : sigma_eta2 est la variance.
    sigma_eta = np.sqrt(sigma_eta2)
    eta = rng.normal(
        loc=0.0,
        scale=sigma_eta,
        size=total_T,
    )

    E_abs_eps = expected_abs_epsilon(dist=dist, nu=nu)

    # f(epsilon_t; theta)
    f_epsilon = (
        alpha * (epsilon < 0.0).astype(float)
        + gamma1 * epsilon
        + gamma2 * (np.abs(epsilon) - E_abs_eps)
    )

    # h contient h_0, ..., h_total_T.
    h = np.empty(total_T + 1, dtype=float)

    # Initialisation simple ; le burn-in réduit son influence.
    h[0] = mu

    y = np.empty(total_T, dtype=float)

    for t in range(total_T):
        # Equation d'observation :
        # y_t = exp(h_t/2) * epsilon_t
        y[t] = np.exp(h[t] / 2.0) * epsilon[t]

        # Equation d'état :
        # h_{t+1} = mu + phi(h_t-mu) + f(epsilon_t) + eta_t
        h[t + 1] = (
            mu
            + phi * (h[t] - mu)
            + f_epsilon[t]
            + eta[t]
        )

    # On enlève le burn-in.
    sl = slice(burn_in, burn_in + T)

    h_kept = h[burn_in:burn_in + T]

    df = pd.DataFrame(
        {
            "t": np.arange(1, T + 1),
            "h": h_kept,
            "sigma2": np.exp(h_kept),
            "sigma": np.exp(h_kept / 2.0),
            "epsilon": epsilon[sl],
            "eta": eta[sl],
            "f_epsilon": f_epsilon[sl],
            "Return": y[sl],
        }
    )

    return df


def print_summary(
    df: pd.DataFrame,
    *,
    mu: float,
    phi: float,
    alpha: float,
    gamma1: float,
    gamma2: float,
    sigma_eta2: float,
    dist: str,
    nu: float | None,
    burn_in: int,
    seed: int | None,
) -> None:
    """
    Affiche un résumé utile pour vérifier rapidement la simulation.
    """
    print("\nTGA-SV simulation")
    print("=" * 64)
    print(f"T             : {len(df)}")
    print(f"burn-in       : {burn_in}")
    print(f"seed          : {seed}")
    print(f"distribution  : {dist}")
    if dist == "ged":
        print(f"nu            : {nu}")

    print("\nTrue parameters")
    print("-" * 64)
    print(f"mu            : {mu}")
    print(f"phi           : {phi}")
    print(f"alpha         : {alpha}")
    print(f"gamma1        : {gamma1}")
    print(f"gamma2        : {gamma2}")
    print(f"sigma_eta^2   : {sigma_eta2}")
    print(f"sigma_eta     : {np.sqrt(sigma_eta2):.6f}")

    print("\nSimulation checks")
    print("-" * 64)
    print(f"mean(epsilon) : {df['epsilon'].mean():.6f}")
    print(f"var(epsilon)  : {df['epsilon'].var(ddof=0):.6f}")
    print(f"mean(Return)  : {df['Return'].mean():.6f}")
    print(f"var(Return)   : {df['Return'].var(ddof=0):.6f}")
    print(f"mean(h)       : {df['h'].mean():.6f}")
    print(f"mean(sigma)   : {df['sigma'].mean():.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate the TGA-SV model used in Mao et al."
    )

    parser.add_argument("--T", type=int, default=500)
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--phi", type=float, default=0.98)
    parser.add_argument("--alpha", type=float, default=0.07)
    parser.add_argument("--gamma1", type=float, default=-0.08)
    parser.add_argument("--gamma2", type=float, default=0.10)
    parser.add_argument("--sigma-eta2", type=float, default=0.05)

    parser.add_argument(
        "--dist",
        choices=["normal", "ged"],
        default="normal",
    )
    parser.add_argument("--nu", type=float, default=1.5)

    parser.add_argument("--burn-in", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Chemin du CSV de sortie.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # nu n'est utilisé que pour la GED.
    nu = args.nu if args.dist == "ged" else None

    df = simulate_tga_sv(
        T=args.T,
        mu=args.mu,
        phi=args.phi,
        alpha=args.alpha,
        gamma1=args.gamma1,
        gamma2=args.gamma2,
        sigma_eta2=args.sigma_eta2,
        dist=args.dist,
        nu=nu,
        burn_in=args.burn_in,
        seed=args.seed,
    )

    print_summary(
        df,
        mu=args.mu,
        phi=args.phi,
        alpha=args.alpha,
        gamma1=args.gamma1,
        gamma2=args.gamma2,
        sigma_eta2=args.sigma_eta2,
        dist=args.dist,
        nu=nu,
        burn_in=args.burn_in,
        seed=args.seed,
    )

    if args.output is None:
        suffix = (
            f"ged_nu{args.nu:g}"
            if args.dist == "ged"
            else "normal"
        )
        output = Path("data") / f"simulated_tga_sv_{suffix}_T{args.T}.csv"
    else:
        output = Path(args.output)

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    print(f"\nSaved to: {output}")


if __name__ == "__main__":
    main()
