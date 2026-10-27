#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
prepare_sp500_data.py

Reproduction des données S&P 500 utilisées dans :
Mao, Czellar, Ruiz & Veiga,
"Asymmetric Stochastic Volatility Models:
 Properties and ABC Filter-based Maximum Likelihood Estimation"

Le papier utilise :
    y_t = 100 * [log(P_t) - log(P_{t-1})]
où P_t est l'Adjusted Close du S&P 500 obtenu via Yahoo Finance.

Périodes :
- Full sample     : 1996-01-02 -> 2015-09-18, attendu : 4964 rendements
- In-sample       : 1996-01-02 -> 2005-12-30, attendu : 2519 rendements
- Out-of-sample   : 2006-01-03 -> 2015-09-18, attendu : 2445 rendements

Sorties :
- data/sp500_full.csv
- data/sp500_in_sample.csv
- data/sp500_out_of_sample.csv

Installation :
    pip install yfinance pandas numpy scipy

Exécution :
    python prepare_sp500_data.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import skew, kurtosis


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

TICKER = "^GSPC"

# On commence avant le 02/01/1996 afin de pouvoir calculer
# le rendement du 02/01/1996 à partir du cours du jour de bourse précédent.
DOWNLOAD_START = "1995-12-20"

# yfinance traite `end` comme une borne exclusive.
# 2015-09-19 permet donc d'inclure le 18/09/2015.
DOWNLOAD_END = "2015-09-19"

FULL_START = "1996-01-02"
FULL_END = "2015-09-18"

IN_SAMPLE_END = "2005-12-30"
OUT_SAMPLE_START = "2006-01-03"

OUTPUT_DIR = Path("data")


# Valeurs reportées dans la Table 2 du papier.
PAPER_STATS = {
    "Full": {
        "n": 4964,
        "mean": 0.023,
        "median": 0.063,
        "max": 10.957,
        "min": -9.470,
        "variance": 1.533,
        "skewness": -0.233,
        "kurtosis": 10.711,
    },
    "In-sample": {
        "n": 2519,
        "mean": 0.028,
        "median": 0.046,
        "max": 5.574,
        "min": -7.113,
        "variance": 1.332,
        "skewness": -0.091,
        "kurtosis": 5.956,
    },
    "Out-of-sample": {
        "n": 2445,
        "mean": 0.018,
        "median": 0.075,
        "max": 10.957,
        "min": -9.470,
        "variance": 1.737,
        "skewness": -0.328,
        "kurtosis": 13.287,
    },
}


# ---------------------------------------------------------------------
# Fonctions
# ---------------------------------------------------------------------

def download_prices() -> pd.DataFrame:
    """Télécharge les cours journaliers du S&P 500 via Yahoo Finance."""
    print(f"Downloading {TICKER} from {DOWNLOAD_START} to {DOWNLOAD_END}...")

    df = yf.download(
        TICKER,
        start=DOWNLOAD_START,
        end=DOWNLOAD_END,
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise RuntimeError("Yahoo Finance n'a renvoyé aucune donnée.")

    # Les versions récentes de yfinance peuvent renvoyer des colonnes MultiIndex.
    if isinstance(df.columns, pd.MultiIndex):
        if "Adj Close" in df.columns.get_level_values(0):
            adj_close = df["Adj Close"]
        elif "Close" in df.columns.get_level_values(0):
            print("WARNING: 'Adj Close' absent, utilisation de 'Close'.")
            adj_close = df["Close"]
        else:
            raise KeyError("Impossible de trouver 'Adj Close' ou 'Close'.")

        if isinstance(adj_close, pd.DataFrame):
            adj_close = adj_close.iloc[:, 0]

    else:
        if "Adj Close" in df.columns:
            adj_close = df["Adj Close"]
        elif "Close" in df.columns:
            print("WARNING: 'Adj Close' absent, utilisation de 'Close'.")
            adj_close = df["Close"]
        else:
            raise KeyError("Impossible de trouver 'Adj Close' ou 'Close'.")

    out = pd.DataFrame({"Adj_Close": adj_close.astype(float)})
    out.index = pd.to_datetime(out.index)
    out.index.name = "Date"
    out = out.sort_index()

    return out


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule :
        Return_t = 100 * (log(P_t) - log(P_{t-1}))

    IMPORTANT :
    ce calcul est fait AVANT de filtrer à partir du 02/01/1996.
    """
    df = df.copy()

    df["Return"] = 100.0 * np.log(df["Adj_Close"]).diff()

    # Filtrage seulement après calcul du rendement.
    df = df.loc[FULL_START:FULL_END].copy()

    # On ne garde aucune observation dont le rendement est manquant.
    df = df.dropna(subset=["Return"])

    return df


def descriptive_stats(x: pd.Series) -> dict:
    """
    Statistiques comparables à la Table 2.

    kurtosis(..., fisher=False) donne la kurtosis de Pearson :
    une loi normale a une kurtosis égale à 3.
    """
    values = x.dropna().to_numpy()

    return {
        "n": len(values),
        "mean": np.mean(values),
        "median": np.median(values),
        "max": np.max(values),
        "min": np.min(values),
        "variance": np.var(values, ddof=1),
        "skewness": skew(values, bias=False),
        "kurtosis": kurtosis(values, fisher=False, bias=False),
    }


def print_comparison(name: str, df: pd.DataFrame) -> None:
    """Compare les statistiques obtenues à celles reportées dans le papier."""
    computed = descriptive_stats(df["Return"])
    expected = PAPER_STATS[name]

    print("\n" + "=" * 72)
    print(name)
    print("=" * 72)
    print(f"{'Statistic':<15} {'Computed':>15} {'Paper':>15} {'Difference':>15}")
    print("-" * 72)

    for key in [
        "n",
        "mean",
        "median",
        "max",
        "min",
        "variance",
        "skewness",
        "kurtosis",
    ]:
        c = computed[key]
        p = expected[key]

        if key == "n":
            print(f"{key:<15} {int(c):>15d} {int(p):>15d} {int(c-p):>15d}")
        else:
            print(f"{key:<15} {c:>15.6f} {p:>15.6f} {c-p:>15.6f}")


def main() -> None:
    # 1. Téléchargement
    prices = download_prices()

    # 2. Construction des rendements
    full = compute_returns(prices)

    # 3. Même découpage que dans le papier
    in_sample = full.loc[FULL_START:IN_SAMPLE_END].copy()
    out_sample = full.loc[OUT_SAMPLE_START:FULL_END].copy()

    # 4. Contrôle des dates
    print("\nObserved periods:")
    print(
        f"Full          : {full.index.min().date()} -> "
        f"{full.index.max().date()} | n={len(full)}"
    )
    print(
        f"In-sample     : {in_sample.index.min().date()} -> "
        f"{in_sample.index.max().date()} | n={len(in_sample)}"
    )
    print(
        f"Out-of-sample : {out_sample.index.min().date()} -> "
        f"{out_sample.index.max().date()} | n={len(out_sample)}"
    )

    # 5. Comparaison aux statistiques du papier
    print_comparison("Full", full)
    print_comparison("In-sample", in_sample)
    print_comparison("Out-of-sample", out_sample)

    # 6. Sauvegarde
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    full.reset_index().to_csv(
        OUTPUT_DIR / "sp500_full.csv",
        index=False
    )
    in_sample.reset_index().to_csv(
        OUTPUT_DIR / "sp500_in_sample.csv",
        index=False
    )
    out_sample.reset_index().to_csv(
        OUTPUT_DIR / "sp500_out_of_sample.csv",
        index=False
    )

    print("\nSaved files:")
    print(f"  {OUTPUT_DIR / 'sp500_full.csv'}")
    print(f"  {OUTPUT_DIR / 'sp500_in_sample.csv'}")
    print(f"  {OUTPUT_DIR / 'sp500_out_of_sample.csv'}")

    # Contrôle simple sur la taille des échantillons.
    expected_sizes = {
        "Full": 4964,
        "In-sample": 2519,
        "Out-of-sample": 2445,
    }
    actual_sizes = {
        "Full": len(full),
        "In-sample": len(in_sample),
        "Out-of-sample": len(out_sample),
    }

    print("\nSample-size check:")
    all_ok = True
    for name, expected in expected_sizes.items():
        actual = actual_sizes[name]
        ok = actual == expected
        all_ok &= ok
        print(
            f"  {name:<15}: {actual} "
            f"(paper: {expected}) -> {'OK' if ok else 'DIFFERENT'}"
        )

    if not all_ok:
        print(
            "\nWARNING: les tailles ne correspondent pas exactement au papier.\n"
            "Cela peut venir de révisions des données historiques Yahoo Finance,\n"
            "du calendrier de cotation ou de la façon dont Yahoo fournit l'Adjusted Close."
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
