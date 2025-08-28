# -*- coding: utf-8 -*-
"""
core/pipeline.py
Orquestração: Greedy -> Evaluate -> (opcional) SA -> (opcional) MC.

- run_greedy_eval: roda apenas Greedy e avalia.
- run_greedy_sa_eval: usa Greedy como warm-start, roda SA e avalia melhor solução.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
import pandas as pd

from .greedy import greedy_allocation
from .evaluate import evaluate_allocation, EvalParams
from .sa import run_sa, SAParams
from .mc import run_monte_carlo as run_mc


def _normalize_inputs_minmax(df_inputs: pd.DataFrame, default_min: int = 1) -> pd.DataFrame:
    """Garante colunas eqp_min/eqp_max coerentes; força eqp_min >= default_min (tipicamente 1)."""
    df = df_inputs.copy()

    # eqp_min: se ausente/nulo/<=0 vira default_min
    if "eqp_min" not in df.columns:
        df["eqp_min"] = default_min
    else:
        df["eqp_min"] = pd.to_numeric(df["eqp_min"], errors="coerce").fillna(default_min).astype(int)
        df.loc[df["eqp_min"] < default_min, "eqp_min"] = default_min

    # eqp_max: se ausente vira teto alto
    if "eqp_max" not in df.columns:
        df["eqp_max"] = 10**9
    else:
        df["eqp_max"] = pd.to_numeric(df["eqp_max"], errors="coerce").fillna(10**9).astype(int)

    return df


def run_greedy_eval(
    df_frota: pd.DataFrame,
    df_ciclo: pd.DataFrame,
    df_inputs: pd.DataFrame,       # colunas: od_id, (eqp_min), eqp_max, w_od, mass_target_ton
    T_op_min: int,
    eqps_habilitados: Optional[List[str]] = None,
    *,
    # ---- knobs de FO ----
    beta_softcap: float = 0.1,           # β: 0=truncado, 0.1=soft, 1=linear
    underutil_penalty: float = 0.0,      # μ: penalidade por ociosidade
    overshoot_penalty_ton: float = 0.0,  # λ: penalidade por excesso (t)
    # ---- demais flags ----
    ignore_outliers_flag: bool = True,
    N_MC: int = 0,  # se > 0 roda MC
) -> Dict[str, Any]:
    # 1) Frota do dia
    df_frota_today = (
        df_frota[df_frota["eqp_id"].isin(eqps_habilitados)].copy()
        if eqps_habilitados is not None else df_frota.copy()
    )

    # 2) Inputs normalizados (obriga eqp_min >= 1)
    df_inputs_norm = _normalize_inputs_minmax(df_inputs, default_min=1)

    # 3) Pré-condição: mínimos possíveis com frota disponível?
    n_avail = int((pd.to_numeric(df_frota_today["available"], errors="coerce").fillna(0).astype(int) == 1).sum())
    min_required = int(df_inputs_norm["eqp_min"].sum())
    if min_required > n_avail:
        raise ValueError(
            f"Soma dos eqp_min ({min_required}) excede caminhões disponíveis ({n_avail}). "
            "Ajuste ODs/frota do dia ou reduza os mínimos."
        )

    # 4) Greedy
    df_alloc = greedy_allocation(
        df_frota=df_frota_today,
        df_ciclo=df_ciclo,
        df_inputs=df_inputs_norm,
        T_op_min=int(T_op_min),
        ignore_outliers_flag=bool(ignore_outliers_flag),
    )

    # 5) Evaluate (com β, μ, λ)
    eval_pkg = evaluate_allocation(
        df_alloc_in=df_alloc[["eqp_id", "od_id"]] if not df_alloc.empty else df_alloc,
        df_inputs=df_inputs_norm,
        df_frota=df_frota_today,
        df_ciclo=df_ciclo,
        T_op_min=int(T_op_min),
        params=EvalParams(
            overshoot_penalty_ton=float(overshoot_penalty_ton),
            ignore_outliers_flag=bool(ignore_outliers_flag),
            beta_softcap=float(beta_softcap),
            underutil_penalty=float(underutil_penalty),
        ),
    )

    # 6) Monte Carlo (se solicitado)
    mc_pkg = None
    if N_MC and N_MC > 0:
        mc_pkg = run_mc(
            aloc_df=df_alloc,
            df_ciclo=df_ciclo,
            df_inputs=df_inputs_norm,
            T_op_min=int(T_op_min),
            N_MC=int(N_MC),
            ignore_outliers_flag=bool(ignore_outliers_flag),
            # cycle_dist / phys_bounds / etc podem ser expostos depois
        )

    return {
        "alloc_greedy": df_alloc,
        "greedy_eval": eval_pkg,
        "mc": mc_pkg,
        "run_params": {
            "T_op_min": int(T_op_min),
            "N_MC": int(N_MC),
            "beta_softcap": float(beta_softcap),
            "underutil_penalty": float(underutil_penalty),
            "overshoot_penalty_ton": float(overshoot_penalty_ton),
            "ignore_outliers_flag": bool(ignore_outliers_flag),
        },
    }


def run_greedy_sa_eval(
    df_frota: pd.DataFrame,
    df_ciclo: pd.DataFrame,
    df_inputs: pd.DataFrame,
    T_op_min: int,
    eqps_habilitados: Optional[List[str]] = None,
    *,
    # ---- knobs de FO ----
    beta_softcap: float = 0.1,
    underutil_penalty: float = 0.0,
    overshoot_penalty_ton: float = 0.0,
    # ---- demais flags ----
    ignore_outliers_flag: bool = True,
    sa_params: Optional[SAParams] = None,
    N_MC: int = 0,  # se > 0 roda MC
) -> Dict[str, Any]:
    """
    Greedy (warm-start) -> SA -> Evaluate melhor solução -> (opcional) MC.
    """
    df_frota_today = (
        df_frota[df_frota["eqp_id"].isin(eqps_habilitados)].copy()
        if eqps_habilitados is not None else df_frota.copy()
    )

    # Inputs normalizados (obriga eqp_min >= 1)
    df_inputs_norm = _normalize_inputs_minmax(df_inputs, default_min=1)

    # Pré-condição: mínimos possíveis com frota disponível?
    n_avail = int((pd.to_numeric(df_frota_today["available"], errors="coerce").fillna(0).astype(int) == 1).sum())
    min_required = int(df_inputs_norm["eqp_min"].sum())
    if min_required > n_avail:
        raise ValueError(
            f"Soma dos eqp_min ({min_required}) excede caminhões disponíveis ({n_avail}). "
            "Ajuste ODs/frota do dia ou reduza os mínimos."
        )

    # Greedy inicial
    start_alloc = greedy_allocation(
        df_frota=df_frota_today,
        df_ciclo=df_ciclo,
        df_inputs=df_inputs_norm,
        T_op_min=int(T_op_min),
        ignore_outliers_flag=bool(ignore_outliers_flag),
    )

    # SA (inclui β, μ, λ nas params)
    sa_params = sa_params or SAParams(
        T0=1.0, Tf=1e-3, iters=2000, moves_per_iter=1,
        overshoot_penalty_ton=float(overshoot_penalty_ton),
        beta_softcap=float(beta_softcap),
        underutil_penalty=float(underutil_penalty),
        ignore_outliers_flag=bool(ignore_outliers_flag),
        random_state=42,
    )
    sa_result = run_sa(
        df_frota=df_frota_today,
        df_ciclo=df_ciclo,
        df_inputs=df_inputs_norm,
        T_op_min=int(T_op_min),
        start_alloc=start_alloc,
        params=sa_params,
    )

    # MC (sobre a melhor solução do SA)
    mc_pkg = None
    if N_MC and N_MC > 0:
        mc_pkg = run_mc(
            aloc_df=sa_result["best_alloc"],
            df_ciclo=df_ciclo,
            df_inputs=df_inputs_norm,
            T_op_min=int(T_op_min),
            N_MC=int(N_MC),
            ignore_outliers_flag=bool(ignore_outliers_flag),
        )

    return {
        "alloc_greedy": start_alloc,
        "sa_best_alloc": sa_result["best_alloc"],
        "sa_best_eval": sa_result["best_eval"],
        "sa_history": sa_result["history"],
        "mc": mc_pkg,
        "run_params": {
            "T_op_min": int(T_op_min),
            "beta_softcap": float(beta_softcap),
            "underutil_penalty": float(underutil_penalty),
            "overshoot_penalty_ton": float(overshoot_penalty_ton),
            "ignore_outliers_flag": bool(ignore_outliers_flag),
            "sa": sa_params.__dict__,
            "N_MC": int(N_MC),
        },
    }
