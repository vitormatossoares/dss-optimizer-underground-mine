# -*- coding: utf-8 -*-
"""
core/evaluate.py
Avaliação de uma alocação (equipamento -> OD) com verificação de restrições e
cálculo de métricas/objetivo.

Mudanças principais:
- FO com "soft-cap": f(att) = att        , se att <= 1
                     f(att) = 1 + β(att-1), se att > 1      (β=beta_softcap)
  * β=0.0  -> comportamento truncado (antigo)
  * β=0.1  -> soft-cap suave (recomendado)
  * β=1.0  -> sem truncar (linear)
- Penalidade de ociosidade por caminhão (μ = underutil_penalty), proporcional à fração ociosa.
- Tabela por OD inclui:
  * throughput_tph (t/h) por truck-hour (usa ΣT_ef)
  * rate_elapsed_tph (t/h) por hora de relógio (usa T_op)
  * tempo_para_meta_h (horas corridas necessárias p/ bater a meta, via rate_elapsed_tph)
  * viagens_total (soma n_trips dos eqps)
  * viavel_no_turno (bool comparando tempo_para_meta com T_op_min)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------#
# Parâmetros de avaliação
# -----------------------------------------------------------------------------#
@dataclass
class EvalParams:
    overshoot_penalty_ton: float = 0.0    # λ: penalidade por excesso (t)
    ignore_outliers_flag: bool = True
    beta_softcap: float = 0.1             # β: soft-cap (0=truncado, 1=linear)
    underutil_penalty: float = 0.0        # μ: penalidade de ociosidade
    debug: bool = False                   # se True, retorna debug_by_od no resultado

# -----------------------------------------------------------------------------#
# Lookups de ciclo médio
# -----------------------------------------------------------------------------#
def _prepare_cycle_lookups(
    df_ciclo: pd.DataFrame,
    df_frota: pd.DataFrame,
    ignore_outliers_flag: bool = True,
) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], Dict[str, str]]:
    ciclo = df_ciclo.copy()
    if ignore_outliers_flag and "is_outlier" in ciclo.columns:
        ciclo = ciclo.loc[ciclo["is_outlier"] != True].copy()

    eqp_to_model = (
        df_frota.loc[:, ["eqp_id", "modelo"]]
        .drop_duplicates(subset=["eqp_id"])
        .set_index("eqp_id")["modelo"]
        .to_dict()
    )

    cycle_eqp_od = (
        ciclo.groupby(["eqp_id", "od_id"])["cycle_min"]
        .mean().astype(float).to_dict()
    )

    ciclo["_modelo"] = ciclo["eqp_id"].map(eqp_to_model)
    ciclo_mod = ciclo.dropna(subset=["_modelo"]).copy()
    cycle_model_od = (
        ciclo_mod.groupby(["_modelo", "od_id"])["cycle_min"]
        .mean().astype(float).to_dict()
    )

    return cycle_eqp_od, cycle_model_od, eqp_to_model

# -----------------------------------------------------------------------------#
# Recomputar produtividade (discreto/realizado)
# -----------------------------------------------------------------------------#
def _recompute_productivity(
    df_alloc: pd.DataFrame,
    df_frota: pd.DataFrame,
    cycle_eqp_od: Dict[Tuple[str, str], float],
    cycle_model_od: Dict[Tuple[str, str], float],
    eqp_to_model: Dict[str, str],
    T_op_min: int,
) -> Tuple[pd.DataFrame, List[str]]:

    alloc = df_alloc.copy()
    if not {"eqp_id", "od_id"}.issubset(alloc.columns):
        raise ValueError("df_alloc_in precisa conter ['eqp_id','od_id'].")

    f = df_frota[["eqp_id","cap_ton","available","df_pct","uf_pct"]].drop_duplicates("eqp_id")
    alloc = alloc.merge(f, on="eqp_id", how="left")

    violations, rows = [], []
    for _, row in alloc.iterrows():
        eqp, od = str(row["eqp_id"]), str(row["od_id"])
        cap = float(row["cap_ton"]) if pd.notna(row["cap_ton"]) else np.nan
        avail = int(row.get("available", 1) or 1)
        dfp = float(row.get("df_pct", 1.0) or 1.0)
        ufp = float(row.get("uf_pct", 1.0) or 1.0)

        if avail != 1 or dfp <= 0 or ufp <= 0 or not pd.notna(cap):
            violations.append(f"Eqp '{eqp}' indisponível ou DF/UF inválidos.")
            continue

        # Fallback por modelo considera a OD na chave
        c = cycle_eqp_od.get((eqp, od)) or cycle_model_od.get((eqp_to_model.get(eqp), od), None)
        if c is None or c <= 0:
            violations.append(f"Sem ciclo válido para eqp '{eqp}' em OD '{od}'.")
            continue

        T_ef = float(T_op_min) * dfp * ufp
        n_trips = int(np.floor(T_ef / c))
        if n_trips <= 0:
            continue

        rows.append({
            "eqp_id": eqp,
            "od_id": od,
            "cycle_min": float(c),
            "cap_ton": cap,
            "T_ef_min": T_ef,
            "n_trips": n_trips,
            "ton_estimado": n_trips * cap,
        })

    cols = ["eqp_id","od_id","cycle_min","cap_ton","T_ef_min","n_trips","ton_estimado"]
    return pd.DataFrame(rows, columns=cols).reset_index(drop=True), violations

# -----------------------------------------------------------------------------#
# Restrições
# -----------------------------------------------------------------------------#
def _check_constraints(alloc: pd.DataFrame, df_inputs: pd.DataFrame) -> Tuple[bool,List[str]]:
    ok, issues = True, []
    if alloc.duplicated("eqp_id").any():
        issues.append("Caminhões alocados em mais de uma OD."); ok = False

    lim_map = df_inputs.set_index("od_id")["eqp_max"].to_dict() if "eqp_max" in df_inputs else {}
    min_map = df_inputs.set_index("od_id")["eqp_min"].astype(int).to_dict() if "eqp_min" in df_inputs else {}

    for od, sub in alloc.groupby("od_id"):
        k = sub["eqp_id"].nunique()
        if k > lim_map.get(od, 1e9):
            issues.append(f"OD '{od}' excede eqp_max."); ok=False
        if k < min_map.get(od, 0):
            issues.append(f"OD '{od}' abaixo de eqp_min."); ok=False

    for od in df_inputs["od_id"].unique():
        if od not in alloc["od_id"].unique() and min_map.get(od,0)>0:
            issues.append(f"OD '{od}' sem eqps mas requer eqp_min."); ok=False

    return ok, issues

# -----------------------------------------------------------------------------#
# Avaliação completa
# -----------------------------------------------------------------------------#
def evaluate_allocation(
    df_alloc_in: pd.DataFrame,
    df_inputs: pd.DataFrame,
    df_frota: pd.DataFrame,
    df_ciclo: pd.DataFrame,
    T_op_min: int,
    params: EvalParams = EvalParams(),
) -> Dict[str, object]:

    cycle_eqp_od, cycle_model_od, eqp_to_model = _prepare_cycle_lookups(
        df_ciclo, df_frota, params.ignore_outliers_flag
    )

    alloc, violations = _recompute_productivity(
        df_alloc_in, df_frota, cycle_eqp_od, cycle_model_od, eqp_to_model, T_op_min
    )

    if alloc.empty:
        return {"alloc":alloc, "by_od":pd.DataFrame(), "objective":0.0,
                "feasible":False, "violations":violations+["Nenhuma linha válida após recompute."]}

    feasible, issues = _check_constraints(alloc, df_inputs)
    violations.extend(issues)

    tgt_map = df_inputs.set_index("od_id")["mass_target_ton"].to_dict() if "mass_target_ton" in df_inputs else {}
    w_map   = df_inputs.set_index("od_id")["w_od"].to_dict() if "w_od" in df_inputs else {}

    by_od = alloc.groupby("od_id", as_index=False).agg(
        ton_total=("ton_estimado","sum"),
        eqp_alocados=("eqp_id","nunique"),
        T_ef_total_min=("T_ef_min","sum"),
    )

    # viagens totais
    trips = alloc.groupby("od_id")["n_trips"].sum().reset_index().rename(columns={"n_trips":"viagens_total"})
    by_od = by_od.merge(trips, on="od_id", how="left")

    # targets / pesos
    by_od["target_t"] = by_od["od_id"].map(tgt_map).fillna(0.0)
    by_od["w_od"] = by_od["od_id"].map(w_map).fillna(0.0)

    # soft-cap (valor de atendimento ponderado)
    att = by_od["ton_total"]/by_od["target_t"].replace(0,np.nan)
    att = att.fillna(0.0)
    f = att.copy(); above = f>1
    if params.beta_softcap <= 0:
        f[above] = 1.0
    elif params.beta_softcap < 1:
        f[above] = 1.0 + params.beta_softcap*(f[above]-1.0)
    by_od["att_ratio"] = att
    by_od["att_value"] = f
    by_od["excesso_t"] = np.maximum(by_od["ton_total"] - by_od["target_t"], 0.0)

    # --- produtividade e tempo para meta (truck-hour x relógio) ---
    # t/h por truck-hour (diagnóstico; usa ΣT_ef)
    by_od["throughput_tph"] = np.where(
        by_od["T_ef_total_min"] > 0,
        by_od["ton_total"] / (by_od["T_ef_total_min"] / 60.0),
        0.0
    )
    # concorrência efetiva média de caminhões no turno
    k_eff = np.where(float(T_op_min) > 0, by_od["T_ef_total_min"] / float(T_op_min), 0.0)
    # t/h por hora de relógio (usa T_op)
    by_od["rate_elapsed_tph"] = by_od["throughput_tph"] * k_eff

    # tempo necessário em horas corridas para bater a meta ao ritmo realizado
    mask = (by_od["target_t"] > 0) & (by_od["rate_elapsed_tph"] > 0)
    by_od["tempo_para_meta_h"] = np.where(mask, by_od["target_t"] / by_od["rate_elapsed_tph"], 0.0)
    by_od["tempo_para_meta_min"] = by_od["tempo_para_meta_h"] * 60.0

    # viabilidade no turno: comparar com T_op_min (horas de relógio)
    by_od["viavel_no_turno"] = (by_od["tempo_para_meta_min"] <= float(T_op_min)).astype(bool)

    # penalidade de ociosidade (por caminhão)
    alloc["used_min"] = alloc["n_trips"] * alloc["cycle_min"]
    by_k = alloc.groupby("eqp_id").agg(used_min=("used_min","sum"),
                                       T_eff=("T_ef_min","max")).reset_index()
    by_k["util"] = np.where(by_k["T_eff"]>0, by_k["used_min"]/by_k["T_eff"], 0.0).clip(0,1)
    underutil_term = (1.0 - by_k["util"]).sum()

    # objetivo final
    objective = float((by_od["w_od"]*by_od["att_value"]).sum()) \
              - params.overshoot_penalty_ton*by_od["excesso_t"].sum() \
              - params.underutil_penalty*underutil_term

    by_od = by_od.sort_values(["w_od","ton_total"], ascending=[False, False]).reset_index(drop=True)

    result = {
        "alloc": alloc.reset_index(drop=True),
        "by_od": by_od[[
            "od_id","ton_total","viagens_total","eqp_alocados","target_t","w_od",
            "att_ratio","att_value","excesso_t",
            "throughput_tph","rate_elapsed_tph",
            "tempo_para_meta_h","tempo_para_meta_min","viavel_no_turno"
        ]],
        "objective": objective,
        "feasible": feasible,
        "violations": violations,
        "by_eqp_util": by_k.sort_values("util"),
        "underutil_sum": underutil_term,
    }

    if params.debug:
        result["debug_by_od"] = by_od.copy()

    return result

__all__ = ["EvalParams","evaluate_allocation"]
