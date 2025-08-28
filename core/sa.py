# -*- coding: utf-8 -*-
"""
core/sa.py
Simulated Annealing para melhorar a alocação inicial (warm-start).

- Espaço de solução: mapeamentos eqp_id -> od_id (ou None).
- Vizinhanças: reatribuir eqp a outra OD, desatribuir eqp, swap entre dois eqps.
- Respeita eqp_max e eqp_min (cobertura mínima) ao propor vizinhos.
- Avaliação: reaproveita core.evaluate.evaluate_allocation (objetivo a maximizar).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import random
import math
import pandas as pd

from .evaluate import evaluate_allocation, EvalParams


@dataclass
class SAParams:
    T0: float = 1.0            # temperatura inicial
    Tf: float = 1e-3           # temperatura final
    iters: int = 2000          # número de iterações
    moves_per_iter: int = 1    # quantos vizinhos por iteração
    overshoot_penalty_ton: float = 0.0
    beta_softcap: float = 0.1
    underutil_penalty: float = 0.0
    ignore_outliers_flag: bool = True
    random_state: int = 42


def _alloc_df_from_mapping(mapping: Dict[str, Optional[str]]) -> pd.DataFrame:
    """Converte mapping eqp->od em DataFrame ['eqp_id','od_id'] (sem None)."""
    rows = [{"eqp_id": e, "od_id": od} for e, od in mapping.items() if od is not None]
    return pd.DataFrame(rows, columns=["eqp_id", "od_id"]) if rows else pd.DataFrame(columns=["eqp_id", "od_id"])


def _mapping_from_alloc_df(df_alloc: pd.DataFrame, eqps: List[str]) -> Dict[str, Optional[str]]:
    """Converte DataFrame de alocação em mapping eqp->od; eqps não presentes ficam None."""
    m = {e: None for e in eqps}
    if df_alloc is not None and not df_alloc.empty:
        for _, row in df_alloc.iterrows():
            m[str(row["eqp_id"])] = str(row["od_id"])
    return m


def _capacity_ok(mapping: Dict[str, Optional[str]],
                 lim_por_od: Dict[str, int],
                 min_por_od: Dict[str, int]) -> bool:
    """Checa se cada OD respeita eqp_min <= count <= eqp_max no mapping."""
    cont: Dict[str, int] = {}
    for _, od in mapping.items():
        if od is None:
            continue
        cont[od] = cont.get(od, 0) + 1

    ods = set(lim_por_od.keys()) | set(min_por_od.keys()) | set(cont.keys())
    for od in ods:
        k = cont.get(od, 0)
        kmin = int(min_por_od.get(od, 0))
        kmax = int(lim_por_od.get(od, 10**9))
        if k < kmin or k > kmax:
            return False
    return True


def _choose_neighbor(
    mapping: Dict[str, Optional[str]],
    ods: List[str],
    lim_por_od: Dict[str, int],
    min_por_od: Dict[str, int],
) -> Dict[str, Optional[str]]:
    """
    Gera vizinho respeitando eqp_min/eqp_max:
      - reassign: move e mantém capacidade válida
      - unassign: só se não quebrar eqp_min do OD atual
      - swap: troca e mantém capacidade válida
    """
    new_map = dict(mapping)
    eqps = list(new_map.keys())
    if not eqps:
        return new_map

    move_type = random.choices(["reassign", "unassign", "swap"], weights=[0.5, 0.2, 0.3], k=1)[0]

    # contagem atual por OD
    counts: Dict[str, int] = {}
    for _, od in new_map.items():
        if od is None:
            continue
        counts[od] = counts.get(od, 0) + 1

    if move_type == "reassign":
        e = random.choice(eqps)
        cur_od = new_map[e]
        cand_ods = [od for od in ods if od != cur_od]
        if not cand_ods:
            return new_map
        od_new = random.choice(cand_ods)

        # simula remoção de cur_od (se houver) e checa eqp_min
        if cur_od is not None:
            after_cur = counts.get(cur_od, 0) - 1
            if after_cur < int(min_por_od.get(cur_od, 0)):
                return new_map  # violaria mínimo da OD atual

        # simula adição em od_new e checa eqp_max
        after_new = counts.get(od_new, 0) + 1
        if after_new > int(lim_por_od.get(od_new, 10**9)):
            return new_map  # estoura máximo

        new_map[e] = od_new
        return new_map

    if move_type == "unassign":
        e = random.choice(eqps)
        cur_od = new_map[e]
        if cur_od is None:
            return new_map
        # só deixar sair se não quebrar eqp_min
        if (counts.get(cur_od, 0) - 1) < int(min_por_od.get(cur_od, 0)):
            return new_map
        new_map[e] = None
        return new_map

    # swap
    if len(eqps) >= 2:
        e1, e2 = random.sample(eqps, 2)
        od1, od2 = new_map[e1], new_map[e2]
        new_map[e1], new_map[e2] = od2, od1
        if _capacity_ok(new_map, lim_por_od, min_por_od):
            return new_map
        # reverte se inválido
        new_map[e1], new_map[e2] = od1, od2
        return new_map

    return new_map


def run_sa(
    df_frota: pd.DataFrame,
    df_ciclo: pd.DataFrame,
    df_inputs: pd.DataFrame,           # colunas: od_id, eqp_max, w_od, mass_target_ton, (eqp_min opcional)
    T_op_min: int,
    start_alloc: Optional[pd.DataFrame] = None,   # opcional: DataFrame eqp_id, od_id (ex.: greedy)
    params: SAParams = SAParams(),
) -> Dict[str, object]:
    """
    Executa Simulated Annealing partindo de uma alocação inicial (se fornecida).
    Retorna:
      {
        "best_alloc": DataFrame (eqp_id, od_id, ciclos, ton_estimado),
        "best_eval":  pacote do evaluate_allocation,
        "history":    lista de dicts com evolução [{iter, T, obj, best_obj}, ...]
      }
    """
    random.seed(params.random_state)

    # considera apenas eqps disponíveis (opcional, melhora o espaço de busca)
    eqps = [str(x) for x in df_frota.loc[df_frota["available"] == 1, "eqp_id"].unique().tolist()]
    ods = [str(x) for x in df_inputs["od_id"].unique().tolist()]
    lim_por_od = df_inputs.set_index("od_id")["eqp_max"].astype(int).to_dict()
    min_por_od = df_inputs["eqp_min"].fillna(0).astype(int).groupby(df_inputs["od_id"]).max().to_dict() \
                 if "eqp_min" in df_inputs.columns else {}

    # mapeamento inicial
    cur_map = _mapping_from_alloc_df(start_alloc, eqps) if start_alloc is not None else {e: None for e in eqps}

    # avalia estado inicial
    cur_df = _alloc_df_from_mapping(cur_map)
    cur_eval = evaluate_allocation(
        df_alloc_in=cur_df,
        df_inputs=df_inputs,
        df_frota=df_frota,
        df_ciclo=df_ciclo,
        T_op_min=T_op_min,
        params=EvalParams(
            overshoot_penalty_ton=params.overshoot_penalty_ton,
            beta_softcap=params.beta_softcap,
            underutil_penalty=params.underutil_penalty,
            ignore_outliers_flag=params.ignore_outliers_flag,
        ),
    )
    cur_obj = float(cur_eval["objective"])

    best_map = dict(cur_map)
    best_eval = cur_eval
    best_obj = cur_obj

    # agenda de resfriamento
    alpha = (params.Tf / params.T0) ** (1.0 / max(params.iters, 1))
    T = params.T0

    history = [{"iter": 0, "T": T, "obj": cur_obj, "best_obj": best_obj}]

    for it in range(1, params.iters + 1):
        cand_map = dict(cur_map)
        for _ in range(params.moves_per_iter):
            cand_map = _choose_neighbor(cand_map, ods, lim_por_od, min_por_od)

        cand_df = _alloc_df_from_mapping(cand_map)
        cand_eval = evaluate_allocation(
            df_alloc_in=cand_df,
            df_inputs=df_inputs,
            df_frota=df_frota,
            df_ciclo=df_ciclo,
            T_op_min=T_op_min,
            params=EvalParams(
                overshoot_penalty_ton=params.overshoot_penalty_ton,
                beta_softcap=params.beta_softcap,
                underutil_penalty=params.underutil_penalty,
                ignore_outliers_flag=params.ignore_outliers_flag,
            ),
        )
        cand_obj = float(cand_eval["objective"])

        # regra de aceitação (maximização)
        accept = False
        if cand_obj >= cur_obj:
            accept = True
        else:
            delta = cand_obj - cur_obj
            prob = math.exp(delta / max(T, 1e-12))
            if random.random() < prob:
                accept = True

        if accept:
            cur_map = cand_map
            cur_eval = cand_eval
            cur_obj = cand_obj

            if cand_obj > best_obj:
                best_map = cand_map
                best_eval = cand_eval
                best_obj = cand_obj

        T = T * alpha
        history.append({"iter": it, "T": T, "obj": cur_obj, "best_obj": best_obj})

    # monta DataFrame final a partir do melhor mapping (evaluate já recalcula ciclos/ton)
    best_alloc_df = best_eval["alloc"].copy()

    return {
        "best_alloc": best_alloc_df,
        "best_eval": best_eval,
        "history": history,
    }