# -*- coding: utf-8 -*-
# core/greedy.py
"""
Heurística gulosa para solução inicial (warm-start).
"""

from __future__ import annotations
import pandas as pd


def greedy_allocation(
    df_frota: pd.DataFrame,
    df_ciclo: pd.DataFrame,
    df_inputs: pd.DataFrame,
    T_op_min: int,
    ignore_outliers_flag: bool = False,  # opcional: alinhar com evaluate/pipeline
) -> pd.DataFrame:
    """
    Constrói uma alocação inicial viável usando heurística gulosa.

    Parâmetros
    ----------
    df_frota : DataFrame
        Tabela de frota (eqp_id, modelo, cap_ton, available, df_pct, uf_pct).
    df_ciclo : DataFrame
        Tabela de tempos de ciclo (od_id, eqp_id, cycle_min[, is_outlier]).
    df_inputs : DataFrame
        Entradas do operador: od_id, eqp_max, w_od, mass_target_ton.
    T_op_min : int
        Tempo operacional do dia (min).
    ignore_outliers_flag : bool
        Se True e existir 'is_outlier' em df_ciclo, ignora essas linhas.

    Retorna
    -------
    DataFrame com alocação: eqp_id, od_id, ciclos, ton_estimado.
    """

    # ---------------- Sane checks mínimos ----------------
    # Garante colunas necessárias (erros claros ajudam no setup)
    req_frota = {"eqp_id", "cap_ton", "available"}
    req_inputs = {"od_id", "eqp_max", "w_od", "mass_target_ton"}
    req_ciclo = {"od_id", "eqp_id", "cycle_min"}

    if not req_frota.issubset(df_frota.columns):
        raise ValueError(f"df_frota precisa conter colunas: {sorted(req_frota)}")
    if not req_inputs.issubset(df_inputs.columns):
        raise ValueError(f"df_inputs precisa conter colunas: {sorted(req_inputs)}")
    if not req_ciclo.issubset(df_ciclo.columns):
        raise ValueError(f"df_ciclo precisa conter colunas: {sorted(req_ciclo)}")

    # 0. Preparar df_ciclo (opcionalmente ignorando outliers)
    ciclos = df_ciclo.copy()
    if ignore_outliers_flag and "is_outlier" in ciclos.columns:
        ciclos = ciclos.loc[ciclos["is_outlier"] != True].copy()
    # remover ciclos inválidos
    ciclos = ciclos[pd.to_numeric(ciclos["cycle_min"], errors="coerce").notna()].copy()
    ciclos = ciclos[ciclos["cycle_min"].astype(float) > 0].copy()

    # 1. Filtrar apenas caminhões disponíveis
    frota_disp = df_frota[df_frota["available"] == 1].copy()

    # Score = df_pct * uf_pct  (representa "quanto o caminhão rende")
    # (se não existirem, assume 1.0 para não penalizar indevidamente)
    dfp = frota_disp["df_pct"] if "df_pct" in frota_disp.columns else 1.0
    ufp = frota_disp["uf_pct"] if "uf_pct" in frota_disp.columns else 1.0
    frota_disp["score"] = pd.to_numeric(dfp, errors="coerce").fillna(0) * \
                          pd.to_numeric(ufp, errors="coerce").fillna(0)
    frota_disp = frota_disp.sort_values("score", ascending=False)

    # 2. Ordenar ODs por prioridade (peso w_od)
    ods_sorted = df_inputs.sort_values("w_od", ascending=False).copy()

    alocacoes = []

    # 3. Iterar sobre cada OD, tentando alocar caminhões
    for _, od in ods_sorted.iterrows():
        od_id = od["od_id"]
        max_cam = int(od["eqp_max"])
        mass_target = float(od["mass_target_ton"])

        # subset de ciclos dessa OD
        ciclos_od = ciclos[ciclos["od_id"] == od_id]

        # candidatos ainda não alocados
        ja_alocados = {a["eqp_id"] for a in alocacoes}
        candidatos = frota_disp[~frota_disp["eqp_id"].isin(ja_alocados)]
        count = 0

        for _, eqp in candidatos.iterrows():
            if count >= max_cam:
                break

            eqp_id = eqp["eqp_id"]
            modelo = eqp.get("modelo", None)  # pode não existir
            cap = float(pd.to_numeric(eqp["cap_ton"], errors="coerce"))

            # 3.1 ciclo do eqp_id
            row_eqp = ciclos_od[ciclos_od["eqp_id"] == eqp_id]
            cycle_min = None
            if not row_eqp.empty:
                cycle_min = float(row_eqp["cycle_min"].mean())
            # 3.2 senão: ciclo médio de qualquer eqp do mesmo modelo
            elif (modelo is not None) and ("modelo" in frota_disp.columns):
                eqps_mesmo_modelo = frota_disp[frota_disp["modelo"] == modelo]["eqp_id"].tolist()
                row_modelo = ciclos_od[ciclos_od["eqp_id"].isin(eqps_mesmo_modelo)]
                if not row_modelo.empty:
                    cycle_min = float(row_modelo["cycle_min"].mean())

            # sem ciclo válido, pula
            if (cycle_min is None) or (cycle_min <= 0):
                continue

            # produtividade = nº de ciclos * capacidade (sem floor/DF/UF aqui)
            ciclos_estimados = float(T_op_min) / float(cycle_min)
            if ciclos_estimados <= 0:
                continue
            ton_estimado = ciclos_estimados * cap

            alocacoes.append(
                {
                    "eqp_id": eqp_id,
                    "od_id": od_id,
                    "ciclos": ciclos_estimados,
                    "ton_estimado": ton_estimado,
                }
            )
            count += 1

            # parar se massa alvo da OD já atingida
            massa_alocada = sum(a["ton_estimado"] for a in alocacoes if a["od_id"] == od_id)
            if massa_alocada >= mass_target:
                break

    # 🔒 robustez: sempre retornar colunas esperadas, mesmo vazio
    df_out = pd.DataFrame(alocacoes)
    if df_out.empty:
        df_out = pd.DataFrame(columns=["eqp_id", "od_id", "ciclos", "ton_estimado"])
    return df_out
