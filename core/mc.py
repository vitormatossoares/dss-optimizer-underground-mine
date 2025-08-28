# -*- coding: utf-8 -*-
# core/mc.py
"""
Monte Carlo para avaliação de alocação de caminhões.

Compatível com a UI:
- Retorna producao_df, frac_df, viagens_df, tempo_df
- Se return_samples=True, retorna samples_ton, samples_frac, samples_viagens, samples_tempo (usados no violino)

Cálculo:
- Nº de viagens usa T_ef_min por eqp (ou T_op_min como fallback)
- Throughput amostrado = soma(cap_ton / ciclo) [t/min]
- Tempo para meta = meta [t] / throughput [t/min]  => minutos (e horas)
"""

from __future__ import annotations
from typing import Tuple, Literal, Optional, Dict
import numpy as np
import pandas as pd

CycleDist = Literal["empirical", "lognormal", "normal"]


def _clean_hist(
    df_ciclo: pd.DataFrame,
    od_id: str,
    eqp_id: Optional[str] = None,
    ignore_outliers_flag: bool = True,
) -> np.ndarray:
    """Série de ciclos (min) para OD (e opc eqp), limpando NaN/<=0 e outliers."""
    if eqp_id is not None:
        m = (df_ciclo["od_id"].astype(str) == str(od_id)) & (df_ciclo["eqp_id"].astype(str) == str(eqp_id))
    else:
        m = (df_ciclo["od_id"].astype(str) == str(od_id))

    ser = pd.to_numeric(df_ciclo.loc[m, "cycle_min"], errors="coerce")

    if ignore_outliers_flag and ("is_outlier" in df_ciclo.columns):
        mask_out = df_ciclo.loc[m, "is_outlier"].fillna(False).astype(bool).to_numpy(copy=True)
        ser = ser.loc[~mask_out]

    x = ser.to_numpy(dtype=float, copy=True)
    x = x[np.isfinite(x)]
    x = x[x > 0.0]
    return x


def _sample_cycle(
    rng: np.random.Generator,
    hist_eqp: np.ndarray,
    hist_od: np.ndarray,
    hist_all: np.ndarray,
    mode: CycleDist,
    phys_bounds: Tuple[float, float] = (3.0, 600.0),
    min_fit_n: int = 12,
) -> float:
    """Sorteia um ciclo (min) conforme 'mode', com fallbacks e limites físicos."""
    lo, hi = float(phys_bounds[0]), float(phys_bounds[1])

    def _bootstrap(a: np.ndarray) -> float:
        if a.size == 0:
            return np.nan
        return float(rng.choice(a, replace=True))

    def _fit_lognormal_and_sample(a: np.ndarray) -> float:
        a = a[a > 0.0]
        if a.size < min_fit_n:
            return np.nan
        loga = np.log(a)
        if loga.size < min_fit_n or not np.isfinite(loga).all():
            return np.nan
        mu = float(np.mean(loga))
        sd = float(np.std(loga, ddof=1))
        if not np.isfinite(mu) or not np.isfinite(sd) or sd <= 1e-12:
            return np.nan
        return float(rng.lognormal(mean=mu, sigma=sd))

    def _fit_normal_and_sample(a: np.ndarray) -> float:
        if a.size < min_fit_n:
            return np.nan
        m = float(np.mean(a))
        sd = float(np.std(a, ddof=1))
        if not np.isfinite(m) or not np.isfinite(sd) or sd <= 1e-12:
            return np.nan
        val = float(rng.normal(loc=m, scale=sd))
        return max(val, lo * 0.5)

    pools = [hist_eqp, hist_od, hist_all]

    if mode == "empirical":
        for p in pools:
            v = _bootstrap(p)
            if np.isfinite(v):
                return float(np.clip(v, lo, hi))
        return float(np.nan)

    fitter = _fit_lognormal_and_sample if mode == "lognormal" else _fit_normal_and_sample
    for p in pools:
        v = fitter(p)
        if np.isfinite(v):
            return float(np.clip(v, lo, hi))

    for p in pools:
        v = _bootstrap(p)
        if np.isfinite(v):
            return float(np.clip(v, lo, hi))

    return float(np.nan)


def run_monte_carlo(
    aloc_df: pd.DataFrame,
    df_ciclo: pd.DataFrame,
    df_inputs: pd.DataFrame,
    T_op_min: int,
    N_MC: int = 1000,
    random_state: int | None = None,
    *,
    cycle_dist: CycleDist = "empirical",
    ignore_outliers_flag: bool = True,
    phys_bounds: Tuple[float, float] = (3.0, 600.0),
    min_fit_n: int = 12,
    return_samples: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Simula N_MC cenários por OD:
      - Produção (t), Fração da meta, Nº de viagens, Tempo p/ meta (min/h)
    """
    rng = np.random.default_rng(random_state)

    if aloc_df is None or aloc_df.empty:
        out = {
            "producao_df": pd.DataFrame(),
            "frac_df": pd.DataFrame(),
            "viagens_df": pd.DataFrame(),
            "tempo_df": pd.DataFrame(),
        }
        if return_samples:
            out.update({
                "samples_ton": pd.DataFrame(),
                "samples_frac": pd.DataFrame(),
                "samples_viagens": pd.DataFrame(),
                "samples_tempo": pd.DataFrame(),
            })
        return out

    aloc_df = aloc_df.copy()

    # nº de ciclos por eqp (se não veio pronto, deriva de T_ef_min/cycle_min)
    if "ciclos" not in aloc_df.columns:
        if "n_trips" in aloc_df.columns:
            aloc_df["ciclos"] = pd.to_numeric(aloc_df["n_trips"], errors="coerce")
        elif {"T_ef_min", "cycle_min"}.issubset(aloc_df.columns):
            aloc_df["ciclos"] = (
                pd.to_numeric(aloc_df["T_ef_min"], errors="coerce")
                / pd.to_numeric(aloc_df["cycle_min"], errors="coerce")
            )
        else:
            raise ValueError("aloc_df precisa ter 'ciclos' ou 'n_trips' (ou T_ef_min e cycle_min).")
    aloc_df["ciclos"] = pd.to_numeric(aloc_df["ciclos"], errors="coerce").astype(float)
    aloc_df.loc[~np.isfinite(aloc_df["ciclos"]) | (aloc_df["ciclos"] <= 0), "ciclos"] = np.nan

    # capacidade (t) por eqp
    if "cap_ton" in aloc_df.columns:
        cap_series = pd.to_numeric(aloc_df["cap_ton"], errors="coerce").astype(float)
    else:
        cap_series = pd.to_numeric(aloc_df["ton_estimado"], errors="coerce").astype(float) / aloc_df["ciclos"]
    cap_map = dict(zip(aloc_df["eqp_id"].astype(str), cap_series))

    # minutos efetivos por eqp (fallback = T_op_min)
    if "T_ef_min" in aloc_df.columns:
        tef_map = dict(
            zip(
                aloc_df["eqp_id"].astype(str),
                pd.to_numeric(aloc_df["T_ef_min"], errors="coerce").astype(float),
            )
        )
    else:
        tef_map = {}

    # pool global de ciclos
    ciclo_all = pd.to_numeric(df_ciclo["cycle_min"], errors="coerce").to_numpy(dtype=float, copy=True)
    ciclo_all = ciclo_all[np.isfinite(ciclo_all)]
    ciclo_all = ciclo_all[ciclo_all > 0.0]
    if ignore_outliers_flag and ("is_outlier" in df_ciclo.columns):
        outmask_all = df_ciclo["is_outlier"].fillna(False).astype(bool).to_numpy(copy=True)
        if outmask_all.size == ciclo_all.size:
            ciclo_all = ciclo_all[~outmask_all]

    producao_res, frac_res, viagens_res, tempo_res = [], [], [], []

    if return_samples:
        samples_ton_all, samples_frac_all = [], []
        samples_viag_all, samples_tempo_all = [], []

    for od_id, grupo_od in aloc_df.groupby("od_id"):
        meta_arr = df_inputs.loc[df_inputs["od_id"] == od_id, "mass_target_ton"].values
        meta = float(meta_arr[0]) if meta_arr.size > 0 else 0.0

        eqps_od = [str(x) for x in grupo_od["eqp_id"].tolist()]
        hist_od = _clean_hist(df_ciclo, od_id=str(od_id), eqp_id=None, ignore_outliers_flag=ignore_outliers_flag)

        prod = np.empty(N_MC, dtype=float)
        frac = np.empty(N_MC, dtype=float)
        viag = np.empty(N_MC, dtype=float)
        tmeta = np.empty(N_MC, dtype=float)  # minutos

        for i in range(N_MC):
            total_ton = 0.0
            total_viagens = 0
            tpm_i = 0.0  # t/min

            for eqp_id in eqps_od:
                cap = float(cap_map.get(eqp_id, np.nan))
                if not np.isfinite(cap) or cap <= 0:
                    continue

                hist_eqp = _clean_hist(df_ciclo, od_id=str(od_id), eqp_id=str(eqp_id), ignore_outliers_flag=ignore_outliers_flag)
                ciclo = _sample_cycle(
                    rng=rng,
                    hist_eqp=hist_eqp,
                    hist_od=hist_od,
                    hist_all=ciclo_all,
                    mode=cycle_dist,
                    phys_bounds=phys_bounds,
                    min_fit_n=min_fit_n,
                )
                if not np.isfinite(ciclo) or ciclo <= 0:
                    continue

                T_eff = float(tef_map.get(eqp_id, T_op_min))
                n_ciclos = int(T_eff // ciclo)
                if n_ciclos > 0:
                    total_viagens += n_ciclos
                    total_ton += n_ciclos * cap

                tpm_i += cap / ciclo

            prod[i] = total_ton
            frac[i] = (total_ton / meta) if meta > 0 else 0.0
            viag[i] = total_viagens
            if meta > 0 and tpm_i > 0:
                tmeta[i] = meta / tpm_i
            elif meta <= 0:
                tmeta[i] = 0.0
            else:
                tmeta[i] = np.nan

        producao_res.append({
            "Frente": od_id,
            "Média": float(np.nanmean(prod)),
            "P5": float(np.nanpercentile(prod, 5)),
            "P50": float(np.nanpercentile(prod, 50)),
            "P95": float(np.nanpercentile(prod, 95)),
        })
        frac_res.append({
            "Frente": od_id,
            "Média": float(np.nanmean(frac)),
            "P5": float(np.nanpercentile(frac, 5)),
            "P50": float(np.nanpercentile(frac, 50)),
            "P95": float(np.nanpercentile(frac, 95)),
            "%>=1": float(np.nanmean(frac >= 1.0)),
        })
        viagens_res.append({
            "Frente": od_id,
            "Média": float(np.nanmean(viag)),
            "P5": float(np.nanpercentile(viag, 5)),
            "P50": float(np.nanpercentile(viag, 50)),
            "P95": float(np.nanpercentile(viag, 95)),
        })
        tempo_res.append({
            "Frente": od_id,
            "Média_h": float(np.nanmean(tmeta) / 60.0),
            "P5_h": float(np.nanpercentile(tmeta, 5) / 60.0),
            "P50_h": float(np.nanpercentile(tmeta, 50) / 60.0),
            "P95_h": float(np.nanpercentile(tmeta, 95) / 60.0),
        })

        if return_samples:
            samples_ton_all.append(pd.DataFrame({"Frente": od_id, "ton": prod}))
            samples_frac_all.append(pd.DataFrame({"Frente": od_id, "frac_meta": frac}))
            samples_viag_all.append(pd.DataFrame({"Frente": od_id, "viagens": viag}))
            samples_tempo_all.append(pd.DataFrame({"Frente": od_id, "tempo_min": tmeta}))

    result = {
        "producao_df": pd.DataFrame(producao_res),
        "frac_df": pd.DataFrame(frac_res),
        "viagens_df": pd.DataFrame(viagens_res),
        "tempo_df": pd.DataFrame(tempo_res),
    }

    if return_samples:
        result["samples_ton"] = pd.concat(samples_ton_all, ignore_index=True) if samples_ton_all else pd.DataFrame()
        result["samples_frac"] = pd.concat(samples_frac_all, ignore_index=True) if samples_frac_all else pd.DataFrame()
        result["samples_viagens"] = pd.concat(samples_viag_all, ignore_index=True) if samples_viag_all else pd.DataFrame()
        result["samples_tempo"] = pd.concat(samples_tempo_all, ignore_index=True) if samples_tempo_all else pd.DataFrame()

    return result


__all__ = ["run_monte_carlo"]
