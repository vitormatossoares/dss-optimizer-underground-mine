# -*- coding: utf-8 -*-
"""
core/schema.py
Validação e normalização dos DATAFRAMES de entrada do otimizador.

Este módulo NÃO lê arquivos; ele assume que os DataFrames já foram carregados
(pelo core/io.py) e com nomes de colunas padronizados:
  - fato_frota: eqp_id, cap_ton, df_pct, uf_pct, modelo, available
  - tempo_ciclo_od: eqp_id, od_id, cycle_min, mov_id

Saídas:
  - funções de validação que retornam (df_limpo, ValidationReport)
  - NÃO realiza side-effects (sem prints). O pipeline decide se aborta ou segue.

Decisões importantes p/ mineração subterrânea (documentadas):
  - Aplicamos limites físicos amplos em 'cycle_min' ANTES da detecção de outliers:
      PHYS_MIN_MIN = 3 min  (ciclos < 3 min são inconsistentes p/ subsolo)
      PHYS_MAX_MIN = 600 min (10 h; acima disso é outlier/erro de apontamento)
    → Leituras fora desse intervalo são descartadas com aviso.
  - Detecção de outliers (marcação) é ROBUSTA e hierárquica:
      1) Grupo (eqp_id, od_id) se n>=8 → IQR (Q1-3*IQR, Q3+3*IQR)
      2) Caso contrário, cai para o grupo do od_id se n>=8 → IQR
      3) Caso contrário, usa quantis globais 0,5% e 99,5% como fallback
    → Por padrão, apenas MARCAMOS (is_outlier=True) e NÃO removemos,
      para permitir análise na UI. Se desejar remover, use drop_outliers=True.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Parâmetros físicos/estatísticos (ajustáveis)
# =============================================================================

# Limites físicos/operacionais (em minutos) para filtrar erros grosseiros
PHYS_MIN_MIN: float = 3.0     # ciclos < 3 min não fazem sentido no subsolo
PHYS_MAX_MIN: float = 600.0   # ciclos > 10 h são apontamento incorreto

# Tamanho mínimo de amostra para aplicar IQR no grupo
MIN_N_IQR: int = 8

# Quantis globais de fallback quando não há amostra suficiente
FALLBACK_QLOW: float = 0.005
FALLBACK_QHI: float  = 0.995


# =============================================================================
# Utilidades de normalização e tipos
# =============================================================================

def _to_ascii_lower(s: str) -> str:
    """Remove acentos e coloca em minúsculo, mantendo apenas ASCII básico."""
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return s.lower().strip()


def _norm_id(s: str) -> str:
    """
    Normaliza identificadores (eqp_id, od_id, modelo):
    - strip
    - colapsa espaços múltiplos
    - preserva maiúsculas (IDs geralmente vêm com hifenizações específicas)
    """
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    s = str(s).strip()
    # colapsa espaços múltiplos dentro do ID
    s = re.sub(r"\s+", " ", s)
    return s


def _coerce_float(x) -> Optional[float]:
    """Converte para float com tolerância (strings com vírgula/ponto)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return np.nan
    if isinstance(x, str):
        x = x.replace(",", ".").strip()
        if x == "":
            return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan


def _coerce_pct_0_1(x) -> Optional[float]:
    """
    Converte percentuais para [0,1].
    - Aceita strings com vírgula/ponto.
    - Se >1 e <=100, assume que veio em percentuais e divide por 100.
    - Se fora de [0,1] após ajuste, retorna NaN.
    """
    v = _coerce_float(x)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return np.nan
    if v > 1.0 and v <= 100.0:
        v = v / 100.0
    if v < 0.0 or v > 1.0:
        return np.nan
    return v


def _coerce_available(x) -> Optional[int]:
    """
    Converte 'available' para {0,1}. Valores não mapeáveis => NaN.
    Aceita: 1/0, '1'/'0', 'true'/'false', 'sim'/'nao', 'yes'/'no'.
    """
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return np.nan
    s = _to_ascii_lower(str(x))
    truthy = {"1", "true", "sim", "yes", "y"}
    falsy = {"0", "false", "nao", "no", "n"}
    if s in truthy:
        return 1
    if s in falsy:
        return 0
    # tenta converter direto para int (ex.: 0/1)
    try:
        v = int(float(str(x).replace(",", ".")))
        return 1 if v != 0 else 0
    except Exception:
        return np.nan


def _canonical_mov_id(s: str) -> str:
    """
    Normaliza mov_id e retorna a versão canônica:
      - aceita apenas "producao final minerio" (sem acentos/case)
    Retornos possíveis:
      - "producao final minerio"  (aceito)
      - ""  (qualquer outro valor)
    """
    x = _to_ascii_lower(s)
    x = re.sub(r"\s+", " ", x)
    if x == "producao final minerio":
        return x
    return ""


# =============================================================================
# Relatório de validação
# =============================================================================

@dataclass
class ValidationIssue:
    level: str          # 'error' | 'warning' | 'info'
    where: str          # nome da tabela/validador
    message: str
    count: Optional[int] = None


@dataclass
class ValidationReport:
    issues: List[ValidationIssue] = field(default_factory=list)

    def add(self, level: str, where: str, message: str, count: Optional[int] = None) -> None:
        self.issues.append(ValidationIssue(level, where, message, count))

    def error(self, where: str, message: str, count: Optional[int] = None) -> None:
        self.add("error", where, message, count)

    def warn(self, where: str, message: str, count: Optional[int] = None) -> None:
        self.add("warning", where, message, count)

    def info(self, where: str, message: str, count: Optional[int] = None) -> None:
        self.add("info", where, message, count)

    @property
    def has_errors(self) -> bool:
        return any(i.level == "error" for i in self.issues)

    def summary(self) -> pd.DataFrame:
        """Retorna um DataFrame com o resumo das mensagens (útil para mostrar na UI)."""
        return pd.DataFrame([i.__dict__ for i in self.issues])


# =============================================================================
# Validação: FATO_FROTA
# =============================================================================

REQUIRED_FROTA_COLS = ["eqp_id", "cap_ton", "df_pct", "uf_pct", "modelo", "available"]


def validate_fato_frota(df_frota: pd.DataFrame, strict: bool = True) -> Tuple[pd.DataFrame, ValidationReport]:
    """
    Valida e normaliza o DataFrame 'fato_frota'.
    Regras:
      - Colunas obrigatórias presentes.
      - eqp_id normalizado e não vazio.
      - cap_ton > 0 (obrigatório). Sem fallback.
      - df_pct/uf_pct em [0,1] (aceita 0–100 => divide por 100).
      - available em {0,1}.
      - Duplicatas de eqp_id com cap_ton divergente => ERRO.

    Retorna:
      df_frota_limpo, ValidationReport
    """
    rep = ValidationReport()
    df = df_frota.copy()

    # 1) Checagem de colunas obrigatórias
    missing = [c for c in REQUIRED_FROTA_COLS if c not in df.columns]
    if missing:
        rep.error("fato_frota", f"Colunas obrigatórias ausentes: {missing}")
        return df, rep

    # 2) Normalizações básicas
    df["eqp_id"] = df["eqp_id"].apply(_norm_id)
    df["modelo"] = df["modelo"].apply(_norm_id)

    # numéricos
    df["cap_ton"] = df["cap_ton"].apply(_coerce_float)
    df["df_pct"] = df["df_pct"].apply(_coerce_pct_0_1)
    df["uf_pct"] = df["uf_pct"].apply(_coerce_pct_0_1)
    df["available"] = df["available"].apply(_coerce_available)

    # 3) Regras de valor
    # eqp_id vazio
    n_empty_ids = int((df["eqp_id"] == "").sum())
    if n_empty_ids > 0:
        rep.error("fato_frota", "Há linhas com eqp_id vazio.", n_empty_ids)

    # cap_ton > 0
    bad_cap = df["cap_ton"].isna() | (df["cap_ton"] <= 0)
    if bad_cap.any():
        rep.error("fato_frota", "cap_ton inválido (NaN ou <= 0).", int(bad_cap.sum()))

    # df_pct / uf_pct inválidos
    bad_df = df["df_pct"].isna()
    bad_uf = df["uf_pct"].isna()
    if bad_df.any():
        rep.warn("fato_frota", "df_pct inválido (fora de [0,1] ou vazio).", int(bad_df.sum()))
    if bad_uf.any():
        rep.warn("fato_frota", "uf_pct inválido (fora de [0,1] ou vazio).", int(bad_uf.sum()))

    # available inválido
    bad_av = df["available"].isna()
    if bad_av.any():
        rep.warn("fato_frota", "available inválido (não mapeável para 0/1).", int(bad_av.sum()))
    # Coerção final: NaN => 1 (assume disponível; a UI ainda controla quem roda no dia)
    df.loc[bad_av, "available"] = 1

    # 4) Duplicatas de eqp_id
    dup = df.duplicated(subset=["eqp_id"], keep=False)
    if dup.any():
        # Verifica divergência de cap_ton entre duplicatas
        diffs = (
            df.loc[dup]
            .groupby("eqp_id")["cap_ton"]
            .nunique(dropna=False)
            .reset_index(name="n_caps")
        )
        confl = diffs[diffs["n_caps"] > 1]
        if not confl.empty:
            rep.error(
                "fato_frota",
                f"Duplicatas de eqp_id com cap_ton divergente: {confl['eqp_id'].tolist()}",
                int(confl.shape[0]),
            )
        else:
            rep.warn("fato_frota", "Existem duplicatas de eqp_id (cap_ton idêntico). Serão deduplicadas.",
                     int(df.loc[dup].shape[0]))
        # Dedup (keep first)
        df = df.drop_duplicates(subset=["eqp_id"], keep="first").copy()

    # 5) Remove linhas com eqp_id vazio (bloqueante)
    if n_empty_ids > 0 and strict:
        # deixa registro no report, e retorna sem alterar
        return df, rep

    # 6) Índices e tipos finais
    df = df.reset_index(drop=True)

    return df, rep


# =============================================================================
# Validação: TEMPO_CICLO_OD
# =============================================================================

REQUIRED_CYCLE_COLS = ["eqp_id", "od_id", "cycle_min", "mov_id"]


def _flag_outliers_cycle_hybrid(df: pd.DataFrame) -> pd.Series:
    """
    Marca outliers de 'cycle_min' usando abordagem hierárquica e robusta:

    1) Preferência por grupo (eqp_id, od_id): se n>=MIN_N_IQR, usa IQR (Q1-3*IQR, Q3+3*IQR)
    2) Se o grupo (eqp_id, od_id) for pequeno, cai para o grupo do od_id (se n>=MIN_N_IQR) com IQR
    3) Se ainda for pequeno, aplica fallback por quantis globais (FALLBACK_QLOW, FALLBACK_QHI)

    Retorna uma Série booleana (True = outlier).
    """
    if df.empty:
        return pd.Series([False] * len(df), index=df.index)

    out = pd.Series(False, index=df.index)

    # quantis globais para fallback
    valid_cycles = df["cycle_min"].dropna()
    if valid_cycles.empty:
        return out
    qlow_g = valid_cycles.quantile(FALLBACK_QLOW)
    qhi_g  = valid_cycles.quantile(FALLBACK_QHI)

    # pré-computa estatísticas por od_id para fallback nível 2
    od_stats = {}
    for od_id, sub_od in df.groupby("od_id"):
        x = sub_od["cycle_min"].astype(float).dropna()
        if len(x) >= MIN_N_IQR:
            q1 = x.quantile(0.25); q3 = x.quantile(0.75); iqr = q3 - q1
            od_stats[od_id] = (q1 - 3.0 * iqr, q3 + 3.0 * iqr)
        else:
            od_stats[od_id] = None  # sinaliza p/ usar global depois

    # aplica por (eqp_id, od_id)
    for (eqp, od), sub in df.groupby(["eqp_id", "od_id"]):
        idx = sub.index
        x = sub["cycle_min"].astype(float)

        lo = hi = None
        if x.notna().sum() >= MIN_N_IQR:
            # IQR no nível (eqp_id, od_id)
            q1 = x.quantile(0.25); q3 = x.quantile(0.75); iqr = q3 - q1
            lo = q1 - 3.0 * iqr
            hi = q3 + 3.0 * iqr
        else:
            # fallback: nível do od_id (se disponível)
            od_bounds = od_stats.get(od)
            if od_bounds is not None:
                lo, hi = od_bounds
            else:
                # fallback global
                lo, hi = qlow_g, qhi_g

        mask = (x < lo) | (x > hi)
        out.loc[idx] = mask

    return out


def validate_tempo_ciclo_od(
    df_ciclo: pd.DataFrame,
    drop_outliers: bool = False,
    phys_min_min: float = PHYS_MIN_MIN,
    phys_max_min: float = PHYS_MAX_MIN,
) -> Tuple[pd.DataFrame, ValidationReport]:
    """
    Valida e normaliza o DataFrame 'tempo_ciclo_od'.

    Regras:
      - Colunas obrigatórias presentes.
      - eqp_id/od_id normalizados (strip, espaços colapsados).
      - cycle_min coerção numérica.
      - mov_id deve ser "Producao Final Minerio" (case-insensitive, sem acentos); outras linhas são descartadas.
      - Limites físicos (phys_min_min, phys_max_min) aplicados ANTES de outliers → descartados com aviso.
      - Outliers são MARCADOS (coluna bool 'is_outlier'); opcionalmente removidos se drop_outliers=True.

    Parâmetros:
      drop_outliers: se True, remove linhas marcadas como outlier.
      phys_min_min / phys_max_min: limites físicos para filtrar erros grosseiros.

    Retorna:
      df_ciclo_limpo, ValidationReport
    """
    rep = ValidationReport()
    df = df_ciclo.copy()

    # 1) Checagem de colunas obrigatórias
    missing = [c for c in REQUIRED_CYCLE_COLS if c not in df.columns]
    if missing:
        rep.error("tempo_ciclo_od", f"Colunas obrigatórias ausentes: {missing}")
        return df, rep

    # 2) Normalizações
    df["eqp_id"] = df["eqp_id"].apply(_norm_id)
    df["od_id"] = df["od_id"].apply(_norm_id)
    df["cycle_min"] = df["cycle_min"].apply(_coerce_float)
    df["mov_id_canon"] = df["mov_id"].apply(_canonical_mov_id)

    # 3) Filtrar mov_id aceito
    not_prod = df["mov_id_canon"] != "producao final minerio"
    if not_prod.any():
        rep.warn("tempo_ciclo_od", "Linhas descartadas por mov_id != 'Producao Final Minerio'.",
                 int(not_prod.sum()))
        df = df.loc[~not_prod].copy()

    # 4) Remover cycles inválidos ou fora de limites físicos
    bad_cycle = df["cycle_min"].isna() | (df["cycle_min"] <= 0)
    if bad_cycle.any():
        rep.warn("tempo_ciclo_od", "Linhas descartadas por cycle_min inválido (NaN/<=0).",
                 int(bad_cycle.sum()))
        df = df.loc[~bad_cycle].copy()

    # limites físicos (antes de outliers)
    phys_bad = (df["cycle_min"] < phys_min_min) | (df["cycle_min"] > phys_max_min)
    if phys_bad.any():
        rep.warn(
            "tempo_ciclo_od",
            f"Linhas descartadas por violar limites físicos ({phys_min_min}–{phys_max_min} min).",
            int(phys_bad.sum()),
        )
        df = df.loc[~phys_bad].copy()

    # 5) Flag de outliers (não remover por padrão)
    if not df.empty:
        df["is_outlier"] = _flag_outliers_cycle_hybrid(df)
        n_out = int(df["is_outlier"].sum())
        if n_out > 0:
            rep.info("tempo_ciclo_od", "Amostras marcadas como outliers de cycle_min (IQR híbrido).", n_out)
        if drop_outliers and n_out > 0:
            df = df.loc[~df["is_outlier"]].copy()
            rep.info("tempo_ciclo_od", "Outliers removidos do dataset após marcação.", n_out)
    else:
        df["is_outlier"] = pd.Series(dtype=bool)

    # 6) Índices e tipos finais
    df = df.drop(columns=["mov_id_canon"]).reset_index(drop=True)

    return df, rep


# =============================================================================
# Interface de verificação rápida para o pipeline
# =============================================================================

def assert_no_errors(*reports: ValidationReport) -> None:
    """
    Lança AssertionError se qualquer relatório tiver erro.
    Útil no pipeline para fail-fast ANTES de rodar o otimizador.
    """
    msgs = []
    for r in reports:
        if r.has_errors:
            df = r.summary()
            for _, row in df[df["level"] == "error"].iterrows():
                msgs.append(f"[{row['where']}] {row['message']}")
    if msgs:
        raise AssertionError("Erros de schema:\n- " + "\n- ".join(msgs))


__all__ = [
    "ValidationIssue",
    "ValidationReport",
    "validate_fato_frota",
    "validate_tempo_ciclo_od",
    "assert_no_errors",
]