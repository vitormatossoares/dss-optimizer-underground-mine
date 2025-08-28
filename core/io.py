# -*- coding: utf-8 -*-
"""
core/io.py
Leitura e normalização de arquivos de dados (CSV/XLSX) para o otimizador.
- Detecta CSV ou Excel automaticamente
- Faz mapeamento de cabeçalhos reais -> nomes canônicos
- Chama os validadores de core/schema.py
- Retorna DataFrames prontos + relatórios de validação + colmap (mapeamentos usados)

Arquivos esperados (em data/raw/, por padrão):
- fato_frota.(csv|xlsx)
- tempo_ciclo_od.(csv|xlsx)
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, Optional

import pandas as pd

from .schema import (
    validate_fato_frota,
    validate_tempo_ciclo_od,
    ValidationReport,
)

# ---------------------------------------------------------------------
# Sinônimos de cabeçalhos -> canônicos
# ---------------------------------------------------------------------
SYN_FROTA: Dict[str, tuple] = {
    "eqp_id":   ("eqp_id", "equipamento", "eqp", "id_equipamento"),
    "cap_ton":  ("cap_ton", "capacidade", "capacidade_ton", "capacidade (t)"),
    "df_pct":   ("df_pct", "disponibilidade_fisica", "disponibilidade", "df"),
    "uf_pct":   ("uf_pct", "utilizacao_fisica", "utilizacao", "uf"),
    "modelo":   ("modelo", "model", "modelo_eqp"),
    "available":("available", "disponivel", "habilitado", "ativo"),
}

SYN_CICLO: Dict[str, tuple] = {
    "eqp_id":    ("eqp_id", "equipamento", "eqp", "id_equipamento"),
    "od_id":     ("od_id", "origem - destino", "origem_destino", "od", "frente"),
    "cycle_min": ("cycle_min", "tempo_ciclo", "tempo de ciclo (min)", "min_ciclo"),
    "mov_id":    ("mov_id", "mov_type", "tipo_movimentacao", "tipo", "tipo de movimento"),
}

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _read_any(path: Path, sheet: Optional[str|int]=None) -> pd.DataFrame:
    """Lê CSV ou XLSX com opções seguras (detecta separador no CSV)."""
    ext = path.suffix.lower()
    if ext in (".csv", ".txt"):
        # tenta vírgula, ponto e vírgula, tab
        for sep in [",", ";", "\t"]:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] > 1:  # separou certo
                return df
        # fallback: deixa o pandas decidir
        return pd.read_csv(path, sep=None, engine="python")
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path, sheet_name=sheet if sheet is not None else 0)
    raise ValueError(f"Extensão não suportada: {ext} ({path.name})")

def _auto_rename(df: pd.DataFrame, synonyms: Dict[str, tuple]) -> Tuple[pd.DataFrame, Dict[str,str]]:
    """
    Renomeia colunas do DF para nomes canônicos, usando sinônimos.
    Retorna (df_renomeado, colmap) onde colmap = {original -> canônico}.
    """
    cols_lower = {c.lower().strip(): c for c in df.columns}
    colmap: Dict[str, str] = {}
    for canon, syns in synonyms.items():
        found = None
        for s in syns:
            key = s.lower().strip()
            if key in cols_lower:
                found = cols_lower[key]
                break
        if found is not None:
            colmap[found] = canon
    if colmap:
        df = df.rename(columns=colmap)
    return df, colmap

# ---------------------------------------------------------------------
# Leitores específicos
# ---------------------------------------------------------------------
def load_fato_frota(path: str|Path) -> Tuple[pd.DataFrame, ValidationReport, Dict[str,str]]:
    """
    Lê e valida 'fato_frota' (capacidade por caminhão, df_pct, uf_pct, etc.).
    - Usa cap_ton EXATAMENTE como na planilha (sem fallback).
    Retorna (df_limpo, report, colmap_usado).
    """
    p = Path(path)
    raw = _read_any(p)
    norm, colmap = _auto_rename(raw, SYN_FROTA)
    df, report = validate_fato_frota(norm, strict=True)
    return df, report, colmap

def load_tempo_ciclo_od(path: str|Path) -> Tuple[pd.DataFrame, ValidationReport, Dict[str,str]]:
    """
    Lê e valida 'tempo_ciclo_od' (histórico de ciclos por (eqp_id, od_id)).
    - Mantém apenas mov_id == 'Producao Final Minerio' (schema faz o filtro).
    - Marca outliers; NÃO remove.
    Retorna (df_limpo, report, colmap_usado).
    """
    p = Path(path)
    raw = _read_any(p)
    norm, colmap = _auto_rename(raw, SYN_CICLO)
    df, report = validate_tempo_ciclo_od(norm)
    return df, report, colmap

# ---------------------------------------------------------------------
# Loader geral (busca por nomes padrão em data/raw/)
# ---------------------------------------------------------------------
def load_from_data_dir(data_dir: str|Path="data/raw") -> Dict[str, object]:
    """
    Procura por 'fato_frota' e 'tempo_ciclo_od' (CSV ou XLSX) dentro de data_dir.
    Retorna um pacote:
      {
        "df_frota": DataFrame,
        "df_ciclo": DataFrame,
        "report_frota": ValidationReport,
        "report_ciclo": ValidationReport,
        "colmap_frota": dict,
        "colmap_ciclo": dict,
        "paths": {"fato_frota": Path, "tempo_ciclo_od": Path}
      }
    Lança FileNotFoundError se algum dos arquivos não for encontrado.
    """
    data_dir = Path(data_dir)
    def _find(prefix: str) -> Path:
        for ext in (".csv", ".xlsx", ".xls"):
            cand = data_dir / f"{prefix}{ext}"
            if cand.exists():
                return cand
        raise FileNotFoundError(f"Arquivo '{prefix}.csv|.xlsx' não encontrado em {data_dir}.")

    p_frota = _find("fato_frota")
    p_ciclo = _find("tempo_ciclo_od")

    df_frota, rep_frota, cmap_frota = load_fato_frota(p_frota)
    df_ciclo, rep_ciclo, cmap_ciclo = load_tempo_ciclo_od(p_ciclo)

    return {
        "df_frota": df_frota,
        "df_ciclo": df_ciclo,
        "report_frota": rep_frota,
        "report_ciclo": rep_ciclo,
        "colmap_frota": cmap_frota,
        "colmap_ciclo": cmap_ciclo,
        "paths": {"fato_frota": p_frota, "tempo_ciclo_od": p_ciclo},
    }