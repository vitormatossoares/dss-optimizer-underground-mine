# pages/02_Otimizador.py
# -*- coding: utf-8 -*-
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Otimizador | Alocação de Frota",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---- Gate simples de autenticação ----
if "auth" not in st.session_state:
    st.error("Acesso restrito. Faça login para continuar.")
    st.page_link("app.py", label="Voltar ao Login", icon="↩️")
    st.stop()

# ---------------- core/ path ----------------
here = Path(__file__).resolve().parent
root = here.parent if (here / "core").exists() else here.parent.parent
(core := root / "core")
(core / "__init__.py").touch(exist_ok=True)
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

# ---- Core imports ----
from core.io import load_from_data_dir
from core.pipeline import run_greedy_sa_eval
from core.sa import SAParams
from core.mc import run_monte_carlo  # Monte Carlo corrigido

# ---------------- dados ----------------
@st.cache_data(show_spinner=True)
def load_data(_data_dir: Path):
    pkg = load_from_data_dir(_data_dir)
    return pkg["df_frota"], pkg["df_ciclo"]

data_dir_default = next(
    (p for p in [root / "data/raw", here.parent / "data/raw"] if p.exists()),
    root / "data/raw",
)
with st.sidebar:
    st.header("Fonte de Dados")
    data_dir = Path(st.text_input("Pasta de dados:", str(data_dir_default)))

df_frota, df_ciclo = load_data(data_dir)

st.title("Sistema de Apoio à Decisão de Alocação de Frota de Produção de Minério")
st.caption("Parametrização do turno • Execução • Registro de resultados")

# ---------------- FO: ler parâmetros salvos na aba Configurações ----------------
beta_softcap = float(st.session_state.get("beta_softcap", 0.10))            # β (0=truncado, 1=linear)
underutil_penalty = float(st.session_state.get("underutil_penalty", 0.05))  # μ
overshoot_penalty_ton = float(st.session_state.get("overshoot_penalty_ton", 1e-4))  # λ

# ---------------- turno ----------------
st.subheader("Dados do Turno")
c1, c2, c3 = st.columns([2, 1, 1])

od_opts = sorted(df_ciclo["od_id"].dropna().astype(str).unique().tolist())
with c1:
    ods_sel = st.multiselect("Frentes (ODs) que atuarão no turno", od_opts)
with c2:
    T_op_min = st.number_input("Tempo operacional do turno (min)", min_value=60, step=30, value=720)
with c3:
    N_MC = st.number_input("Nº de simulações (Monte Carlo)", min_value=0, step=200, value=1000)

# ---------------- tabela editável por OD ----------------
if ods_sel:
    base_rows = [{
        "OD": od,
        "Meta (t)": 0.0,
        "Máx. caminhões (eqp_max)": 2,
        "Prioridade (1–5)": 1,
    } for od in ods_sel]
    df_cfg_default = pd.DataFrame(base_rows)

    st.markdown("#### Parâmetros por Frente")
    df_cfg = st.data_editor(
        df_cfg_default,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Prioridade (1–5)": st.column_config.NumberColumn(min_value=1, max_value=5, step=1),
            "Máx. caminhões (eqp_max)": st.column_config.NumberColumn(min_value=1, step=1),
            "Meta (t)": st.column_config.NumberColumn(min_value=0.0, step=10.0),
        },
    )
else:
    df_cfg = pd.DataFrame(columns=["OD", "Meta (t)", "Máx. caminhões (eqp_max)", "Prioridade (1–5)"])

# Exclusões opcionais
with st.expander("Caminhões a excluir (opcional)"):
    eqp_opts = sorted(df_frota["eqp_id"].astype(str).unique().tolist())
    eqp_excluir = st.multiselect("Excluir estes caminhões:", eqp_opts, default=[])

# ---------------- SA params ----------------
def get_sa_params_from_state() -> SAParams:
    p = st.session_state.get("sa_params", {
        "T0": 1.0, "Tf": 1e-3, "iters": 2000, "moves_per_iter": 1,
        "overshoot_penalty_ton": float(overshoot_penalty_ton),
        "ignore_outliers_flag": True, "random_state": 42
    })
    return SAParams(
        T0=float(p["T0"]), Tf=float(p["Tf"]), iters=int(p["iters"]),
        moves_per_iter=int(p["moves_per_iter"]),
        overshoot_penalty_ton=float(overshoot_penalty_ton),
        beta_softcap=float(beta_softcap),
        underutil_penalty=float(underutil_penalty),
        ignore_outliers_flag=bool(p["ignore_outliers_flag"]),
        random_state=int(p.get("random_state", 42)),
    )

# ---------------- execução ----------------
run_btn = st.button("Executar Simulação", type="primary")

@st.cache_data(show_spinner=True)
def _run_optimizer(
    df_frota, df_ciclo, df_inputs, T_op_min, sa_params,
    beta_softcap: float, underutil_penalty: float, overshoot_penalty_ton: float
):
    # N_MC=0 aqui; o MC é rodado manualmente mais abaixo com a melhor alocação
    return run_greedy_sa_eval(
        df_frota=df_frota,
        df_ciclo=df_ciclo,
        df_inputs=df_inputs,
        T_op_min=int(T_op_min),
        beta_softcap=float(beta_softcap),
        underutil_penalty=float(underutil_penalty),
        overshoot_penalty_ton=float(overshoot_penalty_ton),
        sa_params=sa_params,
        N_MC=0,
        ignore_outliers_flag=bool(sa_params.ignore_outliers_flag),
    )

if run_btn:
    # aplica exclusões
    if eqp_excluir:
        df_frota_run = df_frota.loc[~df_frota["eqp_id"].astype(str).isin(eqp_excluir)].copy()
    else:
        df_frota_run = df_frota.copy()

    if df_cfg.empty:
        st.warning("Selecione ao menos uma frente e preencha a tabela.")
        st.stop()

    # inputs para o pipeline
    df_inputs = pd.DataFrame({
        "od_id": df_cfg["OD"].astype(str),
        "eqp_min": 1,
        "eqp_max": pd.to_numeric(df_cfg["Máx. caminhões (eqp_max)"], errors="coerce").fillna(2).astype(int),
        "w_od": pd.to_numeric(df_cfg["Prioridade (1–5)"], errors="coerce").fillna(1).astype(float),
        "mass_target_ton": pd.to_numeric(df_cfg["Meta (t)"], errors="coerce").fillna(0.0).astype(float),
    })

    sa_params = get_sa_params_from_state()
    res = _run_optimizer(
        df_frota_run, df_ciclo, df_inputs, T_op_min, sa_params,
        beta_softcap, underutil_penalty, overshoot_penalty_ton
    )

    st.success("Simulação concluída.")
    st.session_state["last_result"] = res
    st.session_state["last_inputs"] = df_inputs
    st.session_state["last_Top"] = int(T_op_min)
    st.session_state["last_NMC"] = int(N_MC)

# ---------------- resultados ----------------
if "last_result" in st.session_state:
    res = st.session_state["last_result"]

    st.subheader("Sugestão de Alocação")

    al = res.get("sa_best_alloc", res.get("alloc_greedy"))
    if al is not None and not al.empty:
        lista = al.groupby("od_id")["eqp_id"].apply(lambda s: ", ".join(sorted(map(str, s)))).reset_index()
        for _, row in lista.iterrows():
            st.write(f"**{row['od_id']}**: {row['eqp_id']}")
    else:
        st.info("Nenhum equipamento alocado.")

    # ===== Tabela por OD =====
    by_od_raw = res["sa_best_eval"]["by_od"].copy() if "sa_best_eval" in res else res["greedy_eval"]["by_od"].copy()

    # renomeia para exibição
    by_od_show = by_od_raw.rename(columns={
        "od_id": "Frente (OD)",
        "ton_total": "Produção (t)",
        "viagens_total": "Viagens",
        "eqp_alocados": "Caminhões",
        "target_t": "Meta (t)",
        "w_od": "Prioridade",
        "excesso_t": "Excesso (t)",
        "tempo_para_meta_h": "Tempo p/ bater meta (h)",
        "viavel_no_turno": "Viável no turno?",
    })[
        ["Frente (OD)", "Produção (t)", "Viagens", "Caminhões",
         "Meta (t)", "Prioridade", "Excesso (t)", "Tempo p/ bater meta (h)",
         "Viável no turno?"]
    ]

    # coluna de status textual
    by_od_show.insert(
        by_od_show.columns.get_loc("Viável no turno?") + 1,
        "Status",
        by_od_show["Viável no turno?"].map(lambda b: "✅ Viável" if bool(b) else "❌ Não bate")
    )

    # estilização
    def _color_status(col):
        return [
            "background-color: #d1fae5; color: #065f46; font-weight: 700" if v == "✅ Viável"
            else "background-color: #fee2e2; color: #991b1b; font-weight: 700"
            for v in col
        ]

    styler = (
        by_od_show
        .style
        .apply(_color_status, subset=["Status"])
        .format({
            "Produção (t)": "{:,.0f}",
            "Meta (t)": "{:,.0f}",
            "Excesso (t)": "{:,.0f}",
            "Prioridade": "{:.0f}",
            "Caminhões": "{:.0f}",
            "Tempo p/ bater meta (h)": "{:,.2f}",
        })
    )

    st.markdown("#### SA — Alocação por Frente")
    st.dataframe(styler, use_container_width=True)

    # ===== Monte Carlo (violino + resumos) =====
    if al is not None and not al.empty and int(st.session_state.get("last_NMC", 0)) > 0:
        st.markdown("#### Monte Carlo — Distribuição das Simulações (Violino)")
        df_inputs_last = st.session_state.get("last_inputs")
        T_op_last = int(st.session_state.get("last_Top", 720))
        N_MC_last = int(st.session_state.get("last_NMC", 1000))

        mc_full = run_monte_carlo(
            aloc_df=al,
            df_ciclo=df_ciclo,
            df_inputs=df_inputs_last,
            T_op_min=T_op_last,
            N_MC=N_MC_last,
            random_state=42,
            return_samples=True,
        )

        metric = st.radio(
            "Métrica para o violino",
            ["Fração da meta", "Produção (t)", "Nº de viagens", "Tempo p/ meta (min)"],
            horizontal=True, index=0
        )

        if metric == "Fração da meta":
            samples, y_col, y_label = mc_full.get("samples_frac"), "frac_meta", "Fração da meta"
        elif metric == "Produção (t)":
            samples, y_col, y_label = mc_full.get("samples_ton"), "ton", "Produção (t)"
        elif metric == "Nº de viagens":
            samples, y_col, y_label = mc_full.get("samples_viagens"), "viagens", "Nº de viagens"
        else:
            samples, y_col, y_label = mc_full.get("samples_tempo"), "tempo_min", "Tempo para meta (min)"

        if samples is not None and not samples.empty:
            try:
                import plotly.graph_objects as go
                samples = samples.copy()
                samples["Frente"] = samples["Frente"].astype(str)
                ord_ods = sorted(samples["Frente"].unique().tolist())
                fig = go.Figure()
                for od in ord_ods:
                    y = samples.loc[samples["Frente"] == od, y_col]
                    fig.add_trace(go.Violin(
                        x=[od] * len(y), y=y, name=str(od),
                        box_visible=True, meanline_visible=True, points=False
                    ))
                fig.update_layout(height=420, xaxis_title="Frente (OD)", yaxis_title=y_label)
                st.plotly_chart(fig, use_container_width=True)
            except ModuleNotFoundError:
                import matplotlib.pyplot as plt
                samples = samples.copy()
                ord_ods = sorted(samples["Frente"].unique().tolist())
                fig, ax = plt.subplots(figsize=(8, 4))
                data_list = [samples.loc[samples["Frente"] == od, y_col].dropna().values for od in ord_ods]
                ax.violinplot(data_list, showmeans=True, showmedians=True, showextrema=True)
                ax.set_xticks(range(1, len(ord_ods) + 1))
                ax.set_xticklabels(ord_ods, rotation=0)
                ax.set_xlabel("Frente (OD)")
                ax.set_ylabel(y_label)
                st.pyplot(fig, use_container_width=True)

        # Resumos em grid 2×2
        st.markdown("#### Monte Carlo — Resumos")
        colA1, colA2 = st.columns(2)
        with colA1:
            st.markdown("**Fração da meta (P5, P50, P95, média)**")
            st.dataframe(mc_full["frac_df"], use_container_width=True, height=240)
        with colA2:
            st.markdown("**Produção (t) (P5, P50, P95, média)**")
            st.dataframe(mc_full["producao_df"], use_container_width=True, height=240)

        colB1, colB2 = st.columns(2)
        with colB1:
            st.markdown("**Nº de viagens (P5, P50, P95, média)**")
            st.dataframe(mc_full["viagens_df"], use_container_width=True, height=240)
        with colB2:
            st.markdown("**Tempo p/ meta (h) (P5, P50, P95, média)**")
            st.dataframe(mc_full["tempo_df"], use_container_width=True, height=240)

    # ---------------- salvar CSVs ----------------
    out_dir = root / "data" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if st.button("Aceitar alocação (salvar CSVs)"):
        (res.get("sa_best_alloc", res.get("alloc_greedy")) or pd.DataFrame()).to_csv(
            out_dir / f"alocacaoSA_{ts}.csv", index=False, encoding="utf-8-sig"
        )
        export_by_od = by_od_show.copy()
        export_by_od.to_csv(out_dir / f"tabelaSA_por_OD_{ts}.csv", index=False, encoding="utf-8-sig")
        lista = (res.get("sa_best_alloc", res.get("alloc_greedy")) or pd.DataFrame()).groupby("od_id")["eqp_id"] \
            .apply(lambda s: ", ".join(sorted(map(str, s)))).reset_index()
        if not lista.empty:
            lista.columns = ["Frente (OD)", "Caminhões"]
            lista.to_csv(out_dir / f"lista_caminhoes_por_frente_{ts}.csv", index=False, encoding="utf-8-sig")
        st.success(f"Arquivos salvos em {out_dir}")
