# pages/01_Validacao_de_Dados.py
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import sys

# =================== Gate de login ===================
st.set_page_config(page_title="Validação de Dados | Alocação de Frota", layout="wide", initial_sidebar_state="collapsed")
if "auth" not in st.session_state:
    st.error("Acesso restrito. Faça login para continuar.")
    st.page_link("app.py", label="Voltar ao Login", icon="↩️")
    st.stop()

# =================== Setup / imports core ===================
here = Path(__file__).resolve().parent
root = here.parent if (here / "core").exists() else here.parent.parent
(core := root / "core").mkdir(exist_ok=True)
(core / "__init__.py").touch(exist_ok=True)
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from core.io import load_from_data_dir   # type: ignore

# =================== UI topo ===================
st.title("Validação e Análise Exploratória dos Dados")
st.caption(f"Usuário: **{st.session_state['auth']['usuario']}**")

# =================== Carregamento ===================
@st.cache_data(show_spinner=True)
def load_data(_data_dir: Path):
    pkg = load_from_data_dir(_data_dir)
    return pkg["df_frota"], pkg["df_ciclo"], pkg["report_frota"].summary(), pkg["report_ciclo"].summary()

data_dir_default = next((p for p in [root/"data/raw", here.parent/"data/raw"] if p.exists()), root/"data/raw")
with st.sidebar:
    st.header("Carregamento")
    data_dir = Path(st.text_input("Pasta de dados:", str(data_dir_default)))
    ignore_outliers = st.checkbox("Ignorar outliers de ciclo", value=True)
    bins = st.slider("Bins (histograma)", 5, 80, 30)

df_frota, df_ciclo, rep_frota_df, rep_ciclo_df = load_data(data_dir)
ciclos = df_ciclo.copy()
if ignore_outliers and "is_outlier" in ciclos.columns:
    ciclos = ciclos.loc[ciclos["is_outlier"] != True].copy()

# =================== KPIs ===================
k1,k2,k3,k4 = st.columns(4)
k1.metric("Caminhões únicos", f"{df_frota['eqp_id'].nunique()}")
k2.metric("Frentes (OD)", f"{ciclos['od_id'].nunique()}")
k3.metric("Registros de ciclo", f"{len(ciclos)}")
k4.metric("Outliers marcados (base)", f"{(df_ciclo.get('is_outlier', pd.Series(False,index=df_ciclo.index)).mean()*100):.1f}%")

with st.expander("Relatórios de validação (schema)"):
    c1,c2 = st.columns(2)
    c1.dataframe(rep_frota_df if not rep_frota_df.empty else pd.DataFrame({"info":["Sem avisos/erros na frota."]}), use_container_width=True)
    c2.dataframe(rep_ciclo_df if not rep_ciclo_df.empty else pd.DataFrame({"info":["Sem avisos/erros no tempo de ciclo."]}), use_container_width=True)

# =================== Filtros ===================
od_opts = sorted(ciclos["od_id"].astype(str).unique())
eqp_opts = sorted(df_frota["eqp_id"].astype(str).unique())
modelo_opts = sorted(df_frota["modelo"].dropna().astype(str).unique())

fc1,fc2,fc3 = st.columns(3)
sel_ods = fc1.multiselect("Filtrar ODs", od_opts, default=od_opts[:min(6,len(od_opts))])
sel_eqps = fc2.multiselect("Filtrar Caminhões", eqp_opts, default=[])
sel_modelos = fc3.multiselect("Filtrar Modelo", modelo_opts, default=[])

mask = ciclos["od_id"].astype(str).isin(sel_ods) if sel_ods else np.ones(len(ciclos), bool)
if sel_eqps: mask &= ciclos["eqp_id"].astype(str).isin(sel_eqps)
if sel_modelos:
    modelo_map = df_frota.set_index("eqp_id")["modelo"].astype(str).to_dict()
    mask &= ciclos["eqp_id"].map(lambda x: str(modelo_map.get(x,"")) in set(sel_modelos))
ciclos_f = ciclos.loc[mask].merge(df_frota[["eqp_id","modelo"]], on="eqp_id", how="left")

# =================== Distribuições de ciclo ===================
st.markdown("### Distribuições do Tempo de Ciclo (min)")
c1,c2 = st.columns([2,1])
with c1:
    if ciclos_f.empty:
        st.warning("Sem dados após filtros.")
    else:
        fig_hist = px.histogram(
            ciclos_f, x="cycle_min", nbins=bins, color="od_id",
            barmode="overlay", opacity=0.75,
            labels={"cycle_min":"Tempo de ciclo (min)","od_id":"OD"}
        )
        fig_hist.update_layout(height=420, legend_title_text="OD")
        st.plotly_chart(fig_hist, use_container_width=True)
with c2:
    stats = ciclos_f.groupby("od_id")["cycle_min"].agg(["count","mean","median","std","min","max"]).reset_index()
    stats.rename(columns={"od_id":"OD"}, inplace=True)
    st.dataframe(stats, use_container_width=True)

# =================== Boxplots ===================
st.markdown("### Boxplots")
b1,b2 = st.columns(2)
if not ciclos_f.empty:
    fig_box_od = px.box(ciclos_f, x="od_id", y="cycle_min", points="outliers",
                        labels={"od_id":"OD","cycle_min":"Tempo de ciclo (min)"})
    fig_box_od.update_layout(height=420)
    b1.plotly_chart(fig_box_od, use_container_width=True)

    fig_box_model = px.box(ciclos_f.dropna(subset=["modelo"]), x="modelo", y="cycle_min", points="outliers",
                           labels={"modelo":"Modelo","cycle_min":"Tempo de ciclo (min)"})
    fig_box_model.update_layout(height=420)
    b2.plotly_chart(fig_box_model, use_container_width=True)

# =================== Heatmap com anotações (✅ pedido)
st.markdown("### Heatmap — Tempo médio de ciclo (OD × Caminhão)")
if not ciclos_f.empty:
    piv = ciclos_f.pivot_table(index="od_id", columns="eqp_id", values="cycle_min", aggfunc="mean")
    if not piv.empty:
        z = np.round(piv.values.astype(float), 1)
        fig_heat = go.Figure(data=go.Heatmap(
            z=z, x=piv.columns.astype(str), y=piv.index.astype(str),
            colorscale="Blues", colorbar_title="Ciclo (min)",
            hovertemplate="Caminhão: %{x}<br>OD: %{y}<br>Ciclo médio: %{z} min<extra></extra>",
            zmin=np.nanmin(z), zmax=np.nanmax(z),
        ))
        # anotações em branco
        annotations = []
        for i, od in enumerate(piv.index):
            for j, eqp in enumerate(piv.columns):
                val = z[i, j]
                if np.isfinite(val):
                    annotations.append(dict(
                        x=str(eqp), y=str(od), text=f"{val:.1f}",
                        xref="x", yref="y", showarrow=False, font=dict(color="white", size=12)
                    ))
        fig_heat.update_layout(height=520, annotations=annotations, xaxis_title="Caminhão (eqp_id)", yaxis_title="OD")
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Sem dados suficientes para compor o heatmap.")

# =================== Frota — DF/UF (✅ rótulo do caminhão)
st.markdown("### Frota — DF/UF, Capacidade e Disponibilidade")
f1,f2 = st.columns([2,1])
df_plot = df_frota.copy()
df_plot["df_pct"] = pd.to_numeric(df_plot["df_pct"], errors="coerce")
df_plot["uf_pct"] = pd.to_numeric(df_plot["uf_pct"], errors="coerce")

fig_scatter = px.scatter(
    df_plot, x="df_pct", y="uf_pct", color="modelo", size="cap_ton",
    hover_name="eqp_id", text="eqp_id",  # ← rótulo do caminhão
    labels={"df_pct":"DF (0–1)","uf_pct":"UF (0–1)","cap_ton":"Capacidade (t)"}
)
fig_scatter.update_traces(textposition="top center")
fig_scatter.update_layout(height=420)
f1.plotly_chart(fig_scatter, use_container_width=True)

disp = df_frota["available"].value_counts(dropna=False).rename({1:"Disponível",0:"Indisp."}).reset_index()
disp.columns = ["Status","Qtd"]
fig_bar = px.bar(disp, x="Status", y="Qtd", text="Qtd")
fig_bar.update_traces(textposition="outside")
fig_bar.update_layout(height=420, xaxis={'categoryorder':'array','categoryarray':["Disponível","Indisp."]})
f2.plotly_chart(fig_bar, use_container_width=True)
