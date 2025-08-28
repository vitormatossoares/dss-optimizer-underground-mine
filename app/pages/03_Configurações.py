# pages/Configurações.py
# -*- coding: utf-8 -*-
import streamlit as st

st.set_page_config(page_title="Configurações | Alocação de Frota", layout="wide", initial_sidebar_state="collapsed")
if "auth" not in st.session_state:
    st.error("Acesso restrito. Faça login para continuar.")
    st.page_link("app.py", label="Voltar ao Login", icon="↩️")
    st.stop()

st.title("Configurações do Modelo")

# ---------------------- Defaults ----------------------
DEFAULTS = {
    # Função Objetivo
    "beta_softcap": 0.10,           # β: 0=truncado, 1=linear
    "underutil_penalty": 0.05,      # μ: penalidade por ociosidade
    "overshoot_penalty_ton": 1e-4,  # λ: penalidade por excesso (por tonelada)

    # SA
    "sa_T0": 1.0,
    "sa_Tf": 1e-3,
    "sa_iters": 2000,
    "sa_moves_per_iter": 1,
    "sa_ignore_outliers_flag": True,
    "sa_random_state": 42,
}

def _get(key):  # pega do session_state com default
    return st.session_state.get(key, DEFAULTS[key])

# ---------------------- Função Objetivo ----------------------
st.header("Função Objetivo")

c1, c2, c3 = st.columns(3)
with c1:
    beta = st.slider(
        "Soft-cap β (0=truncado, 1=linear)",
        min_value=0.0, max_value=1.0, value=float(_get("beta_softcap")), step=0.05,
        help="β controla o quanto o excedente acima da meta ainda conta no score. 0=ignora (trunca), 1=100% linear."
    )
with c2:
    mu = st.slider(
        "Penalidade ociosidade μ",
        min_value=0.00, max_value=0.20, value=float(_get("underutil_penalty")), step=0.01,
        help="Penaliza não utilizar caminhões alocados (ociosidade relativa). Valores típicos: 0.03–0.10."
    )
with c3:
    lam = st.number_input(
        "Penalidade por excesso λ (por tonelada)",
        min_value=0.0, value=float(_get("overshoot_penalty_ton")), step=1e-4, format="%.6f",
        help="Custo por tonelada acima da meta (mantém incentivo a atender metas antes de exceder demasiadamente)."
    )

# ---------------------- Parâmetros do SA ----------------------
st.header("Simulated Annealing (SA)")

csa1, csa2, csa3, csa4 = st.columns(4)
with csa1:
    sa_T0 = st.number_input("Temperatura inicial T0", min_value=1e-6, value=float(_get("sa_T0")), step=0.1, format="%.6f")
with csa2:
    sa_Tf = st.number_input("Temperatura final Tf",  min_value=1e-9, value=float(_get("sa_Tf")), step=1e-4, format="%.9f")
with csa3:
    sa_iters = st.number_input("Iterações", min_value=100, step=100, value=int(_get("sa_iters")))
with csa4:
    sa_moves = st.number_input("Movimentos por iteração", min_value=1, step=1, value=int(_get("sa_moves_per_iter")))

csa5, csa6 = st.columns(2)
with csa5:
    sa_ignore = st.toggle("Ignorar outliers de ciclo", value=bool(_get("sa_ignore_outliers_flag")))
with csa6:
    sa_seed = st.number_input("Random state (seed)", min_value=0, step=1, value=int(_get("sa_random_state")))

# ---------------------- Ações ----------------------
cc1, cc2, cc3 = st.columns([1,1,6])
with cc1:
    if st.button("Salvar configurações", type="primary"):
        # FO
        st.session_state["beta_softcap"] = float(beta)
        st.session_state["underutil_penalty"] = float(mu)
        st.session_state["overshoot_penalty_ton"] = float(lam)
        # SA
        st.session_state["sa_params"] = {
            "T0": float(sa_T0),
            "Tf": float(sa_Tf),
            "iters": int(sa_iters),
            "moves_per_iter": int(sa_moves),
            "ignore_outliers_flag": bool(sa_ignore),
            "random_state": int(sa_seed),
            # obs: overshoot_penalty_ton fica fora do SAParams; é da FO (evaluate).
        }
        st.success("Configurações salvas. Abra o *Otimizador* e execute a simulação.")

with cc2:
    if st.button("Restaurar padrão"):
        for k, v in DEFAULTS.items():
            if k.startswith("sa_"):
                continue
            st.session_state[k] = v
        st.session_state["sa_params"] = {
            "T0": DEFAULTS["sa_T0"],
            "Tf": DEFAULTS["sa_Tf"],
            "iters": DEFAULTS["sa_iters"],
            "moves_per_iter": DEFAULTS["sa_moves_per_iter"],
            "ignore_outliers_flag": DEFAULTS["sa_ignore_outliers_flag"],
            "random_state": DEFAULTS["sa_random_state"],
        }
        st.info("Padrões restaurados.")

st.caption("Dica: β=0.1 é um bom padrão (soft-cap suave). μ entre 0.03–0.10. λ entre 0.00005–0.0002.")
