# app/app.py
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from pathlib import Path

# ----------------------------- Config/estilo
st.set_page_config(
    page_title="Sistema de Apoio à Decisão de Alocação de Frota",
    layout="centered",
    initial_sidebar_state="collapsed",
)
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
/* esconder sidebar/menu em várias versões */
.css-1d391kg, .e1fqkh3o10, .stDeployButton, .viewerBadge_container__1QSob {display:none!important;}
/* fundo e tipografia da capa */
body { background: #b4c0c9; }
.block-container { padding-top: 3rem; }
h1, h2, h3, h4 { font-weight: 800; letter-spacing: .2px; }
label, .stTextInput>div>div>input { font-size: 1.05rem; }
</style>
""", unsafe_allow_html=True)

# ----------------------------- Util: localizar e ler usuarios.xlsx
@st.cache_data
def get_users_df() -> pd.DataFrame:
    here = Path(__file__).resolve().parent          # .../app
    project_root = here.parent                      # raiz do projeto
    candidates = [
        here / "data" / "usuarios.xlsx",            # app/data/usuarios.xlsx
        project_root / "data" / "usuarios.xlsx",    # data/usuarios.xlsx  (na raiz)
        here / "usuarios.xlsx",                     # app/usuarios.xlsx   (se existir)
        project_root / "usuarios.xlsx",             # raiz/usuarios.xlsx
    ]
    last_err = None
    for p in candidates:
        if p.exists():
            try:
                # tenta com openpyxl; se não tiver, usa engine padrão
                try:
                    df = pd.read_excel(p, engine="openpyxl")
                except Exception:
                    df = pd.read_excel(p)
                df.columns = [str(c).strip().lower() for c in df.columns]
                if not {"usuario", "senha"}.issubset(df.columns):
                    raise ValueError("A planilha precisa das colunas: 'usuario' e 'senha'.")
                for c in ["usuario", "senha", "nome", "perfil"]:
                    if c in df.columns:
                        df[c] = df[c].astype(str).str.strip()
                return df
            except Exception as e:
                last_err = f"{p}: {e}"
    msg = "Arquivo 'usuarios.xlsx' não encontrado nas pastas esperadas (app/data, data ou raiz)."
    if last_err:
        msg += f"\nÚltimo erro: {last_err}"
    raise FileNotFoundError(msg)

def check_credentials(user: str, pwd: str) -> dict | None:
    df = get_users_df()
    user = (user or "").strip().lower()
    pwd  = (pwd  or "").strip()
    ok = df[(df["usuario"].str.lower() == user) & (df["senha"] == pwd)]
    if ok.empty:
        return None
    row = ok.iloc[0].to_dict()
    return {
        "usuario": row.get("usuario", ""),
        "nome": row.get("nome", row.get("usuario","")),
        "perfil": row.get("perfil",""),
    }

# ----------------------------- UI
st.markdown("### Sistema de Apoio à Decisão de Alocação de Frota de Produção de Minério")
st.markdown("## Login")

# Se já logado, mostrar atalhos + Sair
if "auth" in st.session_state:
    auth = st.session_state["auth"]
    st.success(f"Você já está logado como **{auth.get('nome',auth['usuario'])}**.")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.page_link("pages/01_Validação de Dados.py", label="Validação de Dados", icon="📊")
    with col2:
        st.page_link("pages/02_Simulador.py", label="Otimizador", icon="🧠")
    with col3:
        st.page_link("pages/03_Configurações.py", label="Configurações", icon="⚙️")
    with col4:
        if st.button("Sair", type="secondary"):
            # limpa tudo da sessão (menos chaves internas)
            for k in list(st.session_state.keys()):
                if not k.startswith("_"):
                    del st.session_state[k]
            st.experimental_rerun()
    st.stop()

# Form de login
with st.form("login_form", clear_on_submit=False):
    user = st.text_input("Usuário", placeholder="Usuário")
    pwd = st.text_input("Senha", type="password", placeholder="Senha")
    ok = st.form_submit_button("Entrar")

if ok:
    try:
        auth = check_credentials(user, pwd)
    except Exception as e:
        st.error(f"Erro no carregamento dos usuários: {e}")
        st.stop()

    if auth is None:
        st.error("Usuário ou senha inválidos.")
    else:
        st.session_state["auth"] = auth
        st.success(f"Bem-vindo, {auth['nome']}!")
        # atalhos diretos
        st.page_link("pages/01_Validação de Dados.py", label="Ir para Validação de Dados", icon="➡️")
        st.page_link("pages/02_Simulador.py", label="Ir para Otimizador", icon="➡️")