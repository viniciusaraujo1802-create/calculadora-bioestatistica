import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from scipy import stats

st.set_page_config(page_title="BioEstatística App", layout="wide")
st.title("🧮 Calculadora de Bioestatística")

menu = st.sidebar.selectbox("Escolha a funcionalidade", ["Estatística Descritiva", "Probabilidade Normal"])

if menu == "Estatística Descritiva":
    st.header("Análise Descritiva")
    input_dados = st.text_area("Insira os dados (separados por vírgula ou espaço):")
    if input_dados:
        try:
            dados = np.fromstring(input_dados.replace(',', ' '), sep=' ')
            df = pd.DataFrame(dados, columns=["Valores"])
            col1, col2, col3 = st.columns(3)
            col1.metric("Média", f"{np.mean(dados):.2f}")
            col2.metric("Mediana", f"{np.median(dados):.2f}")
            col3.metric("Desvio Padrão", f"{np.std(dados, ddof=1):.2f}")
            fig = px.histogram(df, x="Valores", nbins=10, title="Distribuição dos Dados")
            st.plotly_chart(fig)
        except:
            st.error("Erro na leitura dos dados. Verifique se digitou apenas números.")

elif menu == "Probabilidade Normal":
    st.header("Distribuição Normal")
    media = st.number_input("Média (μ):", value=0.0)
    dp = st.number_input("Desvio Padrão (σ):", value=1.0)
    x = st.number_input("Valor de interesse (X):", value=0.0)
    prob = stats.norm.cdf(x, loc=media, scale=dp)
    st.write(f"### Probabilidade P(X < {x}) = {prob:.4f} ({prob*100:.2f}%)")
