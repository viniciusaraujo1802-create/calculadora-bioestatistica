import streamlit as st
import PyPDF2
import re
import tempfile
import os
from gtts import gTTS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity

# ===== CONFIG =====
st.set_page_config(page_title="N.I.C.A", page_icon="☀️")

# ===== FUNÇÕES DE PROCESSAMENTO =====

def limpar(texto):
    texto = texto.lower()
    texto = re.sub(r'\n', ' ', texto)
    return texto

def ler_pdf_local(caminho):
    texto = ""
    try:
        with open(caminho, "rb") as arquivo:
            leitor = PyPDF2.PdfReader(arquivo)
            for pagina in leitor.pages:
                text_extracted = pagina.extract_text()
                if text_extracted:
                    texto += text_extracted + " "
        return limpar(texto)
    except FileNotFoundError:
        return ""

def dividir_texto(texto, tamanho=500):
    return [texto[i:i+tamanho] for i in range(0, len(texto), tamanho)]

def buscar_resposta(pergunta, base, vectorizer):
    if not base: return "Documento não encontrado na base."
    pergunta_vec = vectorizer.transform([pergunta])
    base_vec = vectorizer.transform(base)
    similaridades = cosine_similarity(pergunta_vec, base_vec)
    indice = similaridades.argmax()
    return base[indice]

def falar(texto):
    tts = gTTS(texto, lang='pt-br')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        return fp.name

# ===== CARREGAMENTO AUTOMÁTICO (SEM UPLOAD) =====

# Certifique-se que esses arquivos estão no seu GitHub na mesma pasta do app.py
calendario_raw = ler_pdf_local("CALENDARIO-ACADEMICO-2026.pdf")
ppc_raw = ler_pdf_local("PPC.pdf") # Ajuste o nome se necessário

# ===== TREINAMENTO DO MODELO =====

perguntas = [
    "quais materias tem no curso","lista de disciplinas","grade curricular","grade do curso",
    "quando começa o semestre","prazo de matrícula","quando posso trancar","calendario",
    "como funciona o curso","qual objetivo do curso","estágio é obrigatório","ppc"
]
categorias = [
    "matriz","matriz","matriz","matriz",
    "calendario","calendario","calendario","calendario",
    "ppc","ppc","ppc","ppc"
]

@st.cache_resource
def treinar_modelo():
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(perguntas)
    modelo = MultinomialNB()
    modelo.fit(X, categorias)
    return vectorizer, modelo

vectorizer, modelo = treinar_modelo()

# ===== INTERFACE =====

st.markdown("## ☀️ N.I.C.A")
st.markdown("### Núcleo de Inteligência e Conectividade Acadêmica")

# Se você tiver a imagem, coloque-a na pasta e use o nome direto:
# st.image("logo.png", width=100)

pergunta = st.text_input("Olá! Sou o N.I.C.A. Como posso te ajudar hoje?")

# ===== LÓGICA DE RESPOSTA =====

if pergunta:
    # 1. Identifica a categoria da pergunta
    categoria = modelo.predict(vectorizer.transform([pergunta]))[0]
    
    # 2. Seleciona a base de texto correspondente
    if categoria == "calendario" and calendario_raw:
        base = dividir_texto(calendario_raw)
    elif ppc_raw:
        base = dividir_texto(ppc_raw)
    else:
        base = []

    # 3. Busca a resposta específica no texto do PDF
    if base:
        resposta = buscar_resposta(pergunta, base, vectorizer)
        
        st.markdown(f"**📌 Segundo o {categoria.upper()}:**")
        st.write(resposta)

        # 🔊 VOZ
        audio_file = falar(resposta[:300])
        st.audio(audio_file)
    else:
        st.error("Erro: Arquivos PDF não encontrados no servidor. Verifique o repositório.")
