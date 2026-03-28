import streamlit as st
import PyPDF2
import re
import tempfile
import os
from gtts import gTTS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity

# ===== CONFIGURAÇÃO DA PÁGINA =====
st.set_page_config(page_title="N.I.C.A", page_icon="☀️")

# ===== FUNÇÕES TÉCNICAS =====

def limpar(texto):
    texto = texto.lower()
    texto = re.sub(r'\n', ' ', texto)
    return texto

def ler_pdf_local(caminho):
    texto = ""
    if os.path.exists(caminho):
        try:
            with open(caminho, "rb") as arquivo:
                leitor = PyPDF2.PdfReader(arquivo)
                for pagina in leitor.pages:
                    extracao = pagina.extract_text()
                    if extracao:
                        texto += extracao + " "
            return limpar(texto)
        except Exception as e:
            st.error(f"Erro ao ler {caminho}: {e}")
            return ""
    return ""

def dividir_texto(texto, tamanho=600):
    return [texto[i:i+tamanho] for i in range(0, len(texto), tamanho)]

def buscar_resposta(pergunta, base, vectorizer):
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

# ===== CARREGAMENTO DOS DOCUMENTOS (GitHub) =====

# Nomes exatos dos arquivos que você deve subir no repositório:
doc_calendario = ler_pdf_local("CALENDARIO-ACADEMICO-2026.pdf")
doc_ppc = ler_pdf_local("PPC aprovada_no_colegiado_28_de_abril.pdf")
doc_tcc = ler_pdf_local("Regimento-de-TCC Hotelaria.pdf")

# ===== INTELIGÊNCIA / TREINAMENTO =====

# Adicionei gatilhos para TCC
perguntas_treino = [
    "calendario", "quando começam as aulas", "data de matricula", "feriado",
    "ppc", "objetivo do curso", "grade curricular", "horas complementares",
    "tcc", "como fazer o tcc", "quem pode ser orientador", "prazo do tcc", "banca"
]
categorias_treino = [
    "calendario", "calendario", "calendario", "calendario",
    "ppc", "ppc", "ppc", "ppc",
    "tcc", "tcc", "tcc", "tcc", "tcc"
]

@st.cache_resource
def treinar_modelo():
    vec = TfidfVectorizer()
    X = vec.fit_transform(perguntas_treino)
    clf = MultinomialNB()
    clf.fit(X, categorias_treino)
    return vec, clf

vectorizer, modelo = treinar_modelo()

# ===== INTERFACE =====

st.markdown("## ☀️ N.I.C.A")
st.markdown("### Núcleo de Inteligência e Conectividade Acadêmica")

pergunta = st.text_input("Como posso te ajudar com o curso de Hotelaria hoje?")

# ===== PROCESSAMENTO DA RESPOSTA =====

if pergunta:
    # 1. Identifica o assunto
    categoria = modelo.predict(vectorizer.transform([pergunta]))[0]
    
    # 2. Seleciona o texto bruto conforme o assunto
    texto_alvo = ""
    if categoria == "calendario":
        texto_alvo = doc_calendario
    elif categoria == "tcc":
        texto_alvo = doc_tcc
    else:
        texto_alvo = doc_ppc

    # 3. Se o documento existir, busca a resposta
    if texto_alvo:
        base_fragmentada = dividir_texto(texto_alvo)
        resposta_final = buscar_resposta(pergunta, base_fragmentada, vectorizer)

        st.markdown(f"**📌 Informação encontrada no {categoria.upper()}:**")
        st.info(resposta_final)

        # Áudio
        audio_path = falar(resposta_final[:300])
        st.audio(audio_path)
    else:
        st.error(f"O documento de {categoria} não foi encontrado no servidor. Verifique o GitHub.")

# ===== RODAPÉ =====
st.caption("N.I.C.A - Assistente Acadêmico Baseado em Documentos Oficiais.")
