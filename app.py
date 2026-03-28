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

def limpar(texto):
    texto = texto.lower()
    texto = re.sub(r'\n', ' ', texto)
    return texto

def ler_pdf_local(nome_arquivo):
    # Procura o arquivo na pasta atual, ignorando maiúsculas/minúsculas
    arquivos_na_pasta = os.listdir('.')
    alvo = next((f for f in arquivos_na_pasta if f.lower() == nome_arquivo.lower()), None)
    
    if alvo:
        texto = ""
        try:
            with open(alvo, "rb") as arquivo:
                leitor = PyPDF2.PdfReader(arquivo)
                for pagina in leitor.pages:
                    extracao = pagina.extract_text()
                    if extracao: texto += extracao + " "
            return limpar(texto)
        except: return ""
    return ""

def dividir_texto(texto, tamanho=600):
    return [texto[i:i+tamanho] for i in range(0, len(texto), tamanho)]

def buscar_resposta(pergunta, base, vectorizer):
    pergunta_vec = vectorizer.transform([pergunta])
    base_vec = vectorizer.transform(base)
    similaridades = cosine_similarity(pergunta_vec, base_vec)
    return base[similaridades.argmax()]

def falar(texto):
    tts = gTTS(texto, lang='pt-br')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        return fp.name

# ===== CARREGAMENTO =====
doc_calendario = ler_pdf_local("CALENDARIO-ACADEMICO-2026.pdf")
doc_ppc = ler_pdf_local("PPC aprovada_no_colegiado_28_de_abril.pdf")
doc_tcc = ler_pdf_local("Regimento-de-TCC Hotelaria.pdf")

# ===== IA =====
perguntas_treino = ["calendario", "aula", "matricula", "ppc", "grade", "disciplina", "tcc", "orientador", "banca"]
categorias_treino = ["cal", "cal", "cal", "ppc", "ppc", "ppc", "tcc", "tcc", "tcc"]

@st.cache_resource
def treinar():
    vec = TfidfVectorizer()
    clf = MultinomialNB()
    clf.fit(vec.fit_transform(perguntas_treino), categorias_treino)
    return vec, clf

vectorizer, modelo = treinar()

# ===== UI =====
st.markdown("## ☀️ N.I.C.A")
st.markdown("### Núcleo de Inteligência e Conectividade Acadêmica")

# MOSTRA STATUS DOS ARQUIVOS (Para a gente saber se ele achou)
with st.expander("Status do Sistema (Clique para ver se os PDFs carregaram)"):
    st.write(f"📅 Calendário: {'✅ OK' if doc_calendario else '❌ Não encontrado'}")
    st.write(f"📖 PPC: {'✅ OK' if doc_ppc else '❌ Não encontrado'}")
    st.write(f"🎓 TCC: {'✅ OK' if doc_tcc else '❌ Não encontrado'}")

pergunta = st.text_input("Sua dúvida:")

if pergunta:
    cat = modelo.predict(vectorizer.transform([pergunta]))[0]
    texto_alvo = doc_calendario if cat == "cal" else (doc_tcc if cat == "tcc" else doc_ppc)

    if texto_alvo:
        res = buscar_resposta(pergunta, dividir_texto(texto_alvo), vectorizer)
        st.info(res)
        st.audio(falar(res[:300]))
    else:
        st.error("Desculpe, o documento necessário para responder isso não foi encontrado no GitHub.")
