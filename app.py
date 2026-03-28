import streamlit as st
import PyPDF2
import re
import tempfile

from gtts import gTTS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity

# ===== CONFIG =====
st.set_page_config(page_title="N.I.C.A")

# ===== FUNÇÕES =====

def limpar(texto):
    texto = texto.lower()
    texto = re.sub(r'\n', ' ', texto)
    return texto

def ler_pdf(uploaded_file):
    texto = ""
    leitor = PyPDF2.PdfReader(uploaded_file)
    for pagina in leitor.pages:
        texto += pagina.extract_text() + " "
    return texto

def dividir_texto(texto, tamanho=400):
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

# ===== TREINAMENTO =====

perguntas = [
    "quais materias tem no curso","lista de disciplinas","grade curricular",
    "quando começa o semestre","prazo de matrícula","quando posso trancar",
    "como funciona o curso","qual objetivo do curso","o curso exige estágio"
]

categorias = [
    "matriz","matriz","matriz",
    "calendario","calendario","calendario",
    "ppc","ppc","ppc"
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

st.markdown("### 📂 Envie os PDFs")
calendario_file = st.file_uploader("Calendário Acadêmico", type="pdf")
ppc_file = st.file_uploader("PPC do Curso", type="pdf")

pergunta = st.text_input("Digite sua dúvida:")

# ===== PROCESSAMENTO =====

if calendario_file and ppc_file and pergunta:

    calendario = limpar(ler_pdf(calendario_file))
    ppc = limpar(ler_pdf(ppc_file))

    categoria = modelo.predict(vectorizer.transform([pergunta]))[0]

    if categoria == "calendario":
        base = dividir_texto(calendario)
    else:
        base = dividir_texto(ppc)

    resposta = buscar_resposta(pergunta, base, vectorizer)

    st.markdown("### 📌 Resposta:")
    st.write(resposta[:500])

    # 🔊 VOZ
    st.markdown("### 🔊 Ouvir resposta:")
    audio_file = falar(resposta[:300])
    st.audio(audio_file)

    import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ===== CONFIG =====
st.set_page_config(page_title="N.I.C.A", page_icon="logo.png")

# ===== BASE DE TREINAMENTO =====
perguntas = [

# ===== MATRIZ CURRICULAR =====
"quais materias tem no curso",
"quais disciplinas existem",
"lista de disciplinas",
"grade curricular",
"como é a grade do curso",
"quais materias eu vou estudar",
"o curso tem quais disciplinas",
"disciplinas do curso de hotelaria",
"quais materias tem no primeiro periodo",
"quais materias tem no segundo periodo",
"tem pré requisito nas disciplinas",
"quais são os pré requisitos",
"ordem das disciplinas",
"como funciona a matriz curricular",
"estrutura das disciplinas do curso",
"grade do curso completa",

# ===== CALENDÁRIO ACADÊMICO =====
"quando começa o semestre",
"quando inicia o período",
"data de início das aulas",
"inicio das aulas",
"quando termina o semestre",
"data de fim das aulas",
"quando acaba o período",
"prazo de matrícula",
"quando posso fazer matrícula",
"data da matrícula",
"quando abre matrícula",
"até quando posso me matricular",
"prazo de trancamento",
"quando posso trancar disciplina",
"data limite para trancamento",
"quais são os feriados acadêmicos",
"calendário acadêmico do curso",
"datas importantes do semestre",

# ===== PPC =====
"como funciona o curso",
"qual objetivo do curso",
"qual o objetivo do curso de hotelaria",
"o que o curso forma",
"perfil do egresso",
"o que faz um formado em hotelaria",
"como é o curso de hotelaria",
"o curso exige estágio",
"o estágio é obrigatório",
"o curso cobra estágio obrigatório",
"tem estágio no curso",
"precisa fazer estágio",
"carga horária do curso",
"quantas horas tem o curso",
"regras do curso",
"como funciona a formação",
"o que vou aprender no curso"
]

categorias = [

# MATRIZ
"matriz","matriz","matriz","matriz","matriz","matriz","matriz","matriz",
"matriz","matriz","matriz","matriz","matriz","matriz","matriz","matriz",

# CALENDÁRIO
"calendario","calendario","calendario","calendario","calendario","calendario",
"calendario","calendario","calendario","calendario","calendario","calendario",
"calendario","calendario","calendario","calendario","calendario","calendario",

# PPC
"ppc","ppc","ppc","ppc","ppc","ppc","ppc",
"ppc","ppc","ppc","ppc","ppc","ppc",
"ppc","ppc","ppc","ppc"
]

# ===== TREINAMENTO =====
@st.cache_resource
def get_trained_model(questions, categories):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(questions)
    model = MultinomialNB()
    model.fit(X, categories)
    return vectorizer, model

vectorizer, modelo = get_trained_model(perguntas, categorias)

# ===== BASE DE RESPOSTAS =====
respostas = {
    "matriz": """
A matriz curricular do curso de Hotelaria organiza as disciplinas por período,
incluindo pré-requisitos e progressão acadêmica. Consulte a matriz para planejar sua matrícula.
""",

    "calendario": """
O calendário acadêmico define prazos importantes como matrícula, trancamento, início e fim das aulas.
Fique atento às datas para não perder prazos institucionais.
""",

    "ppc": """
O PPC (Projeto Pedagógico do Curso) define os objetivos do curso, perfil do egresso,
estrutura curricular e regras gerais, incluindo a obrigatoriedade de estágio.
"""
}

# ===== INTERFACE =====
st.markdown("## ☀️ N.I.C.A")
st.markdown("### Núcleo de Inteligência e Conectividade Acadêmica")

col1, col2 = st.columns([1, 6])

with col1:
    st.image("/content/ChatGPT Image 25 de mar. de 2026, 15_07_49.png", width=60)

with col2:
    pergunta = st.text_input("Digite sua dúvida:")

# ===== RESPOSTA =====
if pergunta:
    categoria = modelo.predict(vectorizer.transform([pergunta]))[0]
    resposta = respostas[categoria]

    st.markdown("### 📌 Resposta:")
    st.write(resposta)
def dividir_texto(texto, tamanho=500):
    return [texto[i:i+tamanho] for i in range(0, len(texto), tamanho)]

import PyPDF2

def ler_pdf(caminho):
    texto = ""
    with open(caminho, "rb") as arquivo:
        leitor = PyPDF2.PdfReader(arquivo)
        for pagina in leitor.pages:
            texto += pagina.extract_text() + " "
    return texto

# These files are assumed to be present in the content directory.
# If they are not, please upload them.
# calendario = ler_pdf("/content/CALENDARIO-ACADEMICO-2026.pdf")
# ppc = ler_pdf("/content/PPC aprovada_no_colegiado_28_de_abril.pdf")

# For demonstration purposes, using dummy variables if files are not available.
# You should uncomment the lines above and provide the actual PDF files
# for the application to work as intended.
calendario = "Este é um texto de exemplo para o calendário acadêmico."
ppc = "Este é um texto de exemplo para o PPC."

base_conhecimento = dividir_texto(calendario) + dividir_texto(ppc)
