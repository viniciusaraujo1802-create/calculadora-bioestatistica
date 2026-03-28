import streamlit as st
import PyPDF2
import re
import tempfile
import os
import pandas as pd
from gtts import gTTS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity

# ===== 1. CONFIGURAÇÃO E IDENTIDADE DA N.I.C.A =====
st.set_page_config(page_title="N.I.C.A ☀️ UFRRJ", page_icon="☀️", layout="wide")

# Dados Estruturados (Extraídos do PPC e Regimentos que você enviou)
CONHECIMENTO_FIXO = {
    "horas": {
        "resumo": "Total: 200h (Deliberação 078/2007).",
        "detalhes": "Artigo publicado: 20h. Monitoria: 30h/sem. Estágio extra: 30h/sem. Eventos: 5h. Viagem Nacional: 5h. Visita Técnica: 6h/dia (RJ) ou 8h/dia (Fora).",
        "tags": ["horas", "autonoma", "complementar", "pontos", "atividades"]
    },
    "estagio": {
        "resumo": "Total: 205h obrigatórias.",
        "detalhes": "Estágio 1: 105h no Hotel Escola. Estágio 2: 100h externas. Monitoria pode abater até 150h de estágio! Contato: monitoria@ufrrj.br.",
        "tags": ["estagio", "obrigatorio", "hotel escola", "vaga", "abater"]
    },
    "tcc": {
        "resumo": "Disciplinas IS 333 (TCC I) e IS 334 (TCC II).",
        "detalhes": "Modalidades: Monografia, Artigo, Plano de Negócio ou Relatório. Bancas: Agendar 15 dias antes no SIGAA. Precisa de 3 membros.",
        "tags": ["tcc", "monografia", "banca", "defesa", "formatura", "artigo"]
    },
    "orientadores": {
        "resumo": "Corpo docente especializado da Hotelaria.",
        "detalhes": "Sergio Domingos (Gestão), Elga Batista (A&B/Gênero), Mariana Pires (Sustentabilidade/Eventos), Patricia Freitas (Consumo/Criança), Sueli Moreira (Hospitalidade Rural/Gênero).",
        "tags": ["orientador", "professor", "quem orienta", "pesquisa", "ajuda"]
    }
}

# ===== 2. FUNÇÕES DE PROCESSAMENTO DE TEXTO =====
def limpar(texto):
    texto = texto.lower()
    texto = re.sub(r'\n', ' ', texto)
    return texto

def ler_pdf_local(nome_arquivo):
    arquivos_na_pasta = os.listdir('.')
    alvo = next((f for f in arquivos_na_pasta if f.lower() == nome_arquivo.lower()), None)
    if alvo:
        try:
            with open(alvo, "rb") as arquivo:
                leitor = PyPDF2.PdfReader(arquivo)
                texto = " ".join([p.extract_text() for p in leitor.pages if p.extract_text()])
            return limpar(texto)
        except: return ""
    return ""

def dividir_texto(texto, tamanho=700):
    return [texto[i:i+tamanho] for i in range(0, len(texto), tamanho)]

def falar(texto):
    try:
        tts = gTTS(texto[:250], lang='pt-br') # Limite para rapidez
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            return fp.name
    except: return None

# ===== 3. INTELIGÊNCIA ARTIFICIAL (Classificação e Busca) =====
@st.cache_resource
def treinar_modelo():
    perguntas = ["calendario aula matricula", "ppc grade disciplinas curso", "tcc banca monografia orientador", "horas complementares autonomas", "estagio hotel escola"]
    categorias = ["cal", "ppc", "tcc", "horas", "estagio"]
    vec = TfidfVectorizer()
    clf = MultinomialNB()
    clf.fit(vec.fit_transform(perguntas), categorias)
    return vec, clf

vectorizer, modelo = treinar_modelo()

# ===== 4. INTERFACE DO USUÁRIO =====
st.markdown("<h1 style='text-align: center;'>☀️ N.I.C.A.</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'><b>Núcleo de Inteligência e Cidadania Acadêmica</b><br>Hospitalidade UFRRJ</p>", unsafe_allow_html=True)

# Sidebar de Status
with st.sidebar:
    st.header("📂 Base de Conhecimento")
    doc_cal = ler_pdf_local("CALENDARIO-ACADEMICO-2026.pdf")
    doc_ppc = ler_pdf_local("PPC aprovada_no_colegiado_28_de_abril.pdf")
    doc_tcc = ler_pdf_local("Regimento-de-TCC Hotelaria.pdf")
    
    st.write(f"📅 Calendário 2026: {'✅' if doc_cal else '❌'}")
    st.write(f"📖 PPC Hotelaria: {'✅' if doc_ppc else '❌'}")
    st.write(f"🎓 Regimento TCC: {'✅' if doc_tcc else '❌'}")
    st.markdown("---")
    st.write("Desenvolvido por: **Vinicius Araujo**")

# Campo de Pergunta
pergunta_user = st.text_input("Olá! Sou a N.I.C.A. Como posso facilitar sua vida acadêmica hoje?", placeholder="Ex: Como ganho horas com artigo?")

if pergunta_user:
    p_limpa = limpar(pergunta_user)
    resposta = None
    fonte = ""

    # PASSO 1: Busca no Banco de Dados Estruturado (Alta Precisão)
    for chave, info in CONHECIMENTO_FIXO.items():
        if any(tag in p_limpa for tag in info["tags"]):
            resposta = f"**{info['resumo']}** \n\n {info['detalhes']}"
            fonte = "Regulamento Interno de Hotelaria"
            break

    # PASSO 2: Se não for regra fixa, busca nos PDFs usando sua lógica de IA
    if not resposta:
        cat = modelo.predict(vectorizer.transform([p_limpa]))[0]
        texto_alvo = doc_cal if cat == "cal" else (doc_tcc if cat == "tcc" else doc_ppc)
        
        if texto_alvo:
            chunks = dividir_texto(texto_alvo)
            # Similaridade de Cosseno para achar o trecho mais relevante
            v_busca = TfidfVectorizer().fit(chunks)
            sim = cosine_similarity(v_busca.transform([p_limpa]), v_busca.transform(chunks))
            resposta = chunks[sim.argmax()]
            fonte = f"Documento PDF ({cat.upper()})"

    # EXIBIÇÃO DO RESULTADO
    if resposta:
        st.success(f"### Resposta da N.I.C.A:")
        st.write(resposta)
        st.caption(f"Fonte: {fonte}")
        
        # Áudio da resposta
        audio = falar(resposta)
        if audio:
            st.audio(audio)
            
        # Sugestão de ação
        st.markdown("---")
        st.info("💡 **Dica da N.I.C.A:** Se precisar de mais detalhes, verifique o SIGAA ou procure a Secretaria de Hotelaria no ICSA.")
    else:
        st.error("Desculpe, ainda não encontrei essa informação nos manuais do curso.")

# ===== 5. SIMULADOR DE HORAS (EXTRA) =====
with st.expander("📊 Simulador de Horas Complementares"):
    st.write("Marque o que você já fez para ver quanto falta:")
    c1 = st.checkbox("Publiquei Artigo Científico (20h)")
    c2 = st.checkbox("Fiz um Semestre de Monitoria (30h)")
    c3 = st.checkbox("Participei de Evento como Ouvinte (5h)")
    
    total = (20 if c1 else 0) + (30 if c2 else 0) + (5 if c3 else 0)
    st.progress(total / 200)
    st.write(f"Você já tem **{total}h** de 200h necessárias.")
