import streamlit as st

# --- BANCO DE DADOS DENSO (Extraído dos seus PDFs) ---
# Aqui estão as regras reais que o seu PLN deve consultar
BASE_CONHECIMENTO_REAL = {
    "TCC": {
        "modalidades": ["Monografia", "Plano de Negócio", "Artigo Científico", "Relatório de Estágio", "Capítulo de Livro"],
        "regra_banca": "Composição de 3 membros (Orientador + 2). Pedido via SIGAA com 15 dias de antecedência.",
        "matricula": "Atividade AA483. Precisa de anuência do orientador enviada à Secretaria.",
        "troca_orientador": "Exige formulário próprio assinado pelo antigo e novo orientador com carimbo SIAPE."
    },
    "ESTAGIO": {
        "carga_total": 235, # 205h práticas + 30h teóricas
        "hotel_escola": "105h obrigatórias no 5º período (contraturno).",
        "externo": "100h em empresas conveniadas.",
        "abate": "Até 150h podem ser aproveitadas de Monitoria/Iniciação Científica (Art. 20)."
    },
    "HORAS_AUTONOMAS": {
        "total": 200,
        "grupos": {
            "Ensino": "Monitoria (30h/sem), Estágio Extra (30h/sem), Disciplinas Eletivas.",
            "Pesquisa": "Artigo (20h), Resumo em Anais (5h), Apresentação de Trabalho (10h).",
            "Extensão": "Organização de Evento (10h), Ouvinte (5h), Visita Técnica (6h/dia RJ)."
        }
    }
}

# --- LÓGICA DE PLN (Busca por Palavras-Chave e Contexto) ---
def nica_responder(pergunta):
    p = pergunta.lower()
    
    if "estágio" in p or "horas de estágio" in p:
        info = BASE_CONHECIMENTO_REAL["ESTAGIO"]
        return f"Sobre o estágio: São {info['carga_total']}h totais. Você sabia que pode abater até {info['abate']}?"
    
    if "tcc" in p or "modalidade" in p:
        info = BASE_CONHECIMENTO_REAL["TCC"]
        return f"As modalidades aceitas são: {', '.join(info['modalidades'])}. A matrícula é na AA483."
    
    if "horas" in p or "autônoma" in p:
        info = BASE_CONHECIMENTO_REAL["HORAS_AUTONOMAS"]
        return f"Você precisa de 200h. Destaque para Pesquisa: Artigos valem 20h e Resumos valem 5h."

    return "Essa informação específica está nos anexos do PPC. Quer que eu busque o parágrafo exato no PDF?"

# Interface Streamlit
st.title("☀️ N.I.C.A. - Protótipo de Pesquisa")
user_query = st.text_input("Pergunta técnica (PPC/TCC/Calendário):")

if user_query:
    resposta = nica_responder(user_query)
    st.write(resposta)
