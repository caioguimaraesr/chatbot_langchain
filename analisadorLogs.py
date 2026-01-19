
import os
import io
import re
import streamlit as st
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# ==========
# Setup Base
# ==========
load_dotenv()

st.set_page_config(page_title="Analisador de Logs de QA", page_icon="🧪", layout="wide")
st.title("🧪 Analisador Inteligente de Logs de QA")

# =========================
# Configurações de sessão/UI
# =========================
with st.sidebar:
    st.header("Configurações")
    default_model = "gemini-2.5-flash"
    model_name = st.text_input("Modelo (Google Generative AI)", value=default_model)
    temperature = st.slider("Temperatura", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    top_k = st.slider("top_k", 1, 64, 32, 1)
    top_p = st.slider("top_p", 0.1, 1.0, 0.95, 0.05)
    max_output_tokens = st.slider("max_output_tokens", 256, 4096, 2048, 128)
    verbose = st.toggle("Modo verboso (mostrar prompts)", value=False)

    st.markdown("---")
    st.caption("Dica: você pode colar logs no chat ou fazer upload de arquivo .log/.txt")

# =========================
# Inicializa o LLM
# =========================
api_key = os.getenv("API_KEY")
if not api_key:
    st.error("⚠️ API_KEY não encontrada. Crie um arquivo `.env` com API_KEY=SEU_TOKEN.")
    st.stop()

llm = ChatGoogleGenerativeAI(
    model=model_name,
    google_api_key=api_key,
    temperature=temperature,
    convert_system_message_to_human=True,
    max_output_tokens=max_output_tokens,
    top_k=top_k,
    top_p=top_p,
)

# =========================
# Utilitários
# =========================
def read_uploaded_file(file) -> str:
    """Lê um arquivo .log ou .txt e devolve string."""
    try:
        content = file.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")
        return content
    except Exception as e:
        return f"<<Falha ao ler arquivo: {e}>>"

def sanitize_log(log_text: str) -> str:
    """Limpeza leve para evitar lixo de terminal e reduzir ruído."""
    if not log_text:
        return ""
    # Remove ANSI colors
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    text = ansi_escape.sub('', log_text)
    # Limita super logs (opcional)
    max_chars = 50_000
    if len(text) > max_chars:
        text = text[-max_chars:]  # mantém o final (geralmente onde erro ocorre)
    return text.strip()

def detect_stack(log_text: str) -> str:
    """
    Heurística simples para detectar stack (Playwright, Cypress, Selenium, Node, Docker, CI, etc.)
    Apenas indicativo para guiar o prompt.
    """
    txt = log_text.lower()
    tags = []
    if any(k in txt for k in ["playwright", "expect", "locator(", "page.", "tracing.start"]):
        tags.append("Playwright")
    if any(k in txt for k in ["cypress", "cy.", "cypress.config", "cypress run"]):
        tags.append("Cypress")
    if any(k in txt for k in ["selenium", "webdriver", "by.", "chromedriver", "geckodriver"]):
        tags.append("Selenium")
    if any(k in txt for k in ["jest", "mocha", "vitest", "ts-node", "node:internal", "unhandledrejection"]):
        tags.append("Node/Test Runner")
    if any(k in txt for k in ["docker", "container", "image", "pull", "compose", "daemon"]):
        tags.append("Docker")
    if any(k in txt for k in ["github actions", "gitlab-ci", "azure pipelines", "jenkins", "circleci"]):
        tags.append("CI/CD")
    if any(k in txt for k in ["timeout", "timed out", "deadline exceeded"]):
        tags.append("Timeout")
    if any(k in txt for k in ["network", "xhr", "fetch", "net::err", "connection refused", "dns", "ssl"]):
        tags.append("Rede")
    return ", ".join(tags) if tags else "Indefinido"

# =========================
# Few-shot (exemplos)
# =========================
few_shots = [
    {
        "input": """Timeout 30000ms exceeded while waiting for selector ".btn-submit"
  at async /tests/specs/login.spec.ts:42:15
  Note: page had network requests pending""",
        "output": """**Resumo do problema:** Timeout ao aguardar o seletor `.btn-submit`.

**Causa provável:** O elemento não ficou visível por carregamento assíncrono ou pré-condição ausente.

**Solução recomendada:** 
- Use `await page.waitForSelector('.btn-submit', { state: 'visible' })`;
- Garanta que a navegação/resposta terminou (`await page.waitForLoadState('networkidle')`);
- Se houver overlay/spinner, aguarde seu desaparecimento antes do clique.

**Como evitar:** Padronize `waits` baseados em eventos de rede/DOM, estabilize mocks e use fixtures que garantam estado pronto."""
    },
    {
        "input": """CypressError: Timed out retrying after 4000ms: Expected to find element: '.card-title', but never found it.""",
        "output": """**Resumo do problema:** Cypress não localizou o seletor `.card-title` no tempo padrão (4s).

**Causa provável:** Renderização tardia do componente ou seletor incorreto/dinâmico.

**Solução recomendada:**
- Aumente timeouts para esse comando: `cy.get('.card-title', { timeout: 10000 })`;
- Prefira `cy.contains()` quando texto estável estiver presente;
- Garanta rota/fixture pronta antes da asserção: `cy.intercept(...); cy.wait('@alias')`.

**Como evitar:** Centralize `data-testid` estáveis e utilize esperas explícitas por rotas/estados previsíveis."""
    },
    {
        "input": """node:internal/errors:477
Error: connect ECONNREFUSED 127.0.0.1:5432
    at TCPConnectWrap.afterConnect [as oncomplete] (node:net:1549:16)""",
        "output": """**Resumo do problema:** Conexão recusada ao Postgres em `127.0.0.1:5432`.

**Causa provável:** Serviço do banco não está ativo/escutando ou credenciais/host incorretos.

**Solução recomendada:**
- Verifique se o container/serviço do Postgres está `up` e porta exposta;
- Confirme `DATABASE_URL`/variáveis de ambiente;
- Em Docker Compose, use o nome do serviço como host (ex.: `postgres:5432`) e `depends_on`.

**Como evitar:** Healthchecks em CI, retries exponenciais e fixture de base de dados para estado previsível."""
    }
]

examples_prompt = FewShotChatMessagePromptTemplate(
    examples=[
        {"input": ex["input"], "output": ex["output"]} for ex in few_shots
    ],
    example_prompt=ChatPromptTemplate.from_messages(
        [
            ("human", "LOG:\n{input}"),
            ("ai", "{output}")
        ]
    )
)

# =========================
# Prompt principal
# =========================
system_message = """
Você é um especialista sênior em automação de testes (Playwright, Cypress, Selenium, Node, CI/CD, Docker).
Receberá logs de erro ou descrições de falhas.

Regras de resposta (sempre siga este formato):
1) **Resumo do problema** (1-3 linhas)
2) **Causa provável** (listas curtas e objetivas; cite contexto quando possível)
3) **Solução recomendada** (passo-a-passo prático, incluindo trechos de código quando apropriado)
4) **Como evitar** (boas práticas, padrões de espera, configuração, observabilidade)

Se o log estiver incompleto, peça APENAS o essencial (trechos de stacktrace, comando executado, trecho do teste).
Se perceber tecnologia provável (Playwright, Cypress, etc.), adapte a solução ao stack.
Responda em Português-BR, técnico e conciso.
"""

user_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_message),
        # Injetamos meta-informações úteis para guiar o raciocínio
        ("human", "Contexto detectado: {detected_stack}"),
        examples_prompt,
        ("human", "Aqui está o log/descrição do erro:\n\n{user_log}\n\n"
                  "Se necessário, solicite informações faltantes objetivas.")
    ]
)

# Constrói a chain
output_parser = StrOutputParser()
detect_stack_runnable = RunnableLambda(lambda x: detect_stack(x["user_log"]))
chain = (
    {"detected_stack": detect_stack_runnable, "user_log": lambda x: x["user_log"]}
    | user_prompt
    | llm
    | output_parser
)

# ==========
# UI: Entradas
# ==========
col_left, col_right = st.columns([0.6, 0.4], gap="large")

with col_left:
    st.subheader("Cole seu log (ou descrição do erro)")
    user_text = st.text_area(
        "Entrada de texto",
        height=220,
        placeholder='Cole aqui seu log (ex.: timeout de seletor, erro de rede, stacktrace de Node, etc.)'
    )

    st.caption("ou")

    uploaded = st.file_uploader("Faça upload de arquivo .log ou .txt", type=["log", "txt"])
    file_text = ""
    if uploaded is not None:
        file_text = read_uploaded_file(uploaded)

    merged_input = user_text.strip() if user_text.strip() else file_text.strip()

with col_right:
    st.subheader("Opções de análise")
    ask_missing = st.checkbox("Permitir que o agente peça dados faltantes", value=True)
    show_detected = st.checkbox("Mostrar tecnologias detectadas", value=True)
    run_button = st.button("Analisar agora 🧪", type="primary", use_container_width=True)

# ==========
# Execução
# ==========
if run_button:
    if not merged_input:
        st.warning("Forneça um log por texto ou upload de arquivo.")
        st.stop()

    cleaned = sanitize_log(merged_input)
    stack_detected = detect_stack(cleaned)

    if verbose:
        st.code(system_message, language="markdown")

    if show_detected:
        st.info(f"🧭 Stack detectado: **{stack_detected}**")

    with st.spinner("Analisando log..."):
        try:
            # Se o usuário não quer perguntas, adicionamos dica para o modelo evitar follow-ups
            effective_input = cleaned
            if not ask_missing:
                effective_input += "\n\n[Nota: Não solicite mais informações; proponha hipóteses com base apenas no log acima.]"

            result = chain.invoke({"user_log": effective_input})

            # Saída principal
            st.markdown(result)

            # Sugere próximos passos (utilidade)
            st.markdown("---")
            st.subheader("Próximos passos úteis")
            st.markdown(
                "- ❇️ **Reproduza localmente** com logs em DEBUG.\n"
                "- 🧩 **Isole o caso**: teste mínimo que falha.\n"
                "- 🗂️ **Padronize seletores** com `data-testid`.\n"
                "- ⏱️ **Sincronize esperas** (networkidle, intercepts, healthchecks).\n"
                "- 📦 **Verifique containers/serviços** (porta, credenciais, rede).\n"
                "- 📈 **Capture evidências** (screenshot, trace, HAR) no CI.\n"
            )
        except Exception as e:
            st.error(f"Falha ao analisar: {e}")

# ==========
# Rodapé
# ==========
st.markdown("---")
st.caption(
    "Feito com ❤️ para QA. Dica: adicione sua base de conhecimento (RAG) para respostas ainda mais assertivas."
)