"""DeutschBuddy – a RAG-powered German learning assistant (Streamlit UI).

Run with:  streamlit run app.py
"""

import streamlit as st

from rag import DeutschBuddy
from rag.config import settings

st.set_page_config(page_title="DeutschBuddy", page_icon="🇩🇪", layout="wide")

st.markdown("""
<style>
.flag-stripe {height: 6px; border-radius: 3px; margin-bottom: 0.6rem;
  background: linear-gradient(90deg, #000 0 33.3%, #DD0000 33.3% 66.6%, #FFCE00 66.6%);}
.source-card {border: 1px solid rgba(128,128,128,.3); border-radius: 8px; padding: .55rem .75rem;
  margin-bottom: .45rem; font-size: .88rem;}
.source-card .meta {opacity: .7; font-size: .78rem; margin-bottom: .25rem;}
.score {float: right; font-variant-numeric: tabular-nums;}
</style>
<div class="flag-stripe"></div>
""", unsafe_allow_html=True)

CATEGORIES = {"Grammar": "grammar", "Vocabulary": "vocabulary", "Culture": "culture", "My uploads": "uploads"}
EXAMPLE_QUESTIONS = [
    "When do I use Dativ instead of Akkusativ?",
    "Why is it 'das Mädchen' and not 'die Mädchen'?",
    "Wie bestellt man im Restaurant?",
    "What is the history of the Oktoberfest?",
    "How do separable verbs like aufstehen work?",
    "What is Pfand?",
]


# ---------- Engine setup ----------

@st.cache_resource(show_spinner=False)
def get_engine() -> DeutschBuddy:
    engine = DeutschBuddy()
    engine.index_knowledge_base()
    return engine


if not settings.api_key:
    st.title("DeutschBuddy")
    st.error("No Gemini API key found.")
    st.markdown(
        "1. Get a free key at [Google AI Studio](https://aistudio.google.com/apikey)\n"
        "2. Copy `.env.example` to `.env` in the project folder\n"
        "3. Put your key after `GEMINI_API_KEY=` and restart the app")
    st.stop()

try:
    with st.spinner("Loading the knowledge base (the first start embeds all documents)…"):
        engine = get_engine()
except Exception as exc:  # show setup problems in the UI instead of a stack trace
    st.error(f"Could not start the RAG engine: {exc}")
    st.stop()


# ---------- Sidebar ----------

with st.sidebar:
    st.header("Settings")
    level = st.select_slider("Your German level", ["A1", "A2", "B1", "B2", "C1"], value="A2",
                             help="Answers, quizzes and explanations adapt to this level (CEFR).")
    language = st.radio("Explanation language", ["English", "German", "German + English"], index=0)
    chosen = st.multiselect("Search in", list(CATEGORIES), default=list(CATEGORIES),
                            help="Limit retrieval to some parts of the knowledge base.")
    categories = [CATEGORIES[c] for c in chosen] if 0 < len(chosen) < len(CATEGORIES) else None

    with st.expander("Retrieval settings"):
        settings.top_k = st.slider("Chunks retrieved (top-k)", 1, 10, settings.top_k)
        settings.min_score = st.slider("Minimum similarity", 0.0, 1.0, settings.min_score, 0.05,
                                       help="Chunks less similar than this to the question are ignored.")
    show_sources = st.toggle("Show retrieved sources", value=True)

    st.divider()
    st.caption(f"{engine.store.count()} chunks indexed · {settings.chat_model} · {settings.embed_model}")


def render_sources(hits) -> None:
    if not show_sources or not hits:
        return
    with st.expander(f"Sources ({len(hits)} passages retrieved)"):
        for i, hit in enumerate(hits, start=1):
            where = hit.source + (f", page {hit.page}" if hit.page else "")
            section = f" · {hit.section}" if hit.section else ""
            preview = hit.text[:420].replace("<", "&lt;") + ("…" if len(hit.text) > 420 else "")
            st.markdown(
                f'<div class="source-card"><div class="meta"><b>[{i}]</b> {where}{section}'
                f'<span class="score">similarity {hit.score:.2f}</span></div>{preview}</div>',
                unsafe_allow_html=True)


st.title("DeutschBuddy")
st.caption("Ask questions about German grammar, vocabulary and culture. "
           "Every answer is grounded in the knowledge base and cites its sources.")

tab_chat, tab_explain, tab_quiz, tab_docs = st.tabs(
    ["Ask a question", "Sentence explainer", "Quiz", "Knowledge base"])


# ---------- Tab 1: Chat ----------

with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    question = st.chat_input("Ask anything about German… (English or Deutsch)")
    question = question or st.session_state.pop("pending_question", None)

    if not st.session_state.messages and not question:
        st.markdown("**Try one of these:**")
        cols = st.columns(3)
        for i, example in enumerate(EXAMPLE_QUESTIONS):
            if cols[i % 3].button(example, key=f"ex{i}", use_container_width=True):
                st.session_state.pending_question = example
                st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                render_sources(msg.get("sources"))

    if question:
        with st.chat_message("user"):
            st.markdown(question)
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-6:]]
        with st.chat_message("assistant"):
            try:
                hits, stream = engine.ask(question, history, level, language, categories)
                answer = st.write_stream(stream)
                render_sources(hits)
            except Exception as exc:
                answer, hits = f"Sorry, the request failed: {exc}", []
                st.error(answer)
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": hits})

    if st.session_state.messages and st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()


# ---------- Tab 2: Sentence explainer ----------

with tab_explain:
    st.subheader("Translate and explain a sentence")
    st.write("Paste a German sentence (or an English one to translate). DeutschBuddy translates it, "
             "corrects mistakes and explains every word.")
    sentence = st.text_input("Sentence", value="Gestern bin ich mit meinem Bruder ins Kino gegangen.",
                             key="sentence")
    if st.button("Explain", type="primary", disabled=not sentence.strip()):
        with st.spinner("Analysing the sentence…"):
            try:
                st.session_state.explanation = engine.explain_sentence(sentence, level)
            except Exception as exc:
                st.error(f"Could not explain the sentence: {exc}")

    if "explanation" in st.session_state:
        data, hits = st.session_state.explanation
        st.markdown(f"### {data['german']}")
        st.markdown(f"**English:** {data['english']}  \n**Word for word:** *{data['literal']}*")
        if data.get("corrections"):
            st.warning(f"**Corrections:** {data['corrections']}")
        st.markdown("#### Word by word")
        st.dataframe(
            [{"Word": w["word"], "Base form": w["base_form"], "Type": w["type"], "Note": w["note"]}
             for w in data["words"]],
            hide_index=True, use_container_width=True)
        st.markdown("#### Grammar in this sentence")
        for point in data["grammar_points"]:
            cite = f" [{point['source']}]" if point.get("source") else ""
            st.markdown(f"- **{point['title']}**{cite}: {point['explanation']}")
        render_sources(hits)


# ---------- Tab 3: Quiz ----------

with tab_quiz:
    st.subheader("Practice quiz")
    col_topic, col_n = st.columns([3, 1])
    topic = col_topic.text_input("Topic (leave empty for a surprise quiz)", value="Akkusativ and Dativ prepositions")
    n_questions = col_n.number_input("Questions", 3, 10, 5)

    if st.button("Generate quiz", type="primary"):
        with st.spinner("Writing questions from the knowledge base…"):
            try:
                questions, hits = engine.make_quiz(topic, int(n_questions), level, categories)
                for key in [k for k in st.session_state if str(k).startswith("q") and str(k)[1:].isdigit()]:
                    del st.session_state[key]  # forget answers from the previous quiz
                st.session_state.quiz = {"questions": questions, "hits": hits}
            except Exception as exc:
                st.error(f"Could not create the quiz: {exc}")

    quiz = st.session_state.get("quiz")
    if quiz:
        with st.form("quiz_form"):
            answers = []
            for i, q in enumerate(quiz["questions"]):
                answers.append(st.radio(f"**{i + 1}. {q['question']}**", q["options"], index=None, key=f"q{i}"))
            submitted = st.form_submit_button("Check answers")

        if submitted:
            score = 0
            for i, (q, given) in enumerate(zip(quiz["questions"], answers)):
                correct = q["options"][q["answer_index"]]
                if given == correct:
                    score += 1
                    st.success(f"**{i + 1}.** Correct: {correct}. {q['explanation']}")
                else:
                    st.error(f"**{i + 1}.** {'Your answer: ' + given + '. ' if given else 'No answer. '}"
                             f"Correct: **{correct}**. {q['explanation']}")
            total = len(quiz["questions"])
            st.metric("Score", f"{score} / {total}")
            if score == total:
                st.balloons()
        render_sources(quiz["hits"])


# ---------- Tab 4: Knowledge base ----------

with tab_docs:
    st.subheader("Add your own material")
    st.write("Upload German textbook chapters, vocabulary lists or articles. They are split into chunks, "
             "embedded and added to the search index. Use **Search in → My uploads** to ask only about them.")
    files = st.file_uploader("PDF, Markdown or text files", type=["pdf", "md", "txt"], accept_multiple_files=True)
    if files and st.button("Add to knowledge base", type="primary"):
        for f in files:
            with st.spinner(f"Indexing {f.name}…"):
                try:
                    report = engine.add_document(f.name, f.getvalue())
                    st.success(f"{f.name}: {report.total_chunks} chunks ({report.new_chunks} new)")
                except Exception as exc:
                    st.error(f"{f.name}: {exc}")

    st.subheader("Indexed documents")
    sources = engine.store.sources()
    st.dataframe([{"Document": name, "Category": info["category"], "Chunks": info["chunks"]}
                  for name, info in sources.items()], hide_index=True, use_container_width=True)

    uploads = [name for name, info in sources.items() if info["category"] == "uploads"]
    if uploads:
        to_remove = st.selectbox("Remove an uploaded document", uploads, index=None)
        if to_remove and st.button(f"Remove {to_remove}"):
            engine.store.delete_source(to_remove)
            st.rerun()

    with st.expander("Test retrieval"):
        probe = st.text_input("Search the index without asking the LLM", value="Wechselpräpositionen")
        if probe:
            render_sources(engine.retrieve(probe, categories=categories))
