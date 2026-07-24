import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="🚂 Railway FAQ Bot",
    page_icon="🚂",
    layout="wide"
)

# ============================================
# LOAD AI MODEL
# ============================================
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ============================================
# LOAD Q&A FROM EXCEL
# ============================================
@st.cache_data
def load_qa_from_excel():
    """Load Q&A from Excel file."""
    try:
        df = pd.read_excel("railway_qa.xlsx")
        qa_list = []
        for _, row in df.iterrows():
            variations = []
            if pd.notna(row.get("Variations", "")):
                variations = [v.strip() for v in str(row["Variations"]).split("|") if v.strip()]

            qa_list.append({
                "category": str(row.get("Category", "General")),
                "question": str(row["Question"]),
                "variations": variations,
                "answer": str(row["Answer"])
            })
        return qa_list
    except FileNotFoundError:
        st.error("❌ railway_qa.xlsx not found! Please create it.")
        return []

qa_database = load_qa_from_excel()

# ============================================
# BUILD SEARCH INDEX
# ============================================
@st.cache_resource
def build_index(_qa_db):
    """Build AI search index."""
    all_questions = []
    question_to_qa = []

    for qa in _qa_db:
        all_questions.append(qa["question"])
        question_to_qa.append(qa)

        for variation in qa.get("variations", []):
            all_questions.append(variation)
            question_to_qa.append(qa)

    if not all_questions:
        return [], np.array([]), []

    embeddings = model.encode(all_questions, show_progress_bar=False)
    return all_questions, embeddings, question_to_qa

all_questions, question_embeddings, question_to_qa = build_index(qa_database)

# ============================================
# SEARCH FUNCTION
# ============================================
def find_answer(user_question, threshold=0.45):
    """Find best matching answer using AI."""
    if len(all_questions) == 0:
        return None, 0, []

    user_embedding = model.encode([user_question])
    similarities = np.dot(question_embeddings, user_embedding.T).flatten()

    top_indices = np.argsort(similarities)[-3:][::-1]
    best_idx = top_indices[0]
    best_score = similarities[best_idx]

    if best_score >= threshold:
        suggestions = []
        seen = set()
        for idx in top_indices[1:]:
            if similarities[idx] > threshold:
                q = question_to_qa[idx]["question"]
                if q not in seen and q != question_to_qa[best_idx]["question"]:
                    suggestions.append(q)
                    seen.add(q)
        return question_to_qa[best_idx], best_score, suggestions

    return None, 0, []

# ============================================
# STREAMLIT UI
# ============================================
st.title("🚂 Railway Operations Chatbot")
st.caption("Ask me anything about railway operations, safety, signalling, and more!")

# ---- SIDEBAR ----
with st.sidebar:
    st.header("🚂 Railway Bot")
    st.divider()

    # Show topics from Excel
    st.subheader("📋 Topics I Cover")
    categories = {}
    for qa in qa_database:
        cat = qa["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(qa["question"])

    for category in sorted(categories.keys()):
        with st.expander(category):
            for q in categories[category]:
                st.write(f"• {q}")

    st.divider()

    # Quick questions
    st.subheader("⚡ Quick Questions")
    quick_questions = []
    for qa in qa_database[:6]:
        quick_questions.append(qa["question"])

    for qq in quick_questions:
        if st.button(qq, key=f"quick_{qq}", use_container_width=True):
            st.session_state.quick_question = qq

    st.divider()

    # Stats
    st.caption(f"📊 Knowledge base: {len(qa_database)} topics")
    st.caption(f"🔍 Indexed: {len(all_questions)} searchable phrases")

    # Reload button
    if st.button("🔄 Reload Q&A from Excel", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

# ---- CHAT HISTORY ----
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 Welcome! I'm your Railway Operations Assistant.\n\nAsk me anything about maintenance, safety, signalling, or check the sidebar for quick topics!"
        }
    ]

if "quick_question" not in st.session_state:
    st.session_state.quick_question = None

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---- HANDLE INPUT ----
user_input = st.chat_input("Ask about railway operations...")

# Handle quick question
if st.session_state.quick_question:
    user_input = st.session_state.quick_question
    st.session_state.quick_question = None

if user_input:
    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Find answer
    match, confidence, suggestions = find_answer(user_input)

    # Build response
    with st.chat_message("assistant"):
        if match and confidence > 0.6:
            st.markdown(match["answer"])
            st.caption(f"📊 Confidence: {confidence:.0%} | Category: {match['category']}")

            if suggestions:
                with st.expander("🔗 Related Topics"):
                    for s in suggestions:
                        st.write(f"• {s}")

        elif match and confidence > 0.45:
            st.markdown(f"🤔 Did you mean: **{match['question']}**?")
            st.divider()
            st.markdown(match["answer"])
            st.caption(f"📊 Confidence: {confidence:.0%}")

        else:
            st.markdown("😅 I don't have an answer for that yet.\n\nTry rephrasing or check the topics in the sidebar!")

    answer_text = match["answer"] if match else "No answer found."
    st.session_state.messages.append({"role": "assistant", "content": answer_text})