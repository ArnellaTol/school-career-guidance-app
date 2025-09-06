import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import os
import json
import logging
from dotenv import load_dotenv
from huggingface_hub import InferenceClient, login
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex
from streamlit_option_menu import option_menu


# ==========================
#  CONFIG
# ==========================
st.set_page_config(page_title="Профориентация для школьников")
logging.basicConfig(level=logging.DEBUG)
load_dotenv()


# ==========================
#  GLOBAL DATA
# ==========================
out_col_names = ['signsystem', 'technology', 'nature', 'artistic', 'human', 'business']
inp_col_names = [
    'kaz_lang_7', 'liter_7', 'rus_lang_7', 'eng_lang_7', 'math_7', 'comps_7',
    'kaz_hist_7', 'art_7', 'pe_7', 'geography_7', 'biology_7', 'chemistry_7',
    'physics_7', 'world_hist_7', 'Activist', 'Career', 'Tester', 'Creator',
    'Designer', 'Researcher', 'kaz_lang_8', 'liter_8', 'rus_lang_8', 'eng_lang_8',
    'math_8', 'comps_8', 'kaz_hist_8', 'art_8', 'pe_8', 'geography_8', 'biology_8',
    'chemistry_8', 'physics_8', 'world_hist_8', 'kaz_lang_9', 'liter_9', 'rus_lang_9',
    'eng_lang_9', 'math_9', 'comps_9', 'kaz_hist_9', 'art_9', 'pe_9', 'geography_9',
    'biology_9', 'chemistry_9', 'physics_9', 'world_hist_9', 'rights_9', 'kaz_lang_10',
    'liter_10', 'rus_lang_10', 'eng_lang_10', 'math_10', 'comps_10', 'kaz_hist_10',
    'art_10', 'pe_10', 'geography_10', 'biology_10', 'chemistry_10', 'physics_10',
    'world_hist_10'
]

# (сокращено: dict-ы column_names_dict и column_names_dict_ru, checkbox_columns, type_columns, thresholds остаются без изменений)


# ==========================
#  HELPER FUNCTIONS
# ==========================
def create_expander(class_label, cols):
    with st.expander(f"Оценки за {class_label} класс"):
        for col in cols:
            input_values[col] = st.number_input(
                column_names_dict_ru[col], min_value=2, max_value=5, step=1, value=5, key=col
            )

def save_to_dataframe(selected_checkboxes, input_values):
    data = {**selected_checkboxes, **input_values}
    for key in checkbox_columns:
        data[key] = int(data.get(key, False))
    df = pd.DataFrame([data], columns=inp_col_names)
    return df

def apply_model(model_path, input_df):
    model = joblib.load(model_path)
    probabilities = model.predict_proba(input_df)
    if isinstance(probabilities, list):
        probabilities = np.array(probabilities)
    probability_dict = {f'class_{i}': probabilities[i][:, 1] for i in range(len(probabilities))}
    return pd.DataFrame(probability_dict)

def adjust_probabilities(probabilities, thresholds):
    return {key: min(100, (val / thresholds[key]) * 100) for key, val in probabilities.items()}

def display_results(df):
    results = {key: df[key].values[0] for key in df.columns}
    adjusted = adjust_probabilities(results, thresholds)
    selected_types = [type_columns_ru[k] for k, v in adjusted.items() if v >= 100]

    st.write("**Наиболее подходящие типы:**")
    for t in selected_types:
        st.write(f"- {t}")

    chart_data = pd.DataFrame({
        "Тип": [type_columns_ru[k] for k in adjusted.keys()],
        "Вероятность": list(adjusted.values())
    })
    st.dataframe(chart_data, use_container_width=True)
    st.bar_chart(chart_data.set_index("Тип"))


def get_ai_response(answers):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {st.secrets['OPENAI_API_KEY']}", "Content-Type": "application/json"}
    data = {
        "model": "ft:gpt-4o-2024-08-06:personal::An4sVvnb",
        "messages": [
            {"role": "system", "content": "Assistant is an expert in career guidance..."},
            {"role": "user", "content": " ".join([f"{i+1}. {a}" for i, a in enumerate(answers)])}
        ],
        "max_tokens": 500
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

@st.cache_data(show_spinner="Загрузка JSONL файлов...")
def load_jsonl_files(folder_path):
    records = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jsonl"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return records

def login_hf():
    if not os.environ.get("HF_TOKEN"):
        login(token=st.secrets["HF_TOKEN"])

@st.cache_resource
def load_annoy_index(rag_data):
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [item["text"] for item in rag_data]
    index = AnnoyIndex(384, 'angular')
    index.load("index.ann")
    return embedder, index, texts

def generate_career_advice(question: str):
    client = InferenceClient(provider="auto", api_key=st.secrets["HF_TOKEN"])
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=[{"role": "system", "content": "Вы — карьерный консультант"}, {"role": "user", "content": question}],
        max_tokens=350, temperature=0.7
    )
    return response.choices[0].message.content

def generate_rag_career_advice(question: str, embedder, index, texts, k=5):
    q_emb = embedder.encode([question], convert_to_numpy=True)[0]
    ctx = "\n\n".join([texts[i] for i in index.get_nns_by_vector(q_emb, k)])
    client = InferenceClient(provider="auto", api_key=st.secrets["HF_TOKEN"])
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=[
            {"role": "system", "content": f"Вы консультант. Контекст:\n{ctx}"},
            {"role": "user", "content": question}
        ],
        max_tokens=500, temperature=0.7
    )
    return response.choices[0].message.content


# ==========================
#  INTERFACE
# ==========================
input_values = {}
rag_data = load_jsonl_files("./jsonl datafiles")
login_hf()
embedder, annoy_index, texts = load_annoy_index(rag_data)

st.header("Профориентация для школьников")
tabs = st.tabs(["Школьные оценки", "Открытые вопросы", "AI профориентатор"])

# TAB 1
with tabs[0]:
    st.write("**Выберите свой мотивационный тип:**")
    selected_checkboxes = {col: st.checkbox(column_names_dict_ru[col]) for col in checkbox_columns}
    for grade in [7, 8, 9, 10]:
        create_expander(grade, [c for c in inp_col_names if c.endswith(f"_{grade}")])

    if st.button("Получить результат"):
        df = save_to_dataframe(selected_checkboxes, input_values)
        result_df = apply_model("random_forest_model.pkl", df)
        st.session_state["tab1_results"] = result_df

    if "tab1_results" in st.session_state:
        display_results(st.session_state["tab1_results"])


# TAB 2
with tabs[1]:
    user_answers = [st.text_input(q, key=f"answer_{i}") for i, q in enumerate(questions_ru)]
    if st.button("Получить ответ"):
        ai_response = get_ai_response(user_answers)
        st.session_state["tab2_ai_response"] = ai_response

    if "tab2_ai_response" in st.session_state:
        st.write("Ответ ИИ:")
        st.write(st.session_state["tab2_ai_response"])


# TAB 3
with tabs[2]:
    st.title("🎓 AI Карьерный консультант")
    student_question = st.text_area("Введите ваш вопрос:", height=100, key="student_q")
    use_rag = st.toggle("Включить RAG", value=True)

    if st.button("Получить совет"):
        if use_rag:
            st.session_state["tab3_base"] = generate_career_advice(student_question)
            st.session_state["tab3_rag"] = generate_rag_career_advice(student_question, embedder, annoy_index, texts)
        else:
            st.session_state["tab3_base"] = generate_career_advice(student_question)

    if "tab3_base" in st.session_state:
        st.subheader("💡 Базовая модель")
        st.write(st.session_state["tab3_base"])
    if use_rag and "tab3_rag" in st.session_state:
        st.subheader("📚 Модель с RAG")
        st.write(st.session_state["tab3_rag"])
