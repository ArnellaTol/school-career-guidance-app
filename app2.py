# app_fixed.py
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
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import io

# ==========================
#  CONFIG
# ==========================
st.set_page_config(page_title="AI-powered program for school career guidance")
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

checkbox_columns = ['Activist', 'Career', 'Tester', 'Creator', 'Designer', 'Researcher']

# типы профессий
type_columns_en = {
    'class_0': 'Person-Sign System',
    'class_1': 'Person-Technology',
    'class_2': 'Person-Nature',
    'class_3': 'Person-Artistic Image',
    'class_4': 'Person-Person',
    'class_5': 'Person-Business'
}
type_columns_ru = {
    'class_0': 'Человек-Знаковая система',
    'class_1': 'Человек-Техника',
    'class_2': 'Человек-Природа',
    'class_3': 'Человек-Художественный образ',
    'class_4': 'Человек-Человек',
    'class_5': 'Человек-Бизнес'
}
type_columns_kz = {
    'class_0': 'Адам-Белгілік жүйе',
    'class_1': 'Адам-Техника',
    'class_2': 'Адам-Табиғат',
    'class_3': 'Адам-Көркем бейне',
    'class_4': 'Адам-Адам',
    'class_5': 'Адам-Бизнес'
}

thresholds = {
    'class_0': 0.39,
    'class_1': 0.30903005409623036,
    'class_2': 0.23611111111111113,
    'class_3': 0.44833333333333336,
    'class_4': 0.13,
    'class_5': 0.17
}

# Названия предметов (en / ru / kz)
column_names_dict_en = {
    'kaz_lang_7': 'Kazakh Language', 'liter_7': 'Literature', 'rus_lang_7': 'Russian Language',
    'eng_lang_7': 'English Language', 'math_7': 'Mathematics', 'comps_7': 'Informatics',
    'kaz_hist_7': 'History of Kazakhstan', 'art_7': 'Art', 'pe_7': 'Physical Education',
    'geography_7': 'Geography', 'biology_7': 'Biology', 'chemistry_7': 'Chemistry',
    'physics_7': 'Physics', 'world_hist_7': 'World History',
    'Activist': 'Activist', 'Career': 'Careerist', 'Tester': 'Tester', 'Creator': 'Creator',
    'Designer': 'Designer', 'Researcher': 'Researcher',
    'kaz_lang_8': 'Kazakh Language', 'liter_8': 'Literature', 'rus_lang_8': 'Russian Language',
    'eng_lang_8': 'English Language', 'math_8': 'Mathematics', 'comps_8': 'Informatics',
    'kaz_hist_8': 'History of Kazakhstan', 'art_8': 'Art', 'pe_8': 'Physical Education',
    'geography_8': 'Geography', 'biology_8': 'Biology', 'chemistry_8': 'Chemistry',
    'physics_8': 'Physics', 'world_hist_8': 'World History',
    'kaz_lang_9': 'Kazakh Language', 'liter_9': 'Literature', 'rus_lang_9': 'Russian Language',
    'eng_lang_9': 'English Language', 'math_9': 'Mathematics', 'comps_9': 'Informatics',
    'kaz_hist_9': 'History of Kazakhstan', 'art_9': 'Art', 'pe_9': 'Physical Education',
    'geography_9': 'Geography', 'biology_9': 'Biology', 'chemistry_9': 'Chemistry',
    'physics_9': 'Physics', 'world_hist_9': 'World History', 'rights_9': 'Law Fundamentals',
    'kaz_lang_10': 'Kazakh Language', 'liter_10': 'Literature', 'rus_lang_10': 'Russian Language',
    'eng_lang_10': 'English Language', 'math_10': 'Mathematics', 'comps_10': 'Informatics',
    'kaz_hist_10': 'History of Kazakhstan', 'art_10': 'Art', 'pe_10': 'Physical Education',
    'geography_10': 'Geography', 'biology_10': 'Biology', 'chemistry_10': 'Chemistry',
    'physics_10': 'Physics', 'world_hist_10': 'World History'
}
column_names_dict_ru = {
    'kaz_lang_7': 'Казахский язык', 'liter_7': 'Литература', 'rus_lang_7': 'Русский язык',
    'eng_lang_7': 'Английский язык', 'math_7': 'Математика', 'comps_7': 'Информатика',
    'kaz_hist_7': 'История Казахстана', 'art_7': 'Искусство', 'pe_7': 'Физкультура',
    'geography_7': 'География', 'biology_7': 'Биология', 'chemistry_7': 'Химия',
    'physics_7': 'Физика', 'world_hist_7': 'Всемирная история',
    'Activist': 'Активист', 'Career': 'Карьерист', 'Tester': 'Испытатель', 'Creator': 'Творец',
    'Designer': 'Проектировщик', 'Researcher': 'Исследователь',
    'kaz_lang_8': 'Казахский язык', 'liter_8': 'Литература', 'rus_lang_8': 'Русский язык',
    'eng_lang_8': 'Английский язык', 'math_8': 'Математика', 'comps_8': 'Информатика',
    'kaz_hist_8': 'История Казахстана', 'art_8': 'Искусство', 'pe_8': 'Физкультура',
    'geography_8': 'География', 'biology_8': 'Биология', 'chemistry_8': 'Химия',
    'physics_8': 'Физика', 'world_hist_8': 'Всемирная история',
    'kaz_lang_9': 'Казахский язык', 'liter_9': 'Литература', 'rus_lang_9': 'Русский язык',
    'eng_lang_9': 'Английский язык', 'math_9': 'Математика', 'comps_9': 'Информатика',
    'kaz_hist_9': 'История Казахстана', 'art_9': 'Искусство', 'pe_9': 'Физкультура',
    'geography_9': 'География', 'biology_9': 'Биология', 'chemistry_9': 'Химия',
    'physics_9': 'Физика', 'world_hist_9': 'Всемирная история', 'rights_9': 'Основы права',
    'kaz_lang_10': 'Казахский язык', 'liter_10': 'Литература', 'rus_lang_10': 'Русский язык',
    'eng_lang_10': 'Английский язык', 'math_10': 'Математика', 'comps_10': 'Информатика',
    'kaz_hist_10': 'История Казахстана', 'art_10': 'Искусство', 'pe_10': 'Физкультура',
    'geography_10': 'География', 'biology_10': 'Биология', 'chemistry_10': 'Химия',
    'physics_10': 'Физика', 'world_hist_10': 'Всемирная история'
}
column_names_dict_kz = {
    'kaz_lang_7': 'Қазақ тілі', 'liter_7': 'Әдебиет', 'rus_lang_7': 'Орыс тілі',
    'eng_lang_7': 'Ағылшын тілі', 'math_7': 'Математика', 'comps_7': 'Информатика',
    'kaz_hist_7': 'Қазақстан тарихы', 'art_7': 'Бейнелеу өнері', 'pe_7': 'Дене шынықтыру',
    'geography_7': 'География', 'biology_7': 'Биология', 'chemistry_7': 'Химия',
    'physics_7': 'Физика', 'world_hist_7': 'Дүниежүзі тарихы',
    'Activist': 'Белсенді', 'Career': 'Мамандық таңдаушы', 'Tester': 'Тексеруші',
    'Creator': 'Жасаушы', 'Designer': 'Дизайнер', 'Researcher': 'Зерттеуші',
    'kaz_lang_8': 'Қазақ тілі', 'liter_8': 'Әдебиет', 'rus_lang_8': 'Орыс тілі',
    'eng_lang_8': 'Ағылшын тілі', 'math_8': 'Математика', 'comps_8': 'Информатика',
    'kaz_hist_8': 'Қазақстан тарихы', 'art_8': 'Бейнелеу өнері', 'pe_8': 'Дене шынықтыру',
    'geography_8': 'География', 'biology_8': 'Биология', 'chemistry_8': 'Химия',
    'physics_8': 'Физика', 'world_hist_8': 'Дүниежүзі тарихы',
    'kaz_lang_9': 'Қазақ тілі', 'liter_9': 'Әдебиет', 'rus_lang_9': 'Орыс тілі',
    'eng_lang_9': 'Ағылшын тілі', 'math_9': 'Математика', 'comps_9': 'Информатика',
    'kaz_hist_9': 'Қазақстан тарихы', 'art_9': 'Бейнелеу өнері', 'pe_9': 'Дене шынықтыру',
    'geography_9': 'География', 'biology_9': 'Биология', 'chemistry_9': 'Химия',
    'physics_9': 'Физика', 'world_hist_9': 'Дүниежүзі тарихы', 'rights_9': 'Құқық негіздері',
    'kaz_lang_10': 'Қазақ тілі', 'liter_10': 'Әдебиет', 'rus_lang_10': 'Орыс тілі',
    'eng_lang_10': 'Ағылшын тілі', 'math_10': 'Математика', 'comps_10': 'Информатика',
    'kaz_hist_10': 'Қазақстан тарихы', 'art_10': 'Бейнелеу өнері', 'pe_10': 'Дене шынықтыру',
    'geography_10': 'География', 'biology_10': 'Биология', 'chemistry_10': 'Химия',
    'physics_10': 'Физика', 'world_hist_10': 'Дүниежүзі тарихы'
}

# QUESTIONS / EXPANDER TEXTS (previously lang_dict)
lang_meta = {
    "ru": {
        "expander": "Введите оценки за {grade} класс:",
        "most_suitable": "Наиболее подходящие типы:",
        "probability": "Вероятность",
        "type": "Тип",
        "questions": [
            "**1. Какие профессии вас интересуют на данный момент?**",
            "**2. Какие виды деятельности вам точно не интересны?**",
            "**3. Без учета финансовых аспектов, какие виды деятельности или профессии вам нравятся?**",
            "**4. Перечислите свои хобби и интересы:**",
            "**5. Назовите ролевые модели, чьи образы жизни и достижения вас вдохновляют.**",
            "**6. Какие задачи придают вам энергии?**",
            "**7. Какие задачи вас утомляют?**"
        ]
    },
    "en": {
        "expander": "Enter grades for grade {grade}:",
        "most_suitable": "Most suitable types:",
        "probability": "Probability",
        "type": "Type",
        "questions": [
            "**1. Which professions are you currently interested in?**",
            "**2. Which activities are you definitely not interested in?**",
            "**3. Regardless of finances, which activities or professions do you enjoy?**",
            "**4. List your hobbies and interests:**",
            "**5. Name role models whose lifestyles and achievements inspire you.**",
            "**6. Which tasks give you energy?**",
            "**7. Which tasks drain your energy?**"
        ]
    },
    "kz": {
        "expander": "{grade}-сынып бағаларын енгізіңіз:",
        "most_suitable": "Ең қолайлы типтер:",
        "probability": "Ықтималдық",
        "type": "Түрі",
        "questions": [
            "**1. Қазір сізді қандай мамандықтар қызықтырады?**",
            "**2. Сізге мүлдем қызық емес іс-әрекеттер қандай?**",
            "**3. Қаржылық аспектілерді есептемегенде, қандай іс-әрекеттер немесе мамандықтар ұнайды?**",
            "**4. Хоббиіңіз бен қызығушылықтарыңызды жазыңыз:**",
            "**5. Сізді өмір салты мен жетістіктерімен шабыттандыратын тұлғаларды атаңыз.**",
            "**6. Сізге күш-қуат беретін тапсырмалар қандай?**",
            "**7. Сізді шаршататын тапсырмалар қандай?**"
        ]
    }
}

# UI translations
translations = {
    "en": {
        "header": "AI-powered program for school career guidance",
        "tab1": "School grades",
        "tab2": "Open questions",
        "tab3": "AI career assistant",
        "choose_type": "Choose your motivational type:",
        "get_result": "Get result",
        "most_suitable": "Most suitable types:",
        "get_answer": "Get answer",
        "ai_response": "AI Response:",
        "advisor": "🎓 Career Guidance AI Assistant",
        "student_question": "Enter your question:",
        "rag_toggle": "Enable RAG",
        "get_advice": "Get advice",
        "base_model": "💡 Base model",
        "rag_model": "📚 Model with RAG",
        "expander": "Grades for {grade} grade",
        "questions": lang_meta["en"]["questions"]
    },
    "ru": {
        "header": "ИИ программа для школьной профориентации",
        "tab1": "Школьные оценки",
        "tab2": "Открытые вопросы",
        "tab3": "AI профориентатор",
        "choose_type": "Выберите свой мотивационный тип:",
        "get_result": "Получить результат",
        "most_suitable": "Наиболее подходящие типы:",
        "get_answer": "Получить ответ",
        "ai_response": "Ответ ИИ:",
        "advisor": "🎓 Профориентационный AI ассистент",
        "student_question": "Введите ваш вопрос:",
        "rag_toggle": "Включить RAG",
        "get_advice": "Получить совет",
        "base_model": "💡 Базовая модель",
        "rag_model": "📚 Модель с RAG",
        "expander": "Оценки за {grade} класс",
        "questions": lang_meta["ru"]["questions"]
    },
    "kz": {
        "header": "Мектептік кәсіби бағдар беруге арналған ЖИ бағдарлама",
        "tab1": "Мектеп бағалары",
        "tab2": "Ашық сұрақтар",
        "tab3": "ЖИ кәсіби бағдаршы",
        "choose_type": "Өз мотивациялық типіңізді таңдаңыз:",
        "get_result": "Нәтиже алу",
        "most_suitable": "Ең қолайлы типтер:",
        "get_answer": "Жауап алу",
        "ai_response": "ЖИ жауабы:",
        "advisor": "🎓 Кәсіби бағдар беретін ЖИ ассистенті",
        "student_question": "Сұрағыңызды енгізіңіз:",
        "rag_toggle": "RAG қосу",
        "get_advice": "Кеңес алу",
        "base_model": "💡 Негізгі модель",
        "rag_model": "📚 RAG моделі",
        "expander": "{grade} сынып бағалары",
        "questions": lang_meta["kz"]["questions"]
    }
}

# ==========================
#  HELPERS / MODEL / RAG
# ==========================
def create_expander(class_label, cols, lang_meta_dict, column_names_dict, input_values):
    """Создаёт expander для оценок; input_values - dict куда пишем."""
    with st.expander(lang_meta_dict["expander"].format(grade=class_label)):
        for col in cols:
            input_values[col] = st.number_input(
                column_names_dict[col], min_value=2, max_value=5, step=1, value=5, key=col
            )

def save_to_dataframe(selected_checkboxes, input_values):
    data = {**selected_checkboxes, **input_values}
    for key in checkbox_columns:
        data[key] = int(data.get(key, False))
    df = pd.DataFrame([data], columns=inp_col_names)
    return df

def apply_model(model_path, input_df):
    """Загрузка модели и получение вероятностей.
    Подстраховка: если model.predict_proba возвращает список или array."""
    model = joblib.load(model_path)
    probabilities = model.predict_proba(input_df)

    # handle case when predict_proba returns list of arrays (OneVsRest style)
    if isinstance(probabilities, list):
        # each element is (n_samples, 2) — take [:,1]
        probs = np.vstack([arr[:, 1] for arr in probabilities]).T  # (n_samples, n_classes)
    else:
        probs = np.array(probabilities)  # (n_samples, n_classes)

    # build dict class_i -> column
    probability_dict = {f'class_{i}': probs[:, i] for i in range(probs.shape[1])}
    return pd.DataFrame(probability_dict)

def adjust_probabilities(probabilities, thresholds):
    return {key: min(100, (val / thresholds.get(key, 1e-9)) * 100) for key, val in probabilities.items()}

def display_results(df, lang_meta_dict, type_columns_dict):
    results = {key: df[key].values[0] for key in df.columns}
    adjusted = adjust_probabilities(results, thresholds)
    # selected types where adjusted >= 100
    selected_types = [type_columns_dict.get(k, k) for k, v in adjusted.items() if v >= 100]

    st.write(f"**{lang_meta_dict['most_suitable']}**")
    for t_name in selected_types:
        st.write(f"- {t_name}")

    # DataFrame for chart: use labels from lang_meta_dict["type"] and ["probability"]
    chart_data = pd.DataFrame({
        lang_meta_dict["type"]: [type_columns_dict.get(k, k) for k in adjusted.keys()],
        lang_meta_dict["probability"]: list(adjusted.values())
    })
    st.dataframe(chart_data, use_container_width=True)
    st.bar_chart(chart_data.set_index(lang_meta_dict["type"]))

def get_ai_response(answers):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {st.secrets['OPENAI_API_KEY']}", "Content-Type": "application/json"}
    data = {
        "model": "ft:gpt-4o-2024-08-06:personal::An4sVvnb",
        "messages": [
            {
                "role": "system",
                "content": (
                    "Assistant is an expert in career guidance. Assistant should answer in the same language "
                    "as the one in which the user writes answers (could be russian, kazakh, english). "
                    "User answers the following questions: "
                    "1. Какие профессии вас интересуют на данный момент? "
                    "2. Какие виды деятельности вам точно не интересны? "
                    "3. Без учета финансовых аспектов, какие виды деятельности или профессии вам нравятся? "
                    "4. Перечислите свои хобби и интересы: "
                    "5. Назовите ролевые модели, чей образ жизни и достижения вас вдохновляют. "
                    "6. Какие задачи придают вам энергии? "
                    "7. Какие задачи вас утомляют?"
                )
            },
            {
                "role": "user",
                "content": f"1. {answers[0]} 2. {answers[1]} 3. {answers[2]} 4. {answers[3]} 5. {answers[4]} 6. {answers[5]} 7. {answers[6]}"
            }
        ],
        "max_tokens": 500
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

@st.cache_data(show_spinner="Loading...")
def load_jsonl_files(folder_path):
    records = []
    if not os.path.exists(folder_path):
        return records
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
    texts = [item.get("text", "") for item in rag_data]
    index = AnnoyIndex(384, 'angular')
    if os.path.exists("index.ann"):
        index.load("index.ann")
    return embedder, index, texts

def generate_rag_career_advice(question: str, embedder, annoy_index, texts: list, k: int = 5) -> str:
    query_embedding = embedder.encode([question], convert_to_numpy=True)
    indices = annoy_index.get_nns_by_vector(query_embedding[0], k, include_distances=False)
    context_docs = [texts[i] for i in indices if i < len(texts)]
    context = "\n\n".join(context_docs)
    messages = [ {"role": "system", "content": f""" You are a career advisor for high school students. You have access to relevant background knowledge about career paths, student preferences, and educational strategies, shown below. Context: {context} Your only task is to select 3 career paths that are the best possible match for the student's stated interests, strengths, and dislikes. Strict instructions: - Base your suggestions strictly on the student’s message. Do not invent or assume anything not mentioned. - Recommend only career paths that clearly align with what the student enjoys and is good at, and that avoid what they dislike or find difficult. - For each suggested path, explain in 3-4 sentences why it fits this student specifically. - Do not give general advice or list unrelated options "just in case." - Keep the total response under 350 words. Be focused and relevant. If student asks other questions, answer them directly (still use the background context) and do not generate career paths if not asked. """}, {"role": "user", "content": question} ]
    client = InferenceClient(provider="auto", api_key=st.secrets["HF_TOKEN"])
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=messages,
        max_tokens=500,
        temperature=0.7
    )
    answer = response.choices[0].message.content
    if not answer.endswith("."):
        last_period = answer.rfind(".")
        if last_period != -1:
            answer = answer[:last_period + 1]
        else:
            answer = answer.strip()
    return answer

# ==========================
#  PDF SAVE HELPERS
# ==========================
def save_tab1_results_to_pdf(results_df, lang):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 50, {
        "ru": "Результаты профориентационного теста (оценки)",
        "en": "Career guidance results (grades)",
        "kz": "Кәсіби бағдар нәтижелері (бағалар)"
    }[lang])
    c.setFont("Helvetica", 12)
    y = height - 100
    for idx, row in results_df.iterrows():
        for col, val in row.items():
            c.drawString(50, y, f"{col}: {val}")
            y -= 20
            if y < 80:
                c.showPage()
                y = height - 50
    c.save()
    buffer.seek(0)
    return buffer

def save_tab2_results_to_pdf(questions, answers, ai_response, lang):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 50, {
        "ru": "Открытые вопросы и анализ ИИ",
        "en": "Open questions and AI analysis",
        "kz": "Ашық сұрақтар мен ЖИ талдауы"
    }[lang])
    c.setFont("Helvetica", 12)
    y = height - 100
    for q, a in zip(questions, answers):
        c.drawString(50, y, f"Q: {q}")
        y -= 18
        c.drawString(70, y, f"A: {a}")
        y -= 28
        if y < 80:
            c.showPage()
            y = height - 50
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, {"ru": "Ответ ИИ:", "en": "AI Response:", "kz": "ЖИ жауабы:"}[lang])
    y -= 18
    c.setFont("Helvetica", 12)
    for line in ai_response.splitlines():
        c.drawString(50, y, line)
        y -= 16
        if y < 80:
            c.showPage()
            y = height - 50
    c.save()
    buffer.seek(0)
    return buffer

def save_tab3_results_to_pdf(question, rag_response, lang):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 50, {
        "ru": "AI профориентация (RAG модель)",
        "en": "AI career guidance (RAG model)",
        "kz": "ЖИ кәсіби бағдар (RAG моделі)"
    }[lang])
    c.setFont("Helvetica", 12)
    y = height - 100
    c.drawString(50, y, {"ru": "Вопрос ученика:", "en": "Student's question:", "kz": "Оқушы сұрағы:"}[lang])
    y -= 18
    for line in question.splitlines():
        c.drawString(70, y, line)
        y -= 16
        if y < 80:
            c.showPage()
            y = height - 50
    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, {"ru": "Ответ RAG модели:", "en": "RAG model response:", "kz": "RAG модель жауабы:"}[lang])
    y -= 18
    c.setFont("Helvetica", 12)
    for line in rag_response.splitlines():
        c.drawString(50, y, line)
        y -= 16
        if y < 80:
            c.showPage()
            y = height - 50
    c.save()
    buffer.seek(0)
    return buffer

# ==========================
#  INTERFACE
# ==========================
# словарь для отображения (label → value)
lang_options = {"KZ": "kz", "EN": "en", "RU": "ru"}

# селектор справа сверху: создаём 3 колонки, правая узкая
col1, col2, col3 = st.columns([8, 1, 2])
with col3:
    lang_label = st.selectbox(" ", options=list(lang_options.keys()), index=0)

# определяем lang в нижнем регистре и сохраняем в session_state
lang = lang_options[lang_label]
st.session_state["lang"] = lang  # всегда обновляем — безопасно

# t = UI translations, ld = questions/expander strings
t = translations[lang]
ld = lang_meta[lang]

# столбцы названий предметов по языку
column_names_dicts = {"ru": column_names_dict_ru, "en": column_names_dict_en, "kz": column_names_dict_kz}
current_column_names = column_names_dicts[lang]

# загружаем данные RAG и индекс (кэшируются)
input_values = {}
rag_data = load_jsonl_files("./jsonl datafiles")
login_hf()
embedder, annoy_index, texts = load_annoy_index(rag_data)

st.header(t["header"])
tabs = st.tabs([t["tab1"], t["tab2"], t["tab3"]])

# ------------------------
# TAB 1 - Grades
# ------------------------
with tabs[0]:
    with st.form("grades_form"):
        st.write(f"**{t['choose_type']}**")
        selected_checkboxes = {
            col: st.checkbox(current_column_names[col]) for col in checkbox_columns
        }
        for grade in [7, 8, 9, 10]:
            create_expander(
                grade,
                [c for c in inp_col_names if c.endswith(f"_{grade}")],
                ld,
                current_column_names,
                input_values
            )
        submit_tab1 = st.form_submit_button(t["get_result"])

    if submit_tab1:
        df = save_to_dataframe(selected_checkboxes, input_values)
        try:
            result_df = apply_model("random_forest_model.pkl", df)
        except Exception as e:
            st.error(f"Error applying model: {e}")
            result_df = None
        st.session_state["tab1_results"] = result_df

    if "tab1_results" in st.session_state and st.session_state["tab1_results"] is not None:
        # pick right type_columns dict
        if lang == "ru":
            type_columns_dict = type_columns_ru
        elif lang == "kz":
            type_columns_dict = type_columns_kz
        else:
            type_columns_dict = type_columns_en

        display_results(st.session_state["tab1_results"], ld, type_columns_dict)

        # description block (localized)
        if lang == "ru":
            st.title("Типы профессиональной деятельности")
            st.markdown("""
**1. ЧЕЛОВЕК-ЖИВАЯ ПРИРОДА (П).**  
Представители этого типа имеют дело с растительными и живыми организмами, микроорганизмами и условиями их существования  
*(агроном, ветврач, полевод, животновод, кинолог, фермер, геолог)*.

---

**2. ЧЕЛОВЕК-ТЕХНИКА И НЕЖИВАЯ ПРИРОДА (Т).**  
Работники имеют дело с неживыми и техническими объектами труда  
*(слесарь, автомеханик, водитель, инженер, моторист, плотник, штукатур, сварщик, конструктор, контролер, физик, химик)*.

---

**3. ЧЕЛОВЕК-ЧЕЛОВЕК (Ч).**  
Предметом интереса, распознания, обслуживания, преобразования здесь являются социальные системы, сообщества, группы населения, люди разного возраста  
*(учитель, менеджер, врач, страховой агент, воспитатель, няня, продавец, социальный работник, массажист, психолог)*.

---

**4. ЧЕЛОВЕК-ЗНАКОВАЯ СИСТЕМА (З).**  
Естественные и искусственные языки, условные знаки, символы, формулы — вот предметные миры, которые занимают представителей этого типа  
*(бухгалтер, программист, оператор ПК, радиомонтажник, экономист, телефонист, машинистка, переводчик, кассир)*.

---

**5. ЧЕЛОВЕК-ХУДОЖЕСТВЕННЫЙ ОБРАЗ (Х).**  
Явления, факты художественного отображения действительности — вот что занимает представителей этого типа  
*(артист, дирижер, художник, маляр, портной, повар, парикмахер, музыкант, архитектор)*.

---

**6. ЧЕЛОВЕК-БИЗНЕС (Б).**  
Выделен в последнее время в связи с потребностью рынка труда.  
Сюда относятся специальности: *менеджеры, биржевые маклеры, аудиторы, брокеры, дилеры и другие профессии, связанные с коммерческой деятельностью*.
""")

        elif lang == "en":
            st.title("Types of professional activities")
            st.markdown("""
**1. HUMAN–NATURE (N).**  
Work with plants, animals, microorganisms, and their living conditions  
*(agronomist, veterinarian, farmer, dog handler, geologist)*.

---

**2. HUMAN–TECHNOLOGY (T).**  
Work with inanimate objects and technical systems  
*(mechanic, driver, engineer, carpenter, welder, constructor, physicist, chemist)*.

---

**3. HUMAN–HUMAN (H).**  
Work with people, communities, social systems  
*(teacher, manager, doctor, nanny, salesperson, psychologist, social worker)*.

---

**4. HUMAN–SIGN SYSTEMS (S).**  
Work with languages, signs, symbols, codes, formulas  
*(accountant, programmer, operator, economist, translator, cashier)*.

---

**5. HUMAN–ARTISTIC IMAGE (A).**  
Work with artistic creation and representation of reality  
*(actor, conductor, painter, tailor, chef, musician, architect)*.

---

**6. HUMAN–BUSINESS (B).**  
A newer type reflecting labor market demand  
*(managers, brokers, dealers, auditors, entrepreneurs)*.
""")

        elif lang == "kz":
            st.title("Кәсіби қызмет түрлері")
            st.markdown("""
**1. АДАМ–ТІРІ ТАБИҒАТ (Т).**  
Өсімдіктермен, жануарлармен, микроорганизмдермен және олардың тіршілік жағдайларымен жұмыс  
*(агроном, ветеринар, малшы, кинолог, фермер, геолог)*.

---

**2. АДАМ–ТЕХНИКА ЖӘНЕ ӨЛІ ТАБИҒАТ (Т).**  
Өлі және техникалық еңбек объектілерімен жұмыс  
*(слесарь, механик, жүргізуші, инженер, ағаш ұстасы, дәнекерлеуші, физик, химик)*.

---

**3. АДАМ–АДАМ (А).**  
Қоғамдық жүйелермен, қауымдармен, әртүрлі жастағы адамдармен жұмыс  
*(мұғалім, менеджер, дәрігер, тәрбиеші, сатушы, әлеуметтік қызметкер, массажист, психолог)*.

---

**4. АДАМ–БЕЛГІЛІК ЖҮЙЕ (Б).**  
Тілдермен, таңбалармен, формулалармен жұмыс  
*(бухгалтер, бағдарламашы, экономист, аудармашы, кассир)*.

---

**5. АДАМ–КӨРКЕМ БЕЙНЕ (К).**  
Шығармашылық, өнер арқылы шындықты бейнелеу  
*(әртіс, дирижер, суретші, тігінші, аспаз, музыкант, сәулетші)*.

---

**6. АДАМ–БИЗНЕС (Б).**  
Еңбек нарығының сұранысына байланысты жаңа бағыт  
*(менеджерлер, брокерлер, дилерлер, аудиторлар, кәсіпкерлер)*.
""")
        # download PDF button for tab1
        if st.button({"ru": "Сохранить результаты в PDF", "en": "Save results to PDF", "kz": "Нәтижені PDF-қа сақтау"}[lang]):
            if "tab1_results" in st.session_state and st.session_state["tab1_results"] is not None:
                pdf_buf = save_tab1_results_to_pdf(st.session_state["tab1_results"], lang)
                st.download_button(
                    label={"ru": "Скачать PDF", "en": "Download PDF", "kz": "Жүктеу PDF"}[lang],
                    data=pdf_buf,
                    file_name="tab1_results.pdf",
                    mime="application/pdf"
                )
            else:
                st.warning({"ru": "Нет результатов для сохранения", "en": "No results to save", "kz": "Сақтауға нәтиже жоқ"}[lang])

# ------------------------
# TAB 2 - Open questions
# ------------------------
with tabs[1]:
    with st.form("open_questions_form"):
        user_answers = [st.text_input(q, key=f"answer_{i}") for i, q in enumerate(ld["questions"])]
        submit_tab2 = st.form_submit_button(t["get_answer"])

    if submit_tab2:
        ai_response = get_ai_response(user_answers)
        st.session_state["tab2_ai_response"] = ai_response

    if "tab2_ai_response" in st.session_state:
        st.write(t["ai_response"])
        st.write(st.session_state["tab2_ai_response"])

        # кнопка сохранить (скачать) — генерируем PDF и показываем кнопку
        pdf_buffer = save_tab2_results_to_pdf(
            ld["questions"],
            [st.session_state.get(f"answer_{i}", "") for i in range(len(ld["questions"]))],
            st.session_state["tab2_ai_response"],
            lang
        )
        st.download_button(
            label={"ru": "Сохранить в PDF", "en": "Save as PDF", "kz": "PDF сақтау"}[lang],
            data=pdf_buffer,
            file_name="tab2_results.pdf",
            mime="application/pdf"
        )

# ------------------------
# TAB 3 - AI career (RAG only)
# ------------------------
with tabs[2]:
    with st.form("career_form"):
        st.title(t["advisor"])
        student_question = st.text_area(t["student_question"], height=100, key="student_q")
        submit_tab3 = st.form_submit_button(t["get_advice"])

    if submit_tab3:
        rag_answer = generate_rag_career_advice(student_question, embedder, annoy_index, texts)
        st.session_state["tab3_rag"] = rag_answer
        # save student question into session for PDF
        st.session_state["student_q"] = student_question

    if "tab3_rag" in st.session_state:
        st.subheader(t["rag_model"])
        st.write(st.session_state["tab3_rag"])
        pdf_buffer = save_tab3_results_to_pdf(
            st.session_state.get("student_q", ""),
            st.session_state["tab3_rag"],
            lang
        )
        st.download_button(
            label={"ru": "Сохранить в PDF", "en": "Save as PDF", "kz": "PDF сақтау"}[lang],
            data=pdf_buffer,
            file_name="tab3_results.pdf",
            mime="application/pdf"
        )
