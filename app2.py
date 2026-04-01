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
import io
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

import matplotlib.pyplot as plt

styles = getSampleStyleSheet()

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
    'Activist': 'Белсенді', 'Career': 'Мансапқор', 'Tester': 'Тексеруші',
    'Creator': 'Жасаушы', 'Designer': 'Жоспарлаушы', 'Researcher': 'Зерттеуші',
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
        "most_suitable": "Ең қолайлы түрлері:",
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
        "tab4": "Motivational type test",
        "choose_type": "Choose your motivational type:",
        "current_grade_label": "What grade are you currently in?",
        "go_to_tab": "If you don't know your type, go to the 'Motivational type test' tab.",
        "get_result": "Get result",
        "most_suitable": "Most suitable types:",
        "get_answer": "Get answer",
        "ai_response": "AI Response:",
        "advisor": "Career Guidance AI Assistant",
        "student_question": "Enter your question:",
        "rag_toggle": "Enable RAG",
        "get_advice": "Get advice",
        "base_model": "💡 Base model",
        "rag_model": "RAG model response:",
        "expander": "Grades for {grade} grade",
        "questions": lang_meta["en"]["questions"]
    },
    "ru": {
        "header": "ИИ программа для школьной профориентации",
        "tab1": "Школьные оценки",
        "tab2": "Открытые вопросы",
        "tab3": "ИИ профориентатор",
        "tab4": "Тест на мотивационный тип",
        "choose_type": "Выберите свой мотивационный тип:",
        "current_grade_label": "В каком вы сейчас классе?",
        "go_to_tab": "Если вы не знаете свой тип, перейдите на вкладку 'Тест на мотивационный тип'.",
        "get_result": "Получить результат",
        "most_suitable": "Наиболее подходящие типы:",
        "get_answer": "Получить ответ",
        "ai_response": "Ответ ИИ:",
        "advisor": "Профориентационный ИИ ассистент",
        "student_question": "Введите ваш вопрос:",
        "rag_toggle": "Включить RAG",
        "get_advice": "Получить совет",
        "base_model": "💡 Базовая модель",
        "rag_model": "Ответ RAG модели:",
        "expander": "Оценки за {grade} класс",
        "questions": lang_meta["ru"]["questions"]
    },
    "kz": {
        "header": "Мектептік кәсіби бағдар беруге арналған ЖИ бағдарлама",
        "tab1": "Мектеп бағалары",
        "tab2": "Ашық сұрақтар",
        "tab3": "ЖИ кәсіби бағдаршы",
        "tab4": "Мотивациялық типті тест",
        "choose_type": "Өз мотивациялық типіңізді таңдаңыз:",
        "current_grade_label": "Қазір қай сыныпта оқисыз?",
        "go_to_tab": "Егер сіз өз типіңізді білмесеңіз, 'Мотивациялық типті тест' қойындысына өтіңіз.",
        "get_result": "Нәтиже алу",
        "most_suitable": "Ең қолайлы түрлері:",
        "get_answer": "Жауап алу",
        "ai_response": "ЖИ жауабы:",
        "advisor": "Кәсіби бағдар беретін ЖИ ассистенті",
        "student_question": "Сұрағыңызды енгізіңіз:",
        "rag_toggle": "RAG қосу",
        "get_advice": "Кеңес алу",
        "base_model": "💡 Негізгі модель",
        "rag_model": "RAG модельінің жауабы:",
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

def get_available_grades(current_grade):
    """Возвращает список классов, доступных для ручного ввода."""
    if current_grade == "7":
        return [7]
    if current_grade == "8":
        return [7, 8]
    if current_grade == "9":
        return [7, 8, 9]
    # 10-12
    return [7, 8, 9, 10]


def get_grade_subjects(grade, column_names_dict):
    """Возвращает список ключей предметов для класса grade."""
    return [col for col in inp_col_names if col.endswith(f"_{grade}") and col in column_names_dict]


def autofill_grades(prev_grade, next_grade, input_values, column_names_dict):
    """Автозаполняет оценки next_grade по данным prev_grade."""
    prev_subjects = get_grade_subjects(prev_grade, column_names_dict)
    next_subjects = get_grade_subjects(next_grade, column_names_dict)

    prev_map = {subj.rsplit("_", 1)[0]: input_values.get(subj, 5) for subj in prev_subjects}

    filled = {}
    for subj in next_subjects:
        slim = subj.rsplit("_", 1)[0]
        filled[subj] = prev_map.get(slim, 5)

    return filled


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
    """
    Показывает таблицу и рисует bar chart через matplotlib.
    Возвращает BytesIO с PNG изображением графика (seeked to 0).
    """
    # подготовка данных
    results = {key: df[key].values[0] for key in df.columns}
    adjusted = adjust_probabilities(results, thresholds)

    # selected types where adjusted >= 100 (для списка подходящих типов)
    selected_types = [type_columns_dict.get(k, k) for k, v in adjusted.items() if v >= 100]

    st.write(f"**{lang_meta_dict['most_suitable']}**")
    for t_name in selected_types:
        st.write(f"- {t_name}")

    # создаем DataFrame для таблицы/графика
    types_list = [type_columns_dict.get(k, k) for k in adjusted.keys()]
    probs_list = [round(v, 2) for v in adjusted.values()]  # округление до сотых

    chart_data = pd.DataFrame({
        lang_meta_dict["type"]: types_list,
        lang_meta_dict["probability"]: probs_list
    })

    # фиксируем порядок категорий как в таблице
    chart_data[lang_meta_dict["type"]] = pd.Categorical(
        chart_data[lang_meta_dict["type"]],
        categories=types_list,
        ordered=True
    )

    st.dataframe(chart_data, use_container_width=True)

    # --- создаём matplotlib график (чтобы точно иметь изображение в нужном порядке) ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(types_list, probs_list)
    ax.set_xlabel(lang_meta_dict["type"])
    ax.set_ylabel(lang_meta_dict["probability"])
    ax.set_ylim(0, 110)  # чуть выше 100% для визуала
    ax.set_xticklabels(types_list, rotation=30, ha='right')
    fig.tight_layout()

    # сохраняем в BytesIO
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    img_buf.seek(0)

    # показываем график в Streamlit (тот же рисунок)
    st.image(img_buf)

    # вернуть буфер с изображением (чтобы потом положить в PDF)
    img_buf.seek(0)
    return img_buf

TRANSLATOR_PROMPT = """You are a professional translator and text corrector.
- Your only task is to translate any input text into clear, fluent English.
- If the text is already in English, correct grammar, spelling, and style issues.
- If the text mixes languages, translate everything into English.
- Do not add explanations, notes, or any extra content. Return only the translated and corrected text."""


def translate_to_english(text: str) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {st.secrets['OPENAI_API_KEY']}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4o-mini",  # лёгкая и быстрая модель для переводов
        "messages": [
            {"role": "system", "content": TRANSLATOR_PROMPT},
            {"role": "user", "content": text}
        ],
        "max_tokens": 1000,
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()

    try:
        response_json = response.json()
        return response_json["choices"][0]["message"]["content"].strip()
    except Exception:
        return text  # fallback: возвращаем оригинал


def translate_from_english(text: str, lang: str) -> str:
    if lang == "en":
        return text  # если интерфейс на английском — перевод не нужен

    targets = {
        "ru": "Russian",
        "kz": "Kazakh"
    }
    target_lang = targets.get(lang, "English")

    prompt = f"""Translate the following text from English into {target_lang}.
- Keep meaning accurate and natural.
- Do not add explanations or extra text."""

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {st.secrets['OPENAI_API_KEY']}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text}
        ],
        "max_tokens": 800,
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()

    try:
        response_json = response.json()
        return response_json["choices"][0]["message"]["content"].strip()
    except Exception:
        return text



def get_ai_response(answers, lang="en"):
    # --- переводим ответы ученика ---
    translated_answers = [translate_to_english(a) for a in answers]

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {st.secrets['OPENAI_API_KEY']}", "Content-Type": "application/json"}
    data = {
        "model": "ft:gpt-4o-2024-08-06:personal::An4sVvnb",
        "messages": [
            {
                "role": "system",
                "content": (
                    "Assistant is an expert in career guidance. Assistant should answer in english language and keep the total response under 350 words. "
                    "User answers the following questions: "
                    "1. Which professions are you currently interested in? "
                    "2. Which activities are you definitely not interested in? "
                    "3. Regardless of finances, which activities or professions do you enjoy? "
                    "4. List your hobbies and interests: "
                    "5. Name role models whose lifestyles and achievements inspire you. "
                    "6. Which tasks give you energy? "
                    "7. Which tasks drain your energy?"
                )
            },
            {
                "role": "user",
                "content": f"1. {translated_answers[0]} 2. {translated_answers[1]} 3. {translated_answers[2]} 4. {translated_answers[3]} 5. {translated_answers[4]} 6. {translated_answers[5]} 7. {translated_answers[6]}"
            }
        ],
        "max_tokens": 550
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    answer_en = response.json()["choices"][0]["message"]["content"].strip()

    # --- переводим ответ обратно ---
    return translate_from_english(answer_en, lang)


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

def generate_rag_career_advice(question: str, embedder, annoy_index, texts: list, lang="en", k: int = 5) -> str:
    # --- перевод вопроса в английский ---
    translated_question = translate_to_english(question)

    query_embedding = embedder.encode([translated_question], convert_to_numpy=True)
    indices = annoy_index.get_nns_by_vector(query_embedding[0], k, include_distances=False)
    context_docs = [texts[i] for i in indices if i < len(texts)]
    context = "\n\n".join(context_docs)

    messages = [
        {"role": "system", "content": f""" You are a career advisor for high school students. You have access to relevant background knowledge about career paths, student preferences, and educational strategies, shown below. Context: {context} Your only task is to select 3 career paths that are the best possible match for the student's stated interests, strengths, and dislikes. Strict instructions: - Base your suggestions strictly on the student’s message. Do not invent or assume anything not mentioned. - Recommend only career paths that clearly align with what the student enjoys and is good at, and that avoid what they dislike or find difficult. - For each suggested path, explain in 3-4 sentences why it fits this student specifically. - Do not give general advice or list unrelated options "just in case." - Keep the total response under 350 words. Be focused and relevant. If student asks other questions, answer them directly (still use the background context) and do not generate career paths if not asked. """},
        {"role": "user", "content": translated_question}
    ]

    client = InferenceClient(provider="auto", api_key=st.secrets["HF_TOKEN"])
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=messages,
        max_tokens=550,
        temperature=0.7
    )

    answer_en = response.choices[0].message.content.strip()
    if not answer_en.endswith("."):
        last_period = answer_en.rfind(".")
        if last_period != -1:
            answer_en = answer_en[:last_period + 1]
        else:
            answer_en = answer_en.strip()

    # --- переводим ответ обратно ---
    return translate_from_english(answer_en, lang)


# ==========================
#  PDF SAVE HELPERS
# ==========================

styles = getSampleStyleSheet()

def _register_unicode_font():
    """
    Попытаться зарегистрировать DejaVuSans (путь типичный для Linux).
    Если не получилось — вернём None и оставим standard font.
    """
    possible_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/local/share/fonts/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for p in possible_paths:
        try:
            if os.path.exists(p):
                pdfmetrics.registerFont(TTFont('DejaVuSans', p))
                return 'DejaVuSans'
        except Exception:
            continue
    return None

# регистрируем (один раз)
_unicode_font = _register_unicode_font()
if _unicode_font:
    # применим к стилям
    styles['Normal'].fontName = _unicode_font
    styles['Heading1'].fontName = _unicode_font
    styles['Heading2'].fontName = _unicode_font

def save_tab1_to_pdf(results_df, chart_image_io, lang):
    """
    results_df: DataFrame (одна строка с колонками class_0..class_5)
    chart_image_io: BytesIO с PNG (как вернул display_results)
    lang: 'ru'/'en'/'kz'
    """

    # словари перевода
    type_dicts = {
        "ru": type_columns_ru,
        "en": type_columns_en,
        "kz": type_columns_kz
    }
    type_dict = type_dicts.get(lang, type_columns_en)

    # --- применяем adjust_probabilities, как в display_results ---
    results = {key: results_df[key].values[0] for key in results_df.columns}
    adjusted = adjust_probabilities(results, thresholds)

    # превращаем в DataFrame для таблицы
    renamed_df = pd.DataFrame([adjusted]).rename(columns=type_dict).round(2)

    # вычисляем наиболее подходящие типы (>= 100)
    suitable_types = [type_dict.get(col, col) for col, val in adjusted.items() if val >= 100]

    # заголовки
    titles = {
        "ru": "Результаты профориентационного теста",
        "en": "Career guidance results",
        "kz": "Кәсіби бағдар нәтижелері"
    }
    suitable_titles = {
        "ru": "Наиболее подходящие типы:",
        "en": "Most suitable types:",
        "kz": "Ең қолайлы түрлері:"
    }

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=20*mm, leftMargin=20*mm,
        topMargin=20*mm, bottomMargin=20*mm
    )
    elements = []

    elements.append(Paragraph(titles.get(lang, titles['en']), styles['Heading1']))
    elements.append(Spacer(1, 12))

    # --- Блок с подходящими типами ---
    if suitable_types:
        elements.append(Paragraph(suitable_titles[lang], styles['Heading2']))
        for t in suitable_types:
            elements.append(Paragraph(f"- {t}", styles['Normal']))
        elements.append(Spacer(1, 12))

    # --- Таблица ---
    # превращаем в таблицу: строки = типы, колонки = ["Тип", "Значение"]
    transposed_df = renamed_df.T.reset_index()
    transposed_df.columns = [
        {"ru": "Тип", "en": "Type", "kz": "Түрі"}[lang],
        {"ru": "Значение", "en": "Value", "kz": "Мәні"}[lang]
    ]
    # данные для ReportLab таблицы
    data = [list(transposed_df.columns)] + transposed_df.values.tolist()

    table = Table(data, hAlign="LEFT")
    table_style = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]
    if _unicode_font:
        table_style.append(('FONT', (0, 0), (-1, -1), _unicode_font))
    table.setStyle(table_style)
    elements.append(table)
    elements.append(Spacer(1, 12))

    # --- Диаграмма ---
    try:
        chart_image_io.seek(0)
        rl_img = RLImage(chart_image_io, width=160*mm, height=90*mm)
        elements.append(rl_img)
    except Exception as e:
        elements.append(Paragraph("Chart unavailable: " + str(e), styles['Normal']))

    doc.build(elements)
    buffer.seek(0)
    return buffer



def save_tab2_to_pdf(tab2_qas, ai_response, lang):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []

    titles = {
        "ru": "Открытые вопросы и анализ ИИ",
        "en": "Open questions and AI analysis",
        "kz": "Ашық сұрақтар мен ЖИ талдауы"
    }

    q_labels = {
        "ru": "Вопрос:",
        "en": "Question:",
        "kz": "Сұрақ:"
    }

    a_labels = {
        "ru": "Ответ:",
        "en": "Answer:",
        "kz": "Жауап:"
    }

    ai_labels = {
        "ru": "Анализ ИИ:",
        "en": "AI response:",
        "kz": "ЖИ талдауы:"
    }

    elements.append(Paragraph(titles.get(lang, titles['en']), styles['Heading1']))
    elements.append(Spacer(1, 12))

    for q, a in tab2_qas:
        elements.append(Paragraph(f"<b>{q_labels.get(lang, q_labels['en'])}</b> {q}", styles['Normal']))
        elements.append(Paragraph(f"<b>{a_labels.get(lang, a_labels['en'])}</b> {a}", styles['Normal']))
        elements.append(Spacer(1, 6))

    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"<b>{ai_labels.get(lang, ai_labels['en'])}</b>", styles['Normal']))
    elements.append(Paragraph(ai_response.replace("\n", "<br/>"), styles['Normal']))

    doc.build(elements)
    buffer.seek(0)
    return buffer


def save_tab3_to_pdf(question, rag_response, lang):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []

    titles = {
        "ru": "AI профориентация (RAG модель)",
        "en": "AI career guidance (RAG model)",
        "kz": "ЖИ кәсіби бағдар (RAG моделі)"
    }

    question_labels = {
        "ru": "Вопрос ученика:",
        "en": "Student's question:",
        "kz": "Оқушының сұрағы:"
    }

    response_labels = {
        "ru": "Ответ модели RAG:",
        "en": "AI RAG model response:",
        "kz": "RAG моделінің жауабы:"
    }

    elements.append(Paragraph(titles.get(lang, titles['en']), styles['Heading1']))
    elements.append(Spacer(1, 12))

    # Вопрос
    elements.append(Paragraph(f"<b>{question_labels.get(lang, question_labels['en'])}</b><br/>{question}", styles['Normal']))
    elements.append(Spacer(1, 12))
    # Ответ
    elements.append(Paragraph(f"<b>{response_labels.get(lang, response_labels['en'])}</b>", styles['Normal']))
    elements.append(Paragraph(rag_response.replace("\n", "<br/>"), styles['Normal']))

    doc.build(elements)
    buffer.seek(0)
    return buffer


# ==========================
#  INTERFACE
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
tabs = st.tabs([t["tab4"], t["tab1"], t["tab2"], t["tab3"]])

# ------------------------
# TAB 1 - Grades
with tabs[1]:
    st.subheader({
    "en": "Career type prediction from school grades",
    "ru": "Определение типа профессиональной направленности по школьным оценкам",
    "kz": "Мектептегі бағалардан кәсіби бағыттау түрін анықтау"
    }[lang])
    st.markdown({
    "en": "Enter your grades for grades 7-10 and select your motivational type.",
    "ru": "Укажите свои оценки за 7-10 классы и выберите свой мотивационный тип.",
    "kz": "7-10 сыныптардағы бағаларыңызды енгізіп, мотивациялық типіңізді таңдаңыз."
    }[lang])

    if "current_grade_label" not in st.session_state:
        st.session_state["current_grade_label"] = "7"

    current_grade_label = st.selectbox(
        t.get('current_grade_label', ld.get('current_grade_label', 'В каком вы сейчас классе?')),
        options=["7", "8", "9", "10-12"],
        index=["7", "8", "9", "10-12"].index(st.session_state["current_grade_label"]),
        key="current_grade_label"
    )

    with st.form("grades_form"):
        st.write(f"**{t['choose_type']}**")
        st.write(f"{t['go_to_tab']}")

        selected_checkboxes = {
            col: st.checkbox(current_column_names[col]) for col in checkbox_columns
        }

        selected_checkboxes = {
            col: st.checkbox(current_column_names[col]) for col in checkbox_columns
        }

        available_grades = get_available_grades(current_grade_label)

        for grade in available_grades:
            create_expander(
                grade,
                [c for c in inp_col_names if c.endswith(f"_{grade}")],
                ld,
                current_column_names,
                input_values
            )

        # Автозаполнение недоступных классов
        last_manual = available_grades[-1] if available_grades else 7
        for grade in range(last_manual + 1, 11):
            autofilled = autofill_grades(grade - 1, grade, input_values, current_column_names)
            input_values.update(autofilled)

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
        if lang == "ru":
            type_columns_dict = type_columns_ru
        elif lang == "kz":
            type_columns_dict = type_columns_kz
        else:
            type_columns_dict = type_columns_en

        chart_image_buf = display_results(st.session_state["tab1_results"], ld, type_columns_dict)

        if "tab1_results" in st.session_state and st.session_state["tab1_results"] is not None:
            html = st.session_state["tab1_results"].to_html()
            pdf_buffer = save_tab1_to_pdf(
                st.session_state["tab1_results"],
                chart_image_buf,
                lang
            )
            st.download_button(
                label={"ru": "Скачать PDF", "en": "Download PDF", "kz": "PDF жүктеу"}[lang],
                data=pdf_buffer,
                file_name="prof_type_results.pdf",
                mime="application/pdf"
            )
        else:
                st.warning({"ru": "Нет результатов для сохранения", "en": "No results to save", "kz": "Сақтауға нәтиже жоқ"}[lang])

        if lang == "ru":
            st.title("Типы профессиональной направленности")
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
            st.title("Types of professional orientation")
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
            st.title("Кәсіби бағдар беру түрлері")
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
        
# ------------------------
# TAB 2 - Open questions
with tabs[2]:
    st.subheader({
    "en": "Personalized career advice based on your answers",
    "ru": "Персональные советы по выбору профессии на основе ваших ответов",
    "kz": "Жауаптарыңызға негізделген жеке кәсіби кеңес"
    }[lang])
    st.markdown({
    "en": "Answer the following questions about your interests, strengths, and dislikes. Based on your answers, the AI will provide personalized career advice.", 
    "ru": "Ответьте на следующие вопросы о ваших интересах, сильных сторонах и нелюбимых занятиях. На основе ваших ответов ИИ предоставит персональные советы по выбору профессии.",
    "kz": "Қызығушылықтарыңыз, күшті жақтарыңыз және ұнатпайтын істеріңіз туралы келесі сұрақтарға жауап беріңіз. Жауаптарыңызға негізделген ЖИ сізге жеке кәсіби кеңес береді."
    }[lang])

    with st.form("open_questions_form"):
        user_answers = [st.text_input(q, key=f"answer_{i}") for i, q in enumerate(ld["questions"])]
        submit_tab2 = st.form_submit_button(t["get_answer"])

    if submit_tab2:
        ai_response = get_ai_response(user_answers, lang=st.session_state["lang"])
        st.session_state["tab2_ai_response"] = ai_response
        st.session_state["tab2_qas"] = list(zip(ld["questions"], user_answers))  # сохраняем Q&A

    if "tab2_ai_response" in st.session_state:
        st.subheader(t["ai_response"])
        st.write(st.session_state["tab2_ai_response"])

        pdf_buffer = save_tab2_to_pdf(
                st.session_state["tab2_qas"], 
                st.session_state["tab2_ai_response"], 
                lang
            )
        st.download_button(
            label={"ru": "Сохранить в PDF", "en": "Save as PDF", "kz": "PDF сақтау"}[lang],
            data=pdf_buffer,
            file_name="open_questions_analysis.pdf",
            mime="application/pdf"
        )

# ------------------------
# TAB 3 - AI career (RAG only)
with tabs[3]:
    st.subheader(t["advisor"])
    # мини-описание RAG
    st.markdown({
    "en": "Here you can ask follow-up questions based on the career advice you’ve already received. "
          "The system uses RAG (Retrieval-Augmented Generation), which means it retrieves information "
          "from a prepared knowledge base and then generates answers. This helps provide more focused, "
          "reliable, and personalized guidance — not random information from the internet.",
    "ru": "Здесь вы можете задать уточняющие вопросы на основе уже полученных советов. "
          "Система использует RAG (Retrieval-Augmented Generation — генерация с дополнением поиска), "
          "то есть извлекает информацию из подготовленной базы знаний и формирует ответ. "
          "Это помогает давать более точные, надёжные и персонализированные рекомендации — "
          "а не случайную информацию из интернета.",
    "kz": "Мұнда сіз бұрын алған кеңестерге негізделген қосымша сұрақтар қоя аласыз. "
          "Жүйе RAG (Retrieval-Augmented Generation — іздеумен толықтырылған генерация) тәсілін қолданады, "
          "яғни дайын білім қорынан ақпарат алып, жауап құрастырады. "
          "Бұл интернеттегі кездейсоқ ақпарат емес, нақтыланған әрі жекелендірілген нұсқауларды ұсынады."
        }[lang])
    with st.form("career_form"):
        # --- Expander с рекомендациями ---
        with st.expander({"ru": "Рекомендации по формулировке вопроса", 
                          "en": "Recommendations for formulating your question",
                          "kz": "Сұрақты құрастыру бойынша ұсыныстар"}[lang]):
            if lang == "ru":
                st.markdown("""
                **Пример 1: Определение профильных предметов**  
                Я хочу определиться с выбором двух профильных предметов из четырёх (физика, химия, биология, информатика).  
                - Мне нравится делать: …  
                - Мне не нравится делать: …  
                - У меня хорошо получается: …  
                - Мне сложно дается: …  
                - Тип MBTI (не обязательно): …  
                - Мотивационный тип (по Битяновой): …  
                - Желаемая специальность: …  

                **Пример 2: Определение специальности**  
                Я хочу определиться с выбором специальности.  
                - Я собираюсь выбрать (или уже выбрал) такие 2 профильных предмета из 4-х (физика, химия, биология, информатика): …  
                - Мне нравится делать: …  
                - Мне не нравится делать: …  
                - У меня хорошо получается: …  
                - Мне сложно дается: …  
                - Тип MBTI (не обязательно): …  
                - Мотивационный тип (по Битяновой): …  
                
                **Примечание:**
                RAG отвечает на основе нашей проверенной внутренней базы знаний — это даёт персонализированные и согласованные рекомендации и лучше защищает вашу приватность.
                Для самых свежих фактов (даты приёма, конкретные цены и т. п.) используйте официальные сайты.
                """)
            elif lang == "en":
                st.markdown("""
                **Example 1: Choosing school subjects**  
                I want to decide on two profile subjects out of four (physics, chemistry, biology, computer science).  
                - I enjoy doing: …  
                - I don’t like doing: …  
                - I am good at: …  
                - I find it difficult to: …  
                - MBTI type (optional): …  
                - Motivational type (Bitianova): …  
                - Desired major: …  

                **Example 2: Choosing a major**  
                I want to decide on my major.  
                - I am going to choose (or have already chosen) two profile subjects out of four (physics, chemistry, biology, computer science): …  
                - I enjoy doing: …  
                - I don’t like doing: …  
                - I am good at: …  
                - I find it difficult to: …  
                - MBTI type (optional): …  
                - Motivational type (Bitianova): …  
                            
                **Note:**
                The RAG model responds based on our vetted internal knowledge base — this provides personalized and consistent recommendations and better protects your privacy.
                For the most up-to-date facts (admission dates, specific prices, etc.) please refer to official websites.
                """)
            else:  # kz
                st.markdown("""
                **1-мысал: Профильдік пәндерді таңдау**  
                Мен төрт пәннің ішінен (физика, химия, биология, информатика) екі профильдік пәнді таңдағым келеді.  
                - Маған ұнайтын істер: …  
                - Маған ұнамайтын істер: …  
                - Маған жақсы берілетіндер: …  
                - Маған қиын берілетіндер: …  
                - MBTI типі (міндетті емес): …  
                - Мотивациялық тип (Битянова бойынша): …  
                - Таңдағысы келетін мамандық: …  

                **2-мысал: Мамандықты таңдау**  
                Мен мамандық таңдағым келеді.  
                - Мен төрт пәннің ішінен (физика, химия, биология, информатика) осындай 2 профильдік пәнді таңдадым (немесе таңдағалы жатырмын): …  
                - Маған ұнайтын істер: …  
                - Маған ұнамайтын істер: …  
                - Маған жақсы берілетіндер: …  
                - Маған қиын берілетіндер: …  
                - MBTI типі (міндетті емес): …  
                - Мотивациялық тип (Битянова бойынша): …  
                            
                **Ескерту:**
                RAG біздің тексерілген ішкі білім базасына негізделген жауап береді — бұл жеке және үйлесімді ұсыныстар береді және сіздің құпиялылығыңызды жақсы қорғайды.
                Ең соңғы деректер (қабылдау күндері, нақты бағалар және т.б.) үшін ресми сайттарға жүгініңіз.
                """)

        # --- Основное поле ввода ---
        student_question = st.text_area(t["student_question"], height=100, key="student_q")

        # --- Кнопка ---
        submit_tab3 = st.form_submit_button(t["get_advice"])

    if submit_tab3:
        rag_answer = generate_rag_career_advice(student_question, embedder, annoy_index, texts, lang=st.session_state["lang"])
        st.session_state["tab3_rag"] = rag_answer

    if "tab3_rag" in st.session_state:
        st.subheader(t["rag_model"])
        st.write(st.session_state["tab3_rag"])
        pdf_buffer = save_tab3_to_pdf(
                st.session_state["student_q"], 
                st.session_state["tab3_rag"], 
                lang
            )
        st.download_button(
            label={"ru": "Сохранить в PDF", "en": "Save as PDF", "kz": "PDF сақтау"}[lang],
            data=pdf_buffer,
            file_name="ai_response.pdf",
            mime="application/pdf"
        )


# Типы личности и переводы
type_labels = {
    "A": {"ru": "Активист", "en": "Activist", "kz": "Белсенді"},
    "K": {"ru": "Карьерист", "en": "Careerist", "kz": "Мансапқор"},
    "Ip": {"ru": "Испытатель", "en": "Tester", "kz": "Тексеруші"},
    "T": {"ru": "Творец", "en": "Creator", "kz": "Жасаушы"},
    "P": {"ru": "Проектировщик", "en": "Designer", "kz": "Жоспарлаушы"},
    "Il": {"ru": "Исследователь", "en": "Researcher", "kz": "Зерттеуші"},
}


questions = [
    {
        "ru": "Тебе нравится получать похвалу за хорошо сделанную работу?",
        "en": "Do you like receiving praise for a job well done?",
        "kz": "Жақсы орындалған жұмыс үшін мақтау алғанды ұнатасың ба?",
        "codes": ["A"]
    },
    {
        "ru": "Ты переживаешь, когда учитель ставит тебе в дневник плохую отметку?",
        "en": "Do you worry when a teacher gives you a bad grade?",
        "kz": "Мұғалім жаман баға қойған кезде уайымдайсың ба?",
        "codes": ["A"]
    },
    {
        "ru": "Ты переживаешь, когда обнаруживаешь ошибки в работе?",
        "en": "Do you worry when you find mistakes in your work?",
        "kz": "Жұмысыңнан қателер тапқанда уайымдайсың ба?",
        "codes": ["A"]
    },
    {
        "ru": "Часто бывает, что трудности в процессе работы выбивают тебя из колеи?",
        "en": "Do difficulties during work often throw you off track?",
        "kz": "Жұмыс барысында қиындықтар жиі сені жолдан тайдыра ма?",
        "codes": ["A", "T"]
    },
    {
        "ru": "Часто случается, что из-за неудачи ты оставляешь работу незаконченной?",
        "en": "Do you often leave work unfinished because of failure?",
        "kz": "Сәтсіздікке байланысты жұмысты жиі аяқтамай қалдырасың ба?",
        "codes": ["A", "P"]
    },
    {
        "ru": "Тебе нравится выполнять разные сложные поручения, важные для школы или класса?",
        "en": "Do you like completing difficult tasks important for your school or class?",
        "kz": "Мектеп немесе сынып үшін маңызды күрделі тапсырмаларды орындағанды ұнатасың ба?",
        "codes": ["A"]
    },
    {
        "ru": "Ты стремишься быть в компании «на первых ролях»?",
        "en": "Do you try to take leading roles in a group?",
        "kz": "Ұжымда жетекші рөл атқаруға тырысасың ба?",
        "codes": ["A"]
    },
    {
        "ru": "Можно ли сказать, что тебе легко общаться с людьми?",
        "en": "Would you say it’s easy for you to communicate with people?",
        "kz": "Адамдармен сөйлесу саған оңай деп айта аласың ба?",
        "codes": ["A", "Il"]
    },
    {
        "ru": "Тебе нравится быть лучше, чем другие, в какой-либо учебной ситуации?",
        "en": "Do you like being better than others in a learning situation?",
        "kz": "Оқу жағдайында өзгелерден жақсы болуды ұнатасың ба?",
        "codes": ["K"]
    },
    {
        "ru": "Как ты считаешь, при желании ты всегда сможешь получить хорошую оценку?",
        "en": "Do you think you can always get a good grade if you try?",
        "kz": "Қаласаң әрқашан жақсы баға ала аламын деп ойлайсың ба?",
        "codes": ["K"]
    },
    {
        "ru": "Ты чувствуешь обиду или огорчение, если тебя постигла неудача?",
        "en": "Do you feel upset when you fail?",
        "kz": "Сәтсіздікке ұшырағанда ренжисің бе?",
        "codes": ["K"]
    },
    {
        "ru": "Можно ли сказать, что трудности в работе тебя, как правило, не пугают?",
        "en": "Can it be said that difficulties at work usually don’t scare you?",
        "kz": "Жұмыстағы қиындықтар сені әдетте қорқытпайды деп айтуға бола ма?",
        "codes": ["K"]
    },
    {
        "ru": "Ты способен в случае неудачи быстро настроиться и продолжать работу?",
        "en": "Are you able to quickly adjust and keep working after failure?",
        "kz": "Сәтсіздіктен кейін тез бейімделіп, жұмысты жалғастыра аласың ба?",
        "codes": ["K", "Il"]
    },
    {
        "ru": "Как ты считаешь, можно ли поручить тебе ответственное задание?",
        "en": "Do you think you can be trusted with a responsible task?",
        "kz": "Саған жауапты тапсырманы сеніп тапсыруға болады деп ойлайсың ба?",
        "codes": ["K"]
    },
    {
        "ru": "Тебе легко внести оживление в большую компанию?",
        "en": "Is it easy for you to liven up a large group?",
        "kz": "Үлкен топты жандандыру саған оңай ма?",
        "codes": ["K"]
    },
    {
        "ru": "Можно ли про тебя сказать, что тебе больше нравится выполнять работу одному, чем вместе с другими?",
        "en": "Would you say you prefer working alone rather than with others?",
        "kz": "Басқалармен бірге істегеннен гөрі жалғыз жұмыс істеуді жақсы көресің бе?",
        "codes": ["K"]
    },
    {
        "ru": "Тебе важно, чтобы другие люди признавали твои успехи и способности?",
        "en": "Is it important for you that others recognize your success and abilities?",
        "kz": "Өз жетістіктерің мен қабілеттеріңді өзгелер мойындағаны сен үшін маңызды ма?",
        "codes": ["Ip"]
    },
    {
        "ru": "Можно ли про тебя сказать, что ты стараешься как можно быстрее исправить то, что у тебя не получилось?",
        "en": "Do you try to quickly fix things when they don’t work out?",
        "kz": "Саған бірдеңе шықпай қалса, оны тезірек түзетуге тырысасың ба?",
        "codes": ["Ip"]
    },
    {
        "ru": "В случае неудачи тебе важно как можно быстрее разобраться в ее причинах?",
        "en": "When you fail, is it important for you to quickly understand the reasons?",
        "kz": "Сәтсіздік болғанда, оның себебін тезірек түсіну сен үшін маңызды ма?",
        "codes": ["Ip"]
    },
    {
        "ru": "Часто ли, потерпев неудачу в деле, ты чувствуешь себя несправедливо обиженным?",
        "en": "Do you often feel unfairly treated after a failure?",
        "kz": "Сәтсіздіктен кейін жиі әділетсіз ренжіп қаласың ба?",
        "codes": ["Ip", "T"]
    },
    {
        "ru": "Тебе нравится браться за необычные, трудные задания?",
        "en": "Do you like taking on unusual, difficult tasks?",
        "kz": "Әдеттен тыс, қиын тапсырмаларды орындағанды ұнатасың ба?",
        "codes": ["Ip"]
    },
    {
        "ru": "Согласен ли ты с тем, что не ко всем порученным делам нужно относиться одинаково ответственно?",
        "en": "Do you agree that not all tasks need the same level of responsibility?",
        "kz": "Барлық тапсырмаларға бірдей жауапкершілікпен қарау қажет емес деп келісесің бе?",
        "codes": ["Ip"]
    },
    {
        "ru": "Для тебя важно, чтобы в споре победила твоя позиция?",
        "en": "Is it important for you that your position wins in a dispute?",
        "kz": "Дау кезінде сенің ұстанымың жеңгені сен үшін маңызды ма?",
        "codes": ["Ip"]
    },
    {
        "ru": "Тебе нравится соревноваться со своими сверстниками, участвовать в конкурсах?",
        "en": "Do you like competing with peers and joining contests?",
        "kz": "Өз құрдастарыңмен жарысқанды, байқауларға қатысқанды ұнатасың ба?",
        "codes": ["Ip"]
    },
    {
        "ru": "Ты гордишься, когда твои успехи признают окружающие люди?",
        "en": "Do you feel proud when others recognize your achievements?",
        "kz": "Өз жетістіктеріңді басқалар мойындағанда мақтанасың ба?",
        "codes": ["T"]
    },
    {
        "ru": "Можно ли про тебя сказать, что ты стремишься получать только хорошие отметки?",
        "en": "Would you say you always aim for only good grades?",
        "kz": "Әрқашан тек жақсы баға алуға тырысасың ба?",
        "codes": ["T"]
    },
    {
        "ru": "Можно про тебя сказать, что ты не любишь работы, которая выполняется по алгоритму или образцу?",
        "en": "Would you say you dislike work done by a set pattern or algorithm?",
        "kz": "Алгоритм немесе дайын үлгі бойынша жасалатын жұмысты ұнатпайсың ба?",
        "codes": ["T"]
    },
    {
        "ru": "Тебе нравятся необычные задания, где можно проявить воображение?",
        "en": "Do you like unusual tasks where you can show imagination?",
        "kz": "Қиялыңды көрсетуге болатын әдеттен тыс тапсырмаларды ұнатасың ба?",
        "codes": ["T"]
    },
    {
        "ru": "Тебе нравится работать самостоятельно, самому определяя, что и как делать?",
        "en": "Do you like working independently, deciding what and how to do?",
        "kz": "Не істеу және қалай істеуді өзің шешіп, тәуелсіз жұмыс істеуді ұнатасың ба?",
        "codes": ["T"]
    },
    {
        "ru": "Твое желание общаться с людьми часто зависит от настроения?",
        "en": "Does your desire to talk to people often depend on your mood?",
        "kz": "Адамдармен сөйлескің келуі жиі көңіл-күйіңе байланысты ма?",
        "codes": ["T"]
    },
    {
        "ru": "Можно ли сказать, что успехи и достижения – это нормально, естественно для тебя?",
        "en": "Can it be said that success and achievements feel natural to you?",
        "kz": "Жетістіктер мен табыстар сен үшін табиғи нәрсе деп айтуға бола ма?",
        "codes": ["P"]
    },
    {
        "ru": "Для тебя не имеет большого значения, как другие оценят твою работу?",
        "en": "Does it matter little to you how others evaluate your work?",
        "kz": "Басқалардың жұмысыңды қалай бағалайтыны сен үшін маңызды емес пе?",
        "codes": ["P"]
    },
    {
        "ru": "Могут ли трудности в работе заставить тебя переживать?",
        "en": "Can difficulties at work make you worry?",
        "kz": "Жұмыстағы қиындықтар сені уайымдатуы мүмкін бе?",
        "codes": ["P"]
    },
    {
        "ru": "Ты не любишь задания, в которых нет четкой логики или структуры?",
        "en": "Do you dislike tasks without clear logic or structure?",
        "kz": "Айқын логикасы немесе құрылымы жоқ тапсырмаларды ұнатпайсың ба?",
        "codes": ["P"]
    },
    {
        "ru": "Можно ли сказать, что не только ты отвечаешь за результат своей учебы?",
        "en": "Would you say you are not the only one responsible for your study results?",
        "kz": "Оқу нәтижелерің үшін тек өзің ғана емес, басқалар да жауапты деп айтуға бола ма?",
        "codes": ["P"]
    },
    {
        "ru": "Тебе не очень важно быть на виду, проявлять инициативу, влиять на мнение других?",
        "en": "Is it not very important for you to be visible and influence others?",
        "kz": "Көзге түсіп, бастама көтеріп, өзгелерге әсер ету сен үшін аса маңызды емес пе?",
        "codes": ["P", "Il"]
    },
    {
        "ru": "Тебе важно, чтобы были люди, которые разделяют твои интересы?",
        "en": "Is it important for you to have people who share your interests?",
        "kz": "Сенің қызығушылықтарыңды бөлісетін адамдардың болуы маңызды ма?",
        "codes": ["P"]
    },
    {
        "ru": "Важно ли тебе знать, почему ты получил ту или иную отметку?",
        "en": "Is it important for you to know why you got a certain grade?",
        "kz": "Неліктен белгілі бір баға алғаныңды білу сен үшін маңызды ма?",
        "codes": ["Il"]
    },
    {
        "ru": "Можно ли про тебя сказать, что тебе важно докопаться до сути?",
        "en": "Would you say it’s important for you to get to the essence of things?",
        "kz": "Істің мәніне жету сен үшін маңызды деп айтуға бола ма?",
        "codes": ["Il"]
    },
    {
        "ru": "Ты не отказываешься от работы только из-за трудностей?",
        "en": "Do you avoid giving up work just because it’s difficult?",
        "kz": "Тек қиындық болғаны үшін жұмыстан бас тартпайсың ба?",
        "codes": ["Il"]
    },
    {
        "ru": "Ты стараешься ответственно и точно выполнять все задания?",
        "en": "Do you try to complete all tasks responsibly and accurately?",
        "kz": "Барлық тапсырмаларды жауапкершілікпен әрі дәл орындауға тырысасың ба?",
        "codes": ["Il"]
    },
    {
        "ru": "В конфликте ты стараешься примирить спорящих?",
        "en": "In a conflict, do you try to reconcile the arguing sides?",
        "kz": "Қақтығыста дауласып жатқан адамдарды татуластыруға тырысасың ба?",
        "codes": ["Il"]
    }
]


with tabs[0]:
    st.subheader({
    "ru": "Тест на мотивационный тип личности",
    "en": "Motivational Personality Type Test",
    "kz": "Мотивациялық тұлға типі тесті"
    }[lang])
    st.markdown({
    "ru": "Отметь галочкой утверждения, которые соответствуют твоему поведению. В конце ты увидишь, какой у тебя мотивационный тип личности по Битяновой.",
    "en": "Check the statements that correspond to your behavior. At the end, you will see your motivational personality type according to Bitianova.",
    "kz": "Өзіңнің мінез-құлқыңа сәйкес келетін мәлімдемелерді белгілеңіз. Соңында сіз Битяноваға сәйкес мотивациялық тұлға типіңізді көресіз."
    }[lang])
    answers = []
    with st.form("quiz_form"):
        for i, q in enumerate(questions, 1):
            ans = st.checkbox(f"{i}. {q[lang]}", key=f"q{i}")
            answers.append(ans)
        submitted = st.form_submit_button({
            "ru": "Показать результаты",
            "en": "Show results",
            "kz": "Нәтижелерді көрсету"
        }[lang])
    if submitted:
        scores = {code: 0 for code in type_labels.keys()}
        for ans, q in zip(answers, questions):
            if ans:
                for code in q["codes"]:
                    scores[code] += 1

        # Определяем итоговые типы
        max_score = max(scores.values())
        selected = [code for code, val in scores.items() if val >= 7] or [
            code for code, val in scores.items() if val == max_score
        ]

        st.subheader({
        "ru": "Ваши ведущие типы личности:",
        "en": "Your leading personality types:",
        "kz": "Сіздің жетекші тұлға типтеріңіз:"
        }[lang])
        for code in selected:
            st.write("✅", type_labels[code][lang])

        # Диаграмма
        fig, ax = plt.subplots(figsize=(8, 5))  # увеличим размер графика
        bars = ax.bar(
        [type_labels[c][lang] for c in scores.keys()],
        scores.values(),
        width=0.5  # уменьшим ширину колонок, чтобы между ними было больше места
        )

        ax.set_ylabel({
            "ru": "Баллы",
            "en": "Scores",
            "kz": "Ұпайлар"
        }[lang])

        # Увеличим отступы между метками оси X
        ax.set_xticks(range(len(scores)))
        ax.set_xticklabels(
            [type_labels[c][lang] for c in scores.keys()],
            rotation=20,   # чуть повернём, чтобы не налезали
            ha="right"
        )

        st.pyplot(fig)

        st.write({
        "ru": "Теперь вы знаете свой мотивационный тип личности по Битяновой. Вернитесь на вкладку «Школьные оценки», чтобы увидеть, какие профессии вам подходят.",
        "en": "Now you know your motivational personality type according to Bitianova. Go back to the 'School grades' tab to see which professions suit you.",
        "kz": "Енді сіз Битяноваға сәйкес мотивациялық тұлға типіңізді білесіз. Сізге сәйкес келетін мамандықтарды көру үшін «Мектеп бағалары» қойындысына оралыңыз."
        }[lang])




        





