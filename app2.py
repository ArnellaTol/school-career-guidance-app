import streamlit as st
import pandas as pd
import numpy as np
import joblib

# import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="Профориентация для школьников",
)


out_col_names=[
 'signsystem',
 'technology',
 'nature',
 'artistic',
 'human',
 'business']

inp_col_names=[
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

column_names_dict = {
    'kaz_lang_7': 'Kazakh Language',
    'liter_7': 'Literature',
    'rus_lang_7': 'Russian Language',
    'eng_lang_7': 'English Language',
    'math_7': 'Mathematics',
    'comps_7': 'Informatics',
    'kaz_hist_7': 'History of Kazakhstan',
    'art_7': 'Art',
    'pe_7': 'Physical Education',
    'geography_7': 'Geography',
    'biology_7': 'Biology',
    'chemistry_7': 'Chemistry',
    'physics_7': 'Physics',
    'world_hist_7': 'World History',
    'Activist': 'Activist',
    'Career': 'Careerist',
    'Tester': 'Tester',
    'Creator': 'Creator',
    'Designer': 'Designer',
    'Researcher': 'Researcher',
    'kaz_lang_8': 'Kazakh Language',
    'liter_8': 'Literature',
    'rus_lang_8': 'Russian Language',
    'eng_lang_8': 'English Language',
    'math_8': 'Mathematics',
    'comps_8': 'Informatics',
    'kaz_hist_8': 'History of Kazakhstan',
    'art_8': 'Art',
    'pe_8': 'Physical Education',
    'geography_8': 'Geography',
    'biology_8': 'Biology',
    'chemistry_8': 'Chemistry',
    'physics_8': 'Physics',
    'world_hist_8': 'World History',
    'kaz_lang_9': 'Kazakh Language',
    'liter_9': 'Literature',
    'rus_lang_9': 'Russian Language',
    'eng_lang_9': 'English Language',
    'math_9': 'Mathematics',
    'comps_9': 'Informatics',
    'kaz_hist_9': 'History of Kazakhstan',
    'art_9': 'Art',
    'pe_9': 'Physical Education',
    'geography_9': 'Geography',
    'biology_9': 'Biology',
    'chemistry_9': 'Chemistry',
    'physics_9': 'Physics',
    'world_hist_9': 'World History',
    'rights_9': 'Law Fundamentals',
    'kaz_lang_10': 'Kazakh Language',
    'liter_10': 'Literature',
    'rus_lang_10': 'Russian Language',
    'eng_lang_10': 'English Language',
    'math_10': 'Mathematics',
    'comps_10': 'Informatics',
    'kaz_hist_10': 'History of Kazakhstan',
    'art_10': 'Art',
    'pe_10': 'Physical Education',
    'geography_10': 'Geography',
    'biology_10': 'Biology',
    'chemistry_10': 'Chemistry',
    'physics_10': 'Physics',
    'world_hist_10': 'World History'
}


column_names_dict_ru = {
    'kaz_lang_7': 'Казахский язык',
    'liter_7': 'Литература',
    'rus_lang_7': 'Русский язык',
    'eng_lang_7': 'Английский язык',
    'math_7': 'Математика',
    'comps_7': 'Информатика',
    'kaz_hist_7': 'История Казахстана',
    'art_7': 'Искусство',
    'pe_7': 'Физкультура',
    'geography_7': 'География',
    'biology_7': 'Биология',
    'chemistry_7': 'Химия',
    'physics_7': 'Физика',
    'world_hist_7': 'Всемирная история',
    'Activist': 'Активист',
    'Career': 'Карьерист',
    'Tester': 'Испытатель',
    'Creator': 'Творец',
    'Designer': 'Проектировщик',
    'Researcher': 'Исследователь',
    'kaz_lang_8': 'Казахский язык',
    'liter_8': 'Литература',
    'rus_lang_8': 'Русский язык',
    'eng_lang_8': 'Английский язык',
    'math_8': 'Математика',
    'comps_8': 'Информатика',
    'kaz_hist_8': 'История Казахстана',
    'art_8': 'Искусство',
    'pe_8': 'Физкультура',
    'geography_8': 'География',
    'biology_8': 'Биология',
    'chemistry_8': 'Химия',
    'physics_8': 'Физика',
    'world_hist_8': 'Всемирная история',
    'kaz_lang_9': 'Казахский язык',
    'liter_9': 'Литература',
    'rus_lang_9': 'Русский язык',
    'eng_lang_9': 'Английский язык',
    'math_9': 'Математика',
    'comps_9': 'Информатика',
    'kaz_hist_9': 'История Казахстана',
    'art_9': 'Искусство',
    'pe_9': 'Физкультура',
    'geography_9': 'География',
    'biology_9': 'Биология',
    'chemistry_9': 'Химия',
    'physics_9': 'Физика',
    'world_hist_9': 'Всемирная история',
    'rights_9': 'Основы права',
    'kaz_lang_10': 'Казахский язык',
    'liter_10': 'Литература',
    'rus_lang_10': 'Русский язык',
    'eng_lang_10': 'Английский язык',
    'math_10': 'Математика',
    'comps_10': 'Информатика',
    'kaz_hist_10': 'История Казахстана',
    'art_10': 'Искусство',
    'pe_10': 'Физкультура',
    'geography_10': 'География',
    'biology_10': 'Биология',
    'chemistry_10': 'Химия',
    'physics_10': 'Физика',
    'world_hist_10': 'Всемирная история'
}

checkbox_columns = [
    'Activist', 'Career', 'Tester', 'Creator', 'Designer', 'Researcher'
]

def create_expander(class_label, cols):
    # with st.expander(f"Grades for the {class_label}th grade"):
    with st.expander(f"Оценки за {class_label} класс"):
        for col in cols:
            # column_names_dict_ru - column_names_dict
            input_values[col] = st.number_input(column_names_dict_ru[col], min_value=2, max_value=5, step=1, value=5, key=col)

input_values = {}


def save_to_dataframe(selected_checkboxes, input_values):
    data = {**selected_checkboxes, **input_values}
    
    for key in ['Activist', 'Career', 'Tester', 'Creator', 'Designer', 'Researcher']:
        data[key] = int(data.get(key, False))
    
    column_order = [
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
    
    df = pd.DataFrame([data], columns=column_order)
    
    return df




import logging

logging.basicConfig(level=logging.DEBUG)

def apply_model(model_path, input_df):
    logging.debug("Загрузка модели")
    model = joblib.load(model_path)
    
    logging.debug("Применение модели к данным")
    probabilities = model.predict_proba(input_df)
    
    logging.debug(f"Тип probabilities: {type(probabilities)}")
    logging.debug(f"Содержимое probabilities: {probabilities}")
    
    if isinstance(probabilities, list):
        probabilities = np.array(probabilities)
    
    logging.debug(f"Тип после преобразования: {type(probabilities)}")
    
    probability_dict = {f'class_{i}': probabilities[i][:, 1] for i in range(len(probabilities))}
    
    logging.debug("Создание DataFrame с результатами")
    result_df = pd.DataFrame(probability_dict)
    
    return result_df





type_columns = {
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

thresholds = {
    'class_0': 0.39,
    'class_1': 0.30903005409623036,
    'class_2': 0.23611111111111113,
    'class_3': 0.44833333333333336,
    'class_4': 0.13,
    'class_5': 0.17
}



def adjust_probabilities(probabilities, thresholds):
    adjusted_probs = {}
    for key, value in probabilities.items():
        adjusted_probs[key] = min(100, (value / thresholds[key]) * 100)
    return adjusted_probs




def display_results(df):
    results = {key: df[key].values[0] for key in df.columns}
    adjusted_results = adjust_probabilities(results, thresholds)

    # type_columns_ru - type_columns
    selected_types = [
        type_columns_ru[key] 
        for key, value in adjusted_results.items() if value >= 100
    ]

    st.write("**Наиболее подходящие типы профессиональной склонности, выведенные Дж. Холландом:**")
    for typ in selected_types:
        st.write(f"- {typ}")

    st.write("**Вероятности всех типов профессиональной склонности:**")

    # Превратим словарь в DataFrame для отображения
    chart_data = pd.DataFrame({
        "Тип": [type_columns_ru[key] for key in adjusted_results.keys()],
        "Вероятность": list(adjusted_results.values())
    })

    # Показываем таблицу
    st.dataframe(chart_data, use_container_width=True)

    # Дополнительно можно показать bar chart
    st.bar_chart(chart_data.set_index("Тип"))

    



# st.header("Career guidance for students")
st.header("Профориентация для школьников")
# tabs = st.tabs(["School academic records", "Open questions"]) # , "Test ...(under development)"
tabs = st.tabs(["Школьные оценки", "Открытые вопросы", "AI профориентатор"]) # , "Тест ...(в разработке)"

with tabs[0]:
    st.write("**Выберите свой мотивационный тип (школьная анкета 7 класс):**")
    # st.write("**Choose your motivation type (7th grade school questionnaire):**")
    selected_checkboxes = {}
    for col in checkbox_columns:
        selected_checkboxes[col] = st.checkbox(column_names_dict_ru[col]) # column_names_dict
    
    create_expander(7, [col for col in inp_col_names if col.endswith('_7')])
    create_expander(8, [col for col in inp_col_names if col.endswith('_8')])
    create_expander(9, [col for col in inp_col_names if col.endswith('_9')])
    create_expander(10, [col for col in inp_col_names if col.endswith('_10')])

    # if st.button("Get a result"):
    if st.button("Получить результат"):
        df = save_to_dataframe(selected_checkboxes, input_values)
        result_df = apply_model('random_forest_model.pkl', df)
        display_results(result_df)



import requests
from dotenv import load_dotenv

load_dotenv()

my_api = st.secrets["OPENAI_API_KEY"]

def get_ai_response(answers):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {my_api}",
        "Content-Type": "application/json"
    }

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
    response.raise_for_status()  # выдаст ошибку если запрос не удался

    return response.json()["choices"][0]["message"]["content"]


questions_ru = [
    "**1. Какие профессии вас интересуют на данный момент?**",
    "**2. Какие виды деятельности вам точно не интересны?**",
    "**3. Без учета финансовых аспектов, какие виды деятельности или профессии вам нравятся?**",
    "**4. Перечислите свои хобби и интересы:**",
    "**5. Назовите ролевые модели, чьи образы жизни и достижения вас вдохновляют.**",
    "**6. Какие задачи придают вам энергии?**",
    "**7. Какие задачи вас утомляют?**"
]

user_answers = []

with tabs[1]:
    for i, question in enumerate(questions_ru):
        answer = st.text_input(f"{question}", key=f"answer_{i}")
        user_answers.append(answer)

    if st.button("Получить ответ"):
        ai_response = get_ai_response(user_answers)
        st.write("Ответ ИИ:")
        st.write(ai_response)


import os
import json
import streamlit as st
from huggingface_hub import InferenceClient, login
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex


@st.cache_data(show_spinner="Загрузка JSONL файлов...")
def load_jsonl_files(folder_path):
    all_records = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        all_records.append(record)
                    except json.JSONDecodeError:
                        print(f"Ошибка при чтении файла: {filename}")
    return all_records


rag_data = load_jsonl_files("./jsonl datafiles")


def login_hf():
    if not os.environ.get("HF_TOKEN"):
        login(token=st.secrets["HF_TOKEN"])


login_hf()


@st.cache_resource
def load_annoy_index():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [item["text"] for item in rag_data]
    dimension = 384

    annoy_index = AnnoyIndex(dimension, 'angular')
    annoy_index.load("index.ann")

    return embedder, annoy_index, texts


embedder, annoy_index, texts = load_annoy_index()


def generate_career_advice(question: str) -> str:
    messages = [
        {"role": "system", "content":
         """Вы — карьерный консультант для старшеклассников. 
         Если ученик просит предложить карьерные пути, выберите 3 направления, которые лучше всего подходят под его интересы, сильные стороны и предпочтения.
         Ответ держите в пределах 100 слов. Будьте сфокусированы и по делу."""},
        {"role": "user", "content": question}
    ]

    client = InferenceClient(
        provider="auto",
        api_key=st.secrets["HF_TOKEN"]
    )

    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=messages,
        max_tokens=350,
        temperature=0.7
    )

    return response.choices[0].message.content


def generate_rag_career_advice(question: str, embedder, annoy_index, texts: list, k: int = 5) -> str:
    query_embedding = embedder.encode([question], convert_to_numpy=True)

    indices = annoy_index.get_nns_by_vector(query_embedding[0], k, include_distances=False)
    context_docs = [texts[i] for i in indices]

    context = "\n\n".join(context_docs)

    messages = [
        {"role": "system", "content":
         f"""
Вы — карьерный консультант для старшеклассников.

У вас есть доступ к дополнительным материалам (контекст ниже).

Контекст:
{context}

Ваша задача — выбрать 3 карьерных пути, которые лучше всего соответствуют интересам, сильным сторонам и предпочтениям ученика.  
Требования:
- Опираться только на сообщение ученика, не придумывать лишнего.  
- Давать только те варианты, которые явно подходят, и избегать неподходящих.  
- Для каждого варианта объяснить в 3–4 предложениях, почему он подходит именно этому ученику.  
- Не давать общих советов или длинных списков «на всякий случай».  
- Ответ не более 350 слов.  

Если ученик задаёт другие вопросы — отвечайте прямо, используя контекст, но не предлагайте карьерные пути.
"""},

        {"role": "user", "content": question}
    ]

    client = InferenceClient(
        provider="auto",
        api_key=st.secrets["HF_TOKEN"]
    )

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


with tabs[2]:
    st.title("🎓 AI Карьерный консультант для старшеклассников")

    student_question = st.text_area("Введите ваш вопрос о карьере:", height=100)

    use_rag = st.toggle("Включить RAG (добавление знаний из базы)", value=True,
                        help="Использовать дополнительные материалы для улучшения ответа модели.")

    if st.button("Получить совет"):
        with st.spinner("Генерация ответа..."):
            if use_rag:
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("💡 Базовая модель")
                    base_answer = generate_career_advice(student_question)
                    st.write(base_answer)

                with col2:
                    st.subheader("📚 Модель с RAG")
                    rag_answer = generate_rag_career_advice(student_question, embedder, annoy_index, texts)
                    st.write(rag_answer)

            else:
                st.subheader("AI Совет по карьере")
                base_answer = generate_career_advice(student_question)
                st.write(base_answer)