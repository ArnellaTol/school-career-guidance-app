import streamlit as st
import pandas as pd
# import numpy as np
# import joblib

import matplotlib.pyplot as plt
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

def apply_model(_, input_df):
    # Вычисляем средние по блокам предметов
    math_avg = input_df[['math_7', 'math_8', 'math_9', 'math_10']].mean(axis=1).values[0]
    physics_avg = input_df[['physics_7', 'physics_8', 'physics_9', 'physics_10']].mean(axis=1).values[0]
    biology_avg = input_df[['biology_7', 'biology_8', 'biology_9', 'biology_10']].mean(axis=1).values[0]
    literature_avg = input_df[['liter_7', 'liter_8', 'liter_9', 'liter_10']].mean(axis=1).values[0]
    art_avg = input_df[['art_7', 'art_8', 'art_9', 'art_10']].mean(axis=1).values[0]
    history_avg = input_df[['kaz_hist_7', 'kaz_hist_8', 'kaz_hist_9', 'kaz_hist_10']].mean(axis=1).values[0]
    comps_avg = input_df[['comps_7', 'comps_8', 'comps_9', 'comps_10']].mean(axis=1).values[0]

    # Чекбоксы мотивации
    activist = input_df['Activist'].values[0]
    career = input_df['Career'].values[0]
    tester = input_df['Tester'].values[0]
    creator = input_df['Creator'].values[0]
    designer = input_df['Designer'].values[0]
    researcher = input_df['Researcher'].values[0]

    # Словарь "склонностей"
    probabilities = {
        'class_0': 0,  # Знаковая система
        'class_1': 0,  # Техника
        'class_2': 0,  # Природа
        'class_3': 0,  # Художественный образ
        'class_4': 0,  # Человек
        'class_5': 0   # Бизнес
    }

    # --- Условная логика ---
    # Математика + информатика -> знаковая система
    if math_avg >= 4.5 and comps_avg >= 4.6:
        probabilities['class_0'] += 0.7
    elif math_avg >= 4.0:
        probabilities['class_0'] += 0.4

    # Физика + математика -> техника
    if physics_avg >= 4.0 and math_avg >= 4.6:
        probabilities['class_1'] += 0.6
    if designer or tester:
        probabilities['class_1'] += 0.3

    # Биология + химия -> природа
    if biology_avg >= 4.5:
        probabilities['class_2'] += 0.5
    if researcher:
        probabilities['class_2'] += 0.4

    # Литература + искусство -> художественный образ
    if literature_avg >= 4.0 or art_avg >= 4.5:
        probabilities['class_3'] += 0.5
    if creator:
        probabilities['class_3'] += 0.4

    # История + языки -> человек-человек
    if history_avg >= 4.5:
        probabilities['class_4'] += 0.4
    if activist:
        probabilities['class_4'] += 0.4

    # Карьерист + хорошие оценки по обществознанию/праву -> бизнес
    if career:
        probabilities['class_5'] += 0.5
    if input_df['rights_9'].values[0] >= 4:
        probabilities['class_5'] += 0.3

    # Нормализуем так, чтобы было похоже на вероятности
    total = sum(probabilities.values())
    if total > 0:
        for key in probabilities:
            probabilities[key] = probabilities[key] / total

    # В том же формате, как раньше
    result_df = pd.DataFrame([probabilities])
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
    selected_types = [type_columns_ru[key] for key, value in adjusted_results.items() if value >= 100]

    st.write("**Наиболее подходящие типы профессиональной склонности, выведенные Дж. Холландом:**")
    # st.write("**The most appropriate types of professional readiness, derived by L.N. Kabardova:**")
    for typ in selected_types:
        st.write(f"- {typ}")

    st.write("**Вероятности всех типов профессиональной склонности:**")
    # st.write("**The probabilities of all types of professional readiness:**")
    fig, ax = plt.subplots()

    # type_columns_ru - type_columns
    ax.pie(adjusted_results.values(), labels=[type_columns_ru[key] for key in adjusted_results.keys()], autopct='%1.1f%%')
    ax.axis('equal')
    st.pyplot(fig)

    



# st.header("Career guidance for students")
st.header("Профориентация для школьников")
# tabs = st.tabs(["School academic records", "Open questions"]) # , "Test ...(under development)"
tabs = st.tabs(["Школьные оценки", "Открытые вопросы"]) # , "Тест ...(в разработке)"

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


from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

my_api = st.secrets["OPENAI_API_KEY"]

print(my_api)

client = OpenAI(api_key=my_api)
def get_ai_response(answers):
    completion = client.chat.completions.create(
        model="ft:gpt-4o-2024-08-06:personal::An4sVvnb",
        messages=[
        {"role": "system", "content": f"Assistant is an expert in career guidance. Assistant should answer in the same language as the one in which the user writes answers (could be russian, kazakh, english). User answers the following questions: 1. Какие профессии вас интересуют на данный момент? 2. Какие виды деятельности вам точно не интересны? 3. Без учета финансовых аспектов, какие виды деятельности или профессии вам нравятся? 4. Перечислите свои хобби и интересы: 5. Назовите ролевые модели, чей образ жизни и достижения вас вдохновляют. 6. Какие задачи придают вам энергии? 7. Какие задачи вас утомляют?"},
        {"role": "user", "content": f"1. {answers[0]} 2. {answers[1]} 3. {answers[2]} 4. {answers[3]} 5. {answers[4]} 6. {answers[5]} 7. {answers[6]}"},
        ]
    )
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content


questions = [
    "**1. What professions are you interested in at the moment?**",
    "**2. What types of activities are you definitely not interested in?**",
    "**3. Without considering financial aspects, what types of activities or professions do you like?**",
    "**4. List your hobbies and interests:**",
    "**5. Name role models whose lifestyles and achievements inspire you.**\n\n(This list may include famous personalities, characters from books, or your relatives and acquaintances. If you mention relatives or acquaintances, describe what achievements or personal qualities inspire you about them.)\n\nExamples:\n\n• Elon Musk — ambition and ability to achieve the impossible.\n\n• Hermione Granger — determination and curiosity.\n\n• Grandmother — always remains optimistic despite tough times.\n\n• Mathematics teacher — love for the subject and charisma.",
    "**6. What tasks energize you?**\n\n(For example: solving mathematical problems, programming, drawing, reading fiction books, working with people, discussing ideas.)",
    "**7. What tasks drain you?**\n\n(For example: performing routine tasks, working with numbers for long periods, monotonous reports, interacting with many people, etc.)"
]


questions_ru = [
    "**1. Какие профессии вас интересуют на данный момент?**",
    "**2. Какие виды деятельности вам точно не интересны?**",
    "**3. Без учета финансовых аспектов, какие виды деятельности или профессии вам нравятся?**",
    "**4. Перечислите свои хобби и интересы:**",
    "**5. Назовите ролевые модели, чьи образы жизни и достижения вас вдохновляют.**\n\n(В этот список могут входить известные личности, персонажи из книг или ваши родственники и знакомые. Если вы упоминаете родственников или знакомых, опишите, какими достижениями или личными качествами они вас вдохновляют.)\n\nПримеры:\n\n• Илон Маск — амбициозность и способность добиваться невозможного.\n\n• Гермиона Грейнджер — целеустремленность и любознательность.\n\n• Бабушка — несмотря на тяжелые времена, всегда остается оптимистичной.\n\n• Учитель математики — любовь к своему предмету и харизма.",
    "**6. Какие задачи придают вам энергии?**\n\n(Например: решение математических задач, программирование, рисование, чтение художественных книг, работа с людьми, обсуждение идей.)",
    "**7. Какие задачи вас утомляют?**\n\n(Например: выполнение рутинных заданий, длительная работа с цифрами, монотонные отчеты, взаимодействие с большим количеством людей и пр.)"
]


user_answers = []

with tabs[1]:
    for i, question in enumerate(questions_ru): #questions
        answer = st.text_input(f"{question}", key=f"answer_{i}") 
        user_answers.append(answer)
    
    # if st.button("Get a response"):
    if st.button("Получить ответ"):
        ai_response = get_ai_response(user_answers)
        # st.write("AI response:") 
        st.write("Ответ ИИ:")
        st.write(ai_response)
    

