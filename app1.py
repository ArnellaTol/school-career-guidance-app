import os
import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# Set Streamlit page configuration
st.set_page_config(page_title="Career guidance for students")

# Mapping of raw column names to display names in the UI
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

# Columns for checkboxes
checkbox_columns = [
    'Activist', 'Career', 'Tester', 'Creator', 'Designer', 'Researcher'
]

# Create expander for a given grade's columns
def create_expander(class_label, cols):
    with st.expander(f"Grades for the {class_label}th grade"):
        for col in cols:
            input_values[col] = st.number_input(
                column_names_dict[col],
                min_value=2,
                max_value=5,
                step=1,
                value=5,
                key=col
            )

# Dictionary to store numeric inputs
input_values = {}

# Merge checkbox and number input values into a DataFrame
def save_to_dataframe(selected_checkboxes, input_values):
    data = {**selected_checkboxes, **input_values}
    for key in checkbox_columns:
        data[key] = int(data.get(key, False))
    col_order = list(column_names_dict.keys())
    return pd.DataFrame([data], columns=col_order)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load model and apply it to input dataframe
def apply_model(model_path, input_df):
    logging.debug("Loading model")
    model = joblib.load(model_path)
    
    logging.debug("Predicting probabilities")
    probabilities = model.predict_proba(input_df)
    logging.debug(f"Probabilities type: {type(probabilities)}")
    logging.debug(f"Probabilities: {probabilities}")
    if isinstance(probabilities, list):
        probabilities = np.array(probabilities)
    logging.debug(f"Converted type: {type(probabilities)}")
    
    # Build probabilities dict for each class
    prob_dict = {
        f'class_{i}': probabilities[i][:, 1]
        for i in range(len(probabilities))
    }
    logging.debug("Creating results DataFrame")
    return pd.DataFrame(prob_dict)

# Mapping classes to professional types
type_columns = {
    'class_0': 'Person-Sign System',
    'class_1': 'Person-Technology',
    'class_2': 'Person-Nature',
    'class_3': 'Person-Artistic Image',
    'class_4': 'Person-Person',
    'class_5': 'Person-Business'
}

# Thresholds for each class
thresholds = {
    'class_0': 0.39,
    'class_1': 0.30903005409623036,
    'class_2': 0.23611111111111113,
    'class_3': 0.44833333333333336,
    'class_4': 0.13,
    'class_5': 0.17
}

# Adjust probabilities (cap at 100%)
def adjust_probabilities(probabilities, thresholds):
    adjusted = {}
    for key, value in probabilities.items():
        adjusted[key] = min(100, (value / thresholds[key]) * 100)
    return adjusted

# Display results as text and pie chart
def display_results(df):
    results = {key: df[key].values[0] for key in df.columns}
    adj_results = adjust_probabilities(results, thresholds)
    selected_types = [
        type_columns[key]
        for key, value in adj_results.items()
        if value >= 100
    ]
    st.write("**The most appropriate types of professional readiness, derived by L.N. Kabardova:**")
    for typ in selected_types:
        st.write(f"- {typ}")
    st.write("**The probabilities of all types:**")
    fig, ax = plt.subplots()
    ax.pie(
        list(adj_results.values()),
        labels=[type_columns[key] for key in adj_results.keys()],
        autopct='%1.1f%%'
    )
    ax.axis("equal")
    st.pyplot(fig)


# --------------- Streamlit UI 

st.header("Career guidance for students")
tabs = st.tabs([
    "School academic records", 
    "Open questions", 
    "Test ...(under development)"
])

with tabs[0]:
    st.write("**Choose your motivation type (7th grade school questionnaire):**")
    selected_checkboxes = {}
    for col in checkbox_columns:
        selected_checkboxes[col] = st.checkbox(column_names_dict[col])
    
    # Create grade sections
    create_expander(7, [col for col in column_names_dict.keys() if col.endswith('_7')])
    create_expander(8, [col for col in column_names_dict.keys() if col.endswith('_8')])
    create_expander(9, [col for col in column_names_dict.keys() if col.endswith('_9')])
    create_expander(10, [col for col in column_names_dict.keys() if col.endswith('_10')])
    
    if st.button("Get a result"):
        df = save_to_dataframe(selected_checkboxes, input_values)
        result_df = apply_model("random_forest_model.pkl", df)
        display_results(result_df)

# OpenAI Integration 
load_dotenv()
my_api = os.getenv("API_KEY")
print(my_api)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", my_api))

def get_ai_response(answers):
    completion = client.chat.completions.create(
        model="ft:gpt-4o-2024-08-06:personal::An4sVvnb",
        messages=[
            {
                "role": "system",
                "content": "Assistant is an expert in career guidance. Answer in English."
            },
            {
                "role": "user",
                "content": (
                    f"1. {answers[0]} 2. {answers[1]} 3. {answers[2]} "
                    f"4. {answers[3]} 5. {answers[4]} 6. {answers[5]} "
                    f"7. {answers[6]}"
                )
            }
        ]
    )
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content

questions = [
    "**1. What professions are you interested in at the moment?**",
    "**2. What types of activities are you definitely not interested in?**",
    "**3. Without considering financial aspects, what types of activities or professions do you like?**",
    "**4. List your hobbies and interests:**",
    "**5. Name role models whose lifestyles and achievements inspire you.**\n\n(Examples: Elon Musk, Hermione Granger, etc.)",
    "**6. What tasks energize you?**",
    "**7. What tasks drain you?**"
]

user_answers = []

with tabs[1]:
    for i, question in enumerate(questions):
        answer = st.text_input(f"{question}", key=f"answer_{i}")
        user_answers.append(answer)
    
    if st.button("Get a response"):
        ai_response = get_ai_response(user_answers)
        st.write("AI response:")
        st.write(ai_response)




