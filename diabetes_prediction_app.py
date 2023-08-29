# Bibliotecas
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import joblib
import plotly.express as px

# Carregar o modelo SVM treinado
model = joblib.load("svm_model.pkl")

# Carregando conjunto de dados
df = pd.read_csv("C:/Users/Destefani/Desktop/streamlite/diabetes.csv", sep=",")

# Configurações streamlit
st.set_page_config(layout="wide")
st.set_option("deprecation.showPyplotGlobalUse", False)

# Visualizacao streamlit
aba1, aba2 = st.tabs(["Previsão de diabetes","Graficos"])

#Tabelas
long_df = px.data.medals_long()

df_outcame_count = df[["Outcome"]].value_counts("Outcome")

df_diabetic_age_count = df[["Age","Outcome"]].value_counts()
df_diabetic_age_count = df_diabetic_age_count.reset_index(drop=False)

# Função para fazer a previsão
def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age):
    data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
    prediction = model.predict(data)
    return prediction[0]

def main():
    
    with aba1:
        st.title("Prevendo Diabetes")

        # Inputs para informações ao modelo de ML
        user_name = st.text_input("Nome de usuario", placeholder="Insira seu nome")
        pregnancies = st.number_input("Gravidez", min_value=0, max_value=20, value=1)
        glucose = st.number_input("Glicose", min_value=0, value=100)
        blood_pressure = st.number_input("Pressão Sanguínea", min_value=0, value=70)
        skin_thickness = st.number_input("Espessura da Pele", min_value=0, value=20)
        insulin = st.number_input("Insulina", min_value=0, value=79)
        bmi = st.number_input("IMC", min_value=0.0, value=25.0)
        diabetes_pedigree = st.number_input("Pedigree de Diabetes", min_value=0.0, value=0.47)
        age = st.number_input("Idade", min_value=0, value=30)

        if st.button("Prever"):
            # Resultado de 0 ou 1
            # 0 = não possui diabetes | 1 = possui diabetes
            result = predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age)
            if result == 0:
                st.success(f"Parabéns {user_name} você não possui diabetes.")
            else:
                st.error(f"Hey {user_name} cuidado, você possui diabetes.")

    with aba2:
        coluna1, coluna2 = st.columns(2)
        with coluna1:
            
            st.title("Contagem de casos Diabetes vs. Não Diabetes")

            # Calcular a contagem de ocorrências de cada valor na coluna "Outcome"
            outcome_counts = df["Outcome"].value_counts() 
            outcome_counts = outcome_counts.reset_index(drop=False)

            # Desenvolvimento do grafico
            fig_pizza = px.pie(outcome_counts, 
                               values="count", 
                               names="Outcome")

            st.plotly_chart(fig_pizza, use_container_width=True)

    
        with coluna2:
            
            st.title("Contagem de Diabetes vs. Não Diabetes por idade")

            # Desenvolvimento do grafico
            fig = px.histogram(df_diabetic_age_count, 
                                x="Age", 
                                y="count", 
                                color="Outcome", 
                                barmode="group")
            
            st.plotly_chart(fig, use_container_width=True)

    st.title("Contagem de pesquisas por idade")

    # Desenvolvimento do grafico
    fig_bar = px.bar(df_diabetic_age_count, 
                    x="count",
                    y="Age",
                    color="Age", 
                    text="Outcome",
                    orientation="h",
                    title="Contagem de pesquisas por idade")
    
    st.plotly_chart(fig_bar, use_container_width=True)

if __name__ == "__main__":
    main()