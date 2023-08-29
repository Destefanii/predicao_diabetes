import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Carregar o conjunto de dados
df = pd.read_csv('C:/Users/Destefani/Desktop/streamlite/diabetes.csv', sep=',')

# Separar os atributos (features) e rótulos (labels)
X = df.iloc[:, :-1]
y = df['Outcome']

# Dividir o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar os atributos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Criar o modelo SVM
model = SVC(kernel='linear', random_state=42)

# Treinar o modelo
model.fit(X_train_scaled, y_train)

# Fazer previsões no conjunto de treinamento
y_train_pred = model.predict(X_train_scaled)

# Calcular a precisão (accuracy) do modelo no conjunto de treinamento
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Acurácia no conjunto de treinamento: {train_accuracy:.2f}")

# Salvar o modelo treinado
joblib.dump(model, 'svm_model.pkl')

