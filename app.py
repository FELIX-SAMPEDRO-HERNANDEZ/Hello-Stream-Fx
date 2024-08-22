import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#Page configuration
st.set_page_config (
  page_title = 'Iris prediccion',
  layout = 'wide',
  initial_sidebar_state = 'expanded'
)

# Titulo de la app
st.title ('Prediccion clase de orquidea')
# 

# Lectura de datos
iris = load_iris ()
df = pd.DataFrame (iris.data, columns = iris.feature_names)

# Entradas
lengthS = st.sidebar.slider ('sepal length (cm)', 0.0, 1.0, 0.5)
widthS = st.sidebar.slider ('sepal width (cm)', 0.0, 1.0, 0.5)
lengthP = st.sidebar.slider ('petal length (cm)', 0.0, 1.0, 0.5)
widthP = st.sidebar.slider ('petal width (cm)', 0.0, 1.0, 0.5)
st.write (df)


df ['target']=iris.target
# Dividir dataset
X_train, X_test, y_train, y_test = train_test_split (
    df.drop (['target'], axis='columns'), iris.target,
    test_size = 0.2
)
# Creación del modelo
model = RandomForestClassifier ()
# Entrenamiento
model.fit (X_train, y_train)
model.score (X_test, y_test)
y_predicted = model.predict (X_test)

# Entradas
st.write("Entradas")
st.write(lengthS,widthS,lengthP,widthP)
# Salidas 
st.write("Hola")
