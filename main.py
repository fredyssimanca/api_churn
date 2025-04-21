from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Cargar el modelo entrenado
model = joblib.load("modelo_random_forest_churn.pkl")

# Inicializar FastAPI
app = FastAPI(title="API Churn Prediction")

# Definir esquema de entrada
class Cliente(BaseModel):
    ultima_actividad_dias: int
    nivel_satisfaccion: float
    cursos_completados: int
    horas_totales_plataforma: float
    edad: int

# Ruta de prueba
@app.get("/")
def read_root():
    return {"mensaje": "API activa: Predicción de churn"}

# Ruta de predicción
@app.post("/predict")
def predict_churn(cliente: Cliente):
    datos = [[
        cliente.ultima_actividad_dias,
        cliente.nivel_satisfaccion,
        cliente.cursos_completados,
        cliente.horas_totales_plataforma,
        cliente.edad
    ]]
    prediccion = model.predict(datos)[0]
    proba = model.predict_proba(datos)[0][1]
    return {
        "churn": int(prediccion),
        "probabilidad_churn": round(proba, 3)
    }
