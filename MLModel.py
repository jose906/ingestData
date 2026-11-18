from pathlib import Path

import joblib
import numpy as np
MODEL_DIR = Path(__file__).resolve().parent / "model"
svm_pipeline = joblib.load(MODEL_DIR / "svm_category_classifier2.joblib")
svm_sentiment_model = joblib.load(MODEL_DIR / "svm_sentiment_classifier2.joblib")

def predecir_categoria(texto: str, modelo=svm_pipeline, umbral=0.30):
    """
    Devuelve (categoria_predicha, confianza_maxima).
    Si la confianza es menor al umbral, devuelve 'Otros'.
    """
    # Aseguramos que `texto` sea lista
    probs = modelo.predict_proba([texto])[0]
    clases = modelo.classes_

    idx_max = np.argmax(probs)
    categoria = clases[idx_max]
    confianza = probs[idx_max]

    if confianza < umbral:
        return "Otros", float(confianza)
    else:
        return categoria, float(confianza)
def get_sentiment(texto):
    """
    Devuelve (sentimiento_predicho, confianza_maxima)
    """
    probs = svm_sentiment_model.predict_proba([texto])[0]
    clases = svm_sentiment_model.classes_
    
    idx_max = np.argmax(probs)
    sentimiento = clases[idx_max]
    confianza = float(probs[idx_max])
    
    return sentimiento, confianza
    

