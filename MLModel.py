import joblib
import numpy as np
#import spacy
# Load the model
svm_pipeline = joblib.load('model/svm_category_classifier.joblib')
svm_sentiment_model = joblib.load('model/svm_sentimiento_classifier.joblib')
#nlp = spacy.load("es_core_news_lg")
print("Model loaded successfully.")

def predecir_categoria(texto: str, modelo=svm_pipeline, umbral=0.45):
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
    

