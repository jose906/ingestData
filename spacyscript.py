import spacyscript
import mysql.connector
from typing import List, Tuple
from mysql.connector import Error
import spacy
nlp = spacyscript.load("es_core_news_lg")

# Configura tu conexión MySQL
db_config = {
    "host": "34.69.57.221",      # o la IP de tu contenedor / Cloud SQL
    "user": "admin",
    "password": "Admin123!",
    "database": "Analisis",
    "port": 3306,
    
}
# --- 2. Función para extraer entidades ---
def get_entities(text):
    if not text or not isinstance(text, str):
        return {"PER": [], "ORG": [], "LOC": [], "MISC": []}

    doc = nlp(text)
    entidades = {"PER": [], "ORG": [], "LOC": [], "MISC": []}

    for ent in doc.ents:
        if ent.label_ in entidades:
            entidades[ent.label_].append(ent.text)
        else:
            entidades["MISC"].append(ent.text)
    return entidades



