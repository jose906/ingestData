import nltk

nltk.download('all')

import pandas as pd
import re

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

combined_data_df = pd.read_excel('data/combined_data.xlsx')
all_tweets_df = pd.read_excel('data/all_tweets_clasificados.xlsx')

all_tweets_first_4k_df = all_tweets_df.head(4000)

print("Shape of the new DataFrame (first 4000 rows):", all_tweets_first_4k_df.shape)
print("Head of the new DataFrame:")
all_tweets_first_4k_df.head()


columns_to_keep = ['text', 'sentimiento', 'categoria']
all_tweets_first_4k_df = all_tweets_first_4k_df[columns_to_keep]

print("DataFrame after dropping columns (head):")
all_tweets_first_4k_df.head()


columns_to_keep_df = ['text', 'tipo', 'sent']
first_10k = combined_data_df[columns_to_keep_df]

print("DataFrame after dropping columns (head):")
first_10k.head()

first_10k_renamed = first_10k.rename(columns={'tipo': 'categoria', 'sent': 'sentimiento'})

# Concatenate the two DataFrames
combined_final_df = pd.concat([all_tweets_first_4k_df, first_10k_renamed], ignore_index=True)

print("Combined DataFrame head:")
combined_final_df.head()


sentiment_mapping = {-1: 'negativo', 0: 'neutro', 1: 'positivo'}
combined_final_df['sentimiento'] = combined_final_df['sentimiento'].replace(sentiment_mapping)

print("Combined DataFrame head after sentiment mapping:")
combined_final_df.head()

# Define the valid sentiment values
valid_sentiments = ['negativo', 'neutro', 'positivo']

# Filter out rows where 'sentimiento' is not in the valid_sentiments list
combined_final_df = combined_final_df[combined_final_df['sentimiento'].isin(valid_sentiments)]

print("Combined DataFrame head after removing dirty sentiment values:")
print(combined_final_df.head())

print("\nValue counts for 'sentimiento' column after cleaning:")
print(combined_final_df['sentimiento'].value_counts())

print("\nCombined DataFrame shape after cleaning:", combined_final_df.shape)

# Remove rows where 'categoria' is 'Otros'
combined_final_df = combined_final_df[combined_final_df['categoria'] != 'Otros']

print("\nValue counts for 'categoria' column after removing 'Otros' category:")
print(combined_final_df['categoria'].value_counts())
print("\nCombined DataFrame shape after removing 'Otros' category:", combined_final_df.shape)

combined_final_df = combined_final_df[combined_final_df['categoria'] != 'otros']

print("\nValue counts for 'categoria' column after removing 'Otros' category:")
print(combined_final_df['categoria'].value_counts())
print("\nCombined DataFrame shape after removing 'Otros' category:", combined_final_df.shape)

text = list(combined_final_df['text'])

lemmatizer = WordNetLemmatizer()
corpus = []

for i in range(len(text)):
    r = re.sub('[^a-zA-Z]', ' ', text[i])
    r = r.lower()
    r = r.split()
    r = [word for word in r if word not in stopwords.words('english')]
    r = [lemmatizer.lemmatize(word) for word in r]
    r = ' '.join(r)
    corpus.append(r)

combined_final_df['text'] = corpus
combined_final_df.head()



print("Shape before removing duplicates:", combined_final_df.shape)
combined_final_df.drop_duplicates(inplace=True)
print("Shape after removing duplicates:", combined_final_df.shape)
print("\nCombined DataFrame head after removing duplicates:")
combined_final_df.head()

combined_final_df = combined_final_df.sample(frac=1, random_state=42).reset_index(drop=True)


from sklearn.model_selection import train_test_split

X = combined_final_df['text'].astype(str)
y = combined_final_df['sentimiento'].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train:", len(X_train), "Test:", len(X_test))


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

svm_sentimiento = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),   # unigrams + bigrams
        min_df=3,
        max_df=0.9
    )),
    ("svm", SVC(
        kernel="linear",
        probability=True,     # por si luego quieres usar probabilidades
        class_weight="balanced",
        random_state=42
    ))
])

svm_sentimiento.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix

y_pred = svm_sentimiento.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

import joblib

# Save the model
joblib.dump(svm_sentimiento, 'svm_sentimiento_classifier.joblib')

print("Model saved as svm_sentimiento_classifier.joblib")