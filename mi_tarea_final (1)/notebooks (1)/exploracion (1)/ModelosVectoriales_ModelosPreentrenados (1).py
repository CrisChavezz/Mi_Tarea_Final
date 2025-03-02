#!/usr/bin/env python
# coding: utf-8

# In[19]:


#BoW

from sklearn.feature_extraction.text import CountVectorizer

# Ejemplo de documentos
documentos = [
    "El gato se sentó en la alfombra",
    "El perro se acostó en la alfombra",
    "El gato y el perro son amigos"
]

# Crear el modelo BoW
vectorizador = CountVectorizer()
X = vectorizador.fit_transform(documentos)

# Mostrar las palabras y sus índices
print("Vocabulario:", vectorizador.vocabulary_)

# Mostrar la matriz BoW
print("Matriz BoW:\n", X.toarray())


# In[21]:


#pip install matplotlib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

# Cargar el conjunto de datos de reseñas de películas
url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/yelp.csv"
data = pd.read_csv(url)

# Seleccionar una muestra de las reseñas
muestra = data['text'][:100]

# Crear el modelo BoW con eliminación de palabras vacías
vectorizador = CountVectorizer(stop_words='english')
X = vectorizador.fit_transform(muestra)

# Mostrar las palabras y sus índices
#print("Vocabulario:", vectorizador.vocabulary_)

# Mostrar la matriz BoW
#print("Matriz BoW:\n", X.toarray())

# Visualizar las palabras más frecuentes
frecuencias = X.sum(axis=0).A1
palabras = vectorizador.get_feature_names_out()
frecuencia_palabras = pd.DataFrame({'palabra': palabras, 'frecuencia': frecuencias})
frecuencia_palabras = frecuencia_palabras.sort_values(by='frecuencia', ascending=False)

# Graficar las palabras más frecuentes
plt.figure(figsize=(10, 6))
plt.bar(frecuencia_palabras['palabra'][:20], frecuencia_palabras['frecuencia'][:20])
plt.xlabel('Palabras')
plt.ylabel('Frecuencia')
plt.title('Palabras más frecuentes en las reseñas de películas')
plt.xticks(rotation=90)
plt.show()


# In[8]:


from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

# Descargar los datos necesarios para la tokenización
nltk.download('punkt')

# Ejemplo de corpus de texto
corpus = [
    "El gato se sentó en la alfombra",
    "El perro se acostó en la alfombra",
    "El gato y el perro son amigos"
]

# Tokenizar el corpus
tokenized_corpus = [word_tokenize(doc.lower()) for doc in corpus]

# Entrenar el modelo Word2Vec
model = Word2Vec(sentences=tokenized_corpus, vector_size=50, window=5, min_count=1, workers=4)

# Obtener el vector de una palabra
vector_gato = model.wv['gato']
print("Vector para 'gato':", vector_gato)

# Encontrar palabras similares
similar_words = model.wv.most_similar('gato')
print("Palabras similares a 'gato':", similar_words)


# In[22]:


from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Descargar los datos necesarios para la tokenización y stop words
nltk.download('punkt')
nltk.download('stopwords')

# Ejemplo de corpus de texto
corpus = [
    "El gato se sentó en la alfombra",
    "El perro se acostó en la alfombra",
    "El gato y el perro son amigos"
]

# Obtener la lista de stop words en español
stop_words = set(stopwords.words('spanish'))

# Tokenizar el corpus y eliminar stop words
tokenized_corpus = [
    [word for word in word_tokenize(doc.lower()) if word.isalnum() and word not in stop_words]
    for doc in corpus
]

# Entrenar el modelo Word2Vec
model = Word2Vec(sentences=tokenized_corpus, vector_size=50, window=5, min_count=1, workers=4)

# Obtener el vector de una palabra
vector_gato = model.wv['gato']
print("Vector para 'gato':", vector_gato)

# Encontrar palabras similares
similar_words = model.wv.most_similar('gato')
print("Palabras similares a 'gato':", similar_words)


# In[26]:


from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

# Cargar el modelo y el tokenizador preentrenados
modelo = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizador = BertTokenizer.from_pretrained('bert-base-uncased')

# Crear un pipeline de clasificación de sentimientos
clasificador = pipeline('sentiment-analysis', model=modelo, tokenizer=tokenizador)

# Ejemplo de texto
texto = "I love this product! It's amazing and works perfectly."

# Realizar la clasificación de sentimientos
resultado = clasificador(texto)
print("Resultado de la clasificación:", resultado)


# In[18]:


from transformers import pipeline

# Crear un pipeline de clasificación de sentimientos con un modelo específico
clasificador = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

# Ejemplo de texto
texto = "I love this product! It's amazing and works perfectly."

# Realizar la clasificación de sentimientos
resultado = clasificador(texto)
print("Resultado de la clasificación:", resultado)


# In[27]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Descargar el conjunto de stop words en español
nltk.download('stopwords')
nltk.download('punkt')

texto = "Este es un ejemplo de texto para la eliminación de stop words."
tokens = word_tokenize(texto)

# Cargar las stop words en español
stop_words = set(stopwords.words('spanish'))

# Filtrar las stop words
tokens_filtrados = [word for word in tokens if word.lower() not in stop_words]
print(tokens_filtrados)


# In[ ]:




