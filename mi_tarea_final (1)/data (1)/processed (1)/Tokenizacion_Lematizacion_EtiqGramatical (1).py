#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# pip install nltk
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Texto de ejemplo
texto = "-The cats are running and jumped on the table"
tokens = word_tokenize(texto)

# Inicializar el lematizador
wnl = WordNetLemmatizer()

# Realizar la lematización y formatear la salida
print("{0:20}{1:20}".format("--Palabra--", "--Lema--"))
for token in tokens:
    print("{0:20}{1:20}".format(token, wnl.lemmatize(token, pos="v")))


# In[ ]:


#pip install spacy
#!python -m spacy download es_core_news_sm

import spacy

# Cargar el modelo de spaCy para español
nlp = spacy.load("es_core_news_sm")

# Texto de ejemplo
texto = "Los gatos están corriendo y saltaron sobre la mesa."
doc = nlp(texto)

# Realizar la tokenización y lematización, y formatear la salida
print("{0:20}{1:20}".format("--Palabra--", "--Lema--"))
for token in doc:
    print("{0:20}{1:20}".format(token.text, token.lemma_))


# In[ ]:


import spacy

# Cargar el modelo de spaCy para español
nlp = spacy.load("es_core_news_sm")

# Texto de ejemplo
texto = "El gato corre rápido."
doc = nlp(texto)

# Realizar el etiquetado gramatical y mostrar los resultados
print("{0:20}{1:20}".format("--Palabra--", "--Etiqueta--"))
for token in doc:
    print("{0:20}{1:20}".format(token.text, token.pos_))

