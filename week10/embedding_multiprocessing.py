import numpy as np
import multiprocessing
from gensim.models import KeyedVectors
from transformers import BertTokenizer, TFBertModel
import pandas as pd
from joblib import Parallel, delayed

num_cores = multiprocessing.cpu_count()
print(f"Número de núcleos disponibles: {num_cores}")

# Cargar el dataset
wine_df = pd.read_csv('/Users/pabloarcos/Documents/EPN/7 SEMESTRE/RI/ir24a/week10/data/winemag-data_first150k.csv')
corpus = wine_df['description'].tolist()

# Ruta al modelo Word2Vec preentrenado
model_path = '/Users/pabloarcos/Documents/EPN/7 SEMESTRE/RI/ir24a/week10/data/GoogleNews-vectors-negative300.bin.gz'

# Cargar el modelo Word2Vec preentrenado
word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=True)

# Cargar el modelo y tokenizador BERT preentrenado
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Definir funciones para calcular embeddings
def calculate_word2vec_embeddings(text):
    tokens = text.lower().split()
    word_vectors = [word2vec_model[word] for word in tokens if word in word2vec_model]
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)

def generate_word2vec_embeddings(texts, n_jobs=6):
    return np.array(Parallel(n_jobs=n_jobs)(delayed(calculate_word2vec_embeddings)(text) for text in texts))

def calculate_bert_embeddings(text):
    # Inicializar el modelo y tokenizador dentro de la función
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TFBertModel.from_pretrained('bert-base-uncased')
    
    inputs = tokenizer(text, return_tensors='tf', padding=True, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

def generate_bert_embeddings(texts, n_jobs=6):
    return np.array(Parallel(n_jobs=n_jobs)(delayed(calculate_bert_embeddings)(text) for text in texts))

# Reduzca el tamaño del corpus para pruebas iniciales
corpus_sample = corpus[:10]

# Ejemplo de uso: Generar Word2Vec embeddings
word2vec_embeddings = generate_word2vec_embeddings(corpus_sample, n_jobs=6)
print("Word2Vec Embeddings:", word2vec_embeddings)
print("Word2Vec Shape:", word2vec_embeddings.shape)

# Ejemplo de uso: Generar BERT embeddings
bert_embeddings = generate_bert_embeddings(corpus_sample, n_jobs=6)
print("BERT Embeddings:", bert_embeddings)
print("BERT Embeddings Shape:", bert_embeddings.shape)
