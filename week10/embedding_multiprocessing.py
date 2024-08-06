import numpy as np
import multiprocessing
from gensim.models import KeyedVectors
from transformers import BertTokenizer, TFBertModel
import pandas as pd

wine_df = pd.read_csv('data/winemag-data_first150k.csv')
corpus = wine_df['description']

model_path = 'data/GoogleNews-vectors-negative300.bin.gz'

# Cargar el modelo Word2Vec preentrenado
word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=True)

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertModel.from_pretrained('bert-base-uncased')

# Definir la función calculate_embeddings globalmente
def calculate_embeddings(text):
    tokens = text.lower().split()
    word_vectors = [word2vec_model[word] for word in tokens if word in word2vec_model]
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)

def generate_word2vec_embeddings(texts):
    # Crear un pool de procesos
    pool = multiprocessing.Pool()

    # Calcular los embeddings para cada texto en paralelo usando map
    embeddings = pool.map(calculate_embeddings, texts)

    # Cerrar el pool y esperar a que todos los procesos terminen
    pool.close()
    pool.join()

    return np.array(embeddings)

def calculate_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors='tf', padding=True, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

def generate_bert_embeddings(texts):
    # Usar multiprocessing para calcular embeddings en paralelo
    with multiprocessing.Pool(processes=4) as pool:
        embeddings = pool.map(calculate_bert_embeddings, texts)
    return np.array(embeddings)


if __name__ == "__main__":
    word2vec_embeddings = generate_word2vec_embeddings(corpus)
    print("Word2Vec Embeddings:", word2vec_embeddings)
    print("Word2Vec Shape:", word2vec_embeddings.shape)

    bert_embeddings = generate_bert_embeddings(corpus)
    print("BERT Embeddings:", bert_embeddings)
    print("BERT Embeddings Shape:", bert_embeddings.shape)