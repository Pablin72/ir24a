import numpy as np
import multiprocessing
from gensim.models import KeyedVectors
from transformers import BertTokenizer, TFBertModel
import pandas as pd

dataframe_wine = pd.read_csv('week10/data/winemag-data_first150k.csv')
text_corpus = dataframe_wine['description']

path_to_model = 'data/GoogleNews-vectors-negative300.bin.gz'

# Cargar el modelo Word2Vec preentrenado
vector_model = KeyedVectors.load_word2vec_format(path_to_model, binary=True)

# Load pre-trained BERT model and tokenizer
text_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Definir la función calculate_embeddings globalmente
def compute_word2vec_embeddings(text):
    tokens = text.lower().split()
    vectors = [vector_model[word] for word in tokens if word in vector_model]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(vector_model.vector_size)

def generate_word2vec_embeddings(texts):
    # Crear un pool de procesos
    process_pool = multiprocessing.Pool()

    # Calcular los embeddings para cada texto en paralelo usando map
    embeddings = process_pool.map(compute_word2vec_embeddings, texts)

    # Cerrar el pool y esperar a que todos los procesos terminen
    process_pool.close()
    process_pool.join()

    return np.array(embeddings)

def compute_bert_embeddings(text):
    inputs = text_tokenizer(text, return_tensors='tf', padding=True, truncation=True)
    outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

def generate_bert_embeddings(texts):
    # Usar multiprocessing para calcular embeddings en paralelo
    with multiprocessing.Pool(processes=4) as process_pool:
        embeddings = process_pool.map(compute_bert_embeddings, texts)
    return np.array(embeddings)


if __name__ == "__main__":
    # Ejemplo de uso: text_corpus es tu lista de textos
    word2vec_embeddings = generate_word2vec_embeddings(text_corpus)
    print("Word2Vec Embeddings:", word2vec_embeddings)
    print("Word2Vec Shape:", word2vec_embeddings.shape)

    # Ejemplo de uso
    bert_embeddings = generate_bert_embeddings(text_corpus)
    print("BERT Embeddings:", bert_embeddings)
    print("BERT Embeddings Shape:", bert_embeddings.shape)
