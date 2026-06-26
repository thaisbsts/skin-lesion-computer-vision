import tensorflow as tf
import numpy as np

SEED = 42 # Semente para tornar os resultados reproduzíveis
IMG_SIZE = 224 # Tamanho para o qual todas as imagens serão redimensionadas (ideal para o modelo) 
BATCH_SIZE = 32 # Tamanho do lote que o modelo usará para redefinir os pesos
EPOCHS = 15 # Número de vezes que o modelo passará por todas as imagens
FINE_TUNE_EPOCHS = 10 # Quantidade de épocas da fase de fine-tuning (parte do modelo é descongelada para ajustar seus pesos ao dataset) 

tf.random.set_seed(SEED)
np.random.seed(SEED)