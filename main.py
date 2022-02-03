import os
import tempfile
import numpy as np
import tensorflow as tf
import Data_Handler as DH
import tensorflow_recommenders as tfrs
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.optimizers import Adagrad
from Model_Class import ClassProductModel as CPM
 



paths = {
    'Data' : os.path.join(os.getcwd(),'Data'),
    'Notebooks' : os.path.join(os.getcwd(),'notebooks'),
    'models' : os.path.join(os.getcwd(),'models')
}

# Data Handling

[train_user_id, train_products_id], [test_user_id, test_products_id], unique_user_id, unique_product_id, user_id, product_id = DH.return_data_sets()


# Query Model
embedding_dimension = 32

user_model = Sequential(layers=[
    Embedding(len(unique_user_id) + 1,
              embedding_dimension),
    Embedding(embedding_dimension, 
              embedding_dimension / 2)
])

# Candidate Model


product_model = Sequential(layers=[
    Embedding(len(unique_user_id) + 1,
              embedding_dimension),
    Embedding(embedding_dimension, 
              embedding_dimension / 2)
])


product_data = tf.data.Dataset.from_tensors(product_id)

metrics = tfrs.metrics.FactorizedTopK(
    candidates=product_data.batch(64).map(product_model)
)

task = tfrs.tasks.Retrieval(
    metrics=metrics
)

# Full Model

model = CPM(
    task=task,
    user_model=user_model,
    product_model=product_model
)


model.compile(
    optimizer = Adagrad(learning_rate=0.1)
)

train = {
    'user_id': list(train_user_id),
    'product_id': list(train_products_id),
}

test = {
    'user_id': list(test_user_id),
    'product_id': list(test_products_id),
}

cached_train = tf.data.Dataset.from_tensors(np.array(train)).shuffle(100_000).batch(512).cache()
cached_test = tf.data.Dataset.from_tensors(np.array(test)).batch(512).cache()

model.fit(
    cached_train,
    epochs= 3
    )

model.evaluate(
    cached_test, 
    return_dict=True
    )

with tempfile.TemporaryDirectory() as tmp:
    pass