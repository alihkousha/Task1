import pandas as pd
import os
import tensorflow as tf
import numpy as np

paths = {
    'Data' : os.path.join(os.getcwd(),'Data'),
    'Notebooks' : os.path.join(os.getcwd(),'notebooks'),
    'models' : os.path.join(os.getcwd(),'models')
}

df = pd.read_csv(os.path.join(paths['Data'],'amazon_co_ecommerce_sample.csv'))


users = pd.DataFrame({'user':df.manufacturer.unique()})
users['user_id'] = [i+1 for i in range(users.shape[0])]


pro = pd.DataFrame({'product':df.product_name.unique()})
pro['product_id'] = [i+1 for i in range(pro.shape[0])]

new_df = df.set_index('manufacturer').join(users.set_index('user')).set_index('product_name').join(pro.set_index('product'))

products = list(new_df.index)

products = tf.convert_to_tensor(products)

products = tf.data.Dataset.from_tensor_slices(products)

sells = tf.data.Dataset.from_tensor_slices(
    (
        tf.cast(new_df.index.values,tf.string),
        tf.cast(new_df.user_id.values,tf.int64)
    )
)

sells = sells.map(lambda x: {
    "product_title" : x[0],
    "user_id": x[1],
})