import os
import pandas as pd


paths = {
    'Data' : os.path.join(os.getcwd(),'Data'),
    'Notebooks' : os.path.join(os.getcwd(),'notebooks'),
    'models' : os.path.join(os.getcwd(),'models')
}


df = pd.read_csv(paths['Data']+"/amazon_co_ecommerce_sample.csv")

users = pd.DataFrame({'user': df[:]['manufacturer'].unique()})	
users['user_id'] = [i+1 for i in range(users.shape[0])]


products = pd.DataFrame({'product': df[:]['product_name'].unique()})
products['product_id'] = [i+1 for i in range(products.shape[0])]


new_df = df.set_index('manufacturer').join(users.set_index('user')).set_index('product_name').join(products.set_index('product'))


new_df = new_df.sample(frac = 1)

user_id = new_df[:]['user_id'].to_numpy()
products_id = new_df[:]['product_id'].to_numpy()

train_user_id, test_user_id = user_id[:8000], user_id[8000:] 
train_products_id, test_products_id = products_id[:8000], products_id[8000:]

unique_user_id = users[:]['user_id'].to_numpy()
unique_product_id = products[:]['product_id'].to_numpy()


def return_data_sets(
    train_sets : list = [train_user_id, train_products_id],
    test_sets : list = [test_user_id, test_products_id],
    unique_u_id = unique_user_id,
    unique_p_id = unique_product_id,
    product_id = products_id,
    user_id = user_id
                    ):
    return [train_sets, test_sets, unique_u_id, unique_p_id, user_id, product_id]