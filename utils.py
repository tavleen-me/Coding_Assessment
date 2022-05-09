# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 14:54:53 2021

@author: Patrick
"""

def open_pickle(path_in, file_name):
    import pickle
    tmp = pickle.load(open(path_in + file_name, "rb"))
    return tmp

def write_pickle(path_in, file_name, var_in):
    import pickle
    pickle.dump(var_in, open(path_in + file_name, "wb"))

def vec_fun(df_in, path_in):
    from sklearn.feature_extraction.text import CountVectorizer
    import pandas as pd
    my_vec = CountVectorizer()
    my_vec_text = pd.DataFrame(my_vec.fit_transform(df_in).toarray())
    my_vec_text.columns = my_vec.get_feature_names() 
    write_pickle(path_in, "vec.pkl", my_vec)
    return my_vec_text

def perf_metrics(model_in, x_in, y_true):
    from sklearn.metrics import precision_recall_fscore_support
    y_pred = model_in.predict(x_in)
    metrics = precision_recall_fscore_support(
        y_true, y_pred, average='weighted')
    return metrics

def my_rf(x_in, y_in, out_in):
    from sklearn.ensemble import RandomForestClassifier
    my_rf_m = RandomForestClassifier(random_state = 1,
                                  n_estimators = 750,
                                  max_depth = 15, 
                                  min_samples_split = 5,  
                                  min_samples_leaf = 1)
    my_rf_m.fit(x_in, y_in)
    write_pickle(out_in, "rf.pkl", my_rf_m)
    return my_rf_m

def split_data(x_in, y_in, split_fraction):
    from sklearn.model_selection import train_test_split
    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
        x_in, y_in, test_size=(1.0 - split_fraction), random_state=42)
    return X_train_t, X_test_t, y_train_t, y_test_t

def my_pca(df_in, n_conp_in, path_in):
    from sklearn.decomposition import PCA
    pca_m = PCA(n_components = n_conp_in)
    pca_data_t = pca_m.fit_transform(df_in)
    write_pickle(path_in, "pca.pkl", pca_m)
    return pca_data_t