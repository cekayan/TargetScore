"""
trainer.py
Contains the training loop with:
1) An 85%-15% train-test split.
2) 5-Fold CV on the training set.
3) Reporting of the average CV correlation (both train and validation) and test set performance.
"""

import numpy as np
import random
from sklearn.model_selection import KFold, train_test_split
from sklearn.decomposition import PCA
import xgboost as xgb
from plotting import classification_metric
from neural_models import CustomTSModel, CustomAttentionModel, WeightedAverageEnsemble
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.optimizers import Adam

def run_training(data, model_type, model_name):

    feature_names = [
        "drug_vecs", "time_vecs", "dose_vecs", "dim_vecs",
        "vectors_cna", "vectors_mexp", "mut_vec_1", "mut_vec_2",
        "baselines"
    ]

    train_data = {}
    test_data = {}

    for feature_name in feature_names + ['labels']:
        train_f, test_f = train_test_split(data[feature_name], test_size=0.15, random_state=42)
        train_data[feature_name] = train_f
        if feature_name != 'labels':
            test_data[feature_name] = test_f
        if feature_name == 'labels':
            y_test = test_f

    train_features = []
    val_features = []
    pcas = []
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_train_corr = []
    cv_val_corr = []
    fold = 0
    for train_idx, val_idx in kf.split(train_f):

        for feature_name in feature_names:
            if feature_name in ["vectors_cna", "mut_vec_1", "mut_vec_2"]:
                pca = PCA(n_components=10)
                train_feature = pca.fit_transform(train_data[feature_name][train_idx])
                val_feature = pca.transform(train_data[feature_name][val_idx])
                pcas.append(pca)
            
            else:
                train_feature = train_data[feature_name][train_idx]
                val_feature = train_data[feature_name][val_idx]
        
            train_features.append(train_feature)
            val_features.append(val_feature)  

        if model_type == 'c-ml':
            train_features = np.concatenate(train_features, axis=1)
            val_features = np.concatenate(val_features, axis=1)
        
        X_tr, X_val = train_features, val_features
        y_tr, y_val = train_data['labels'][train_idx], train_data['labels'][val_idx]

        print(X_tr.shape)
        
        # Instantiate an XGBoost regressor (you can adjust hyperparameters as needed)
        if model_name == 'xgb':
            model_cv = xgb.XGBRegressor(
            objective='reg:squarederror',
            max_depth=3,
            learning_rate=0.3,
            n_estimators=30,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
            )
        elif model_name == 'rf':
            model_cv = RandomForestRegressor(n_estimators=30, criterion='squared_error', max_depth=3)
        elif model_name == 'ensemble':
            model_xgb = xgb.XGBRegressor(
            objective='reg:squarederror',
            max_depth=3,
            learning_rate=0.3,
            n_estimators=30,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
            )
            model_rf = RandomForestRegressor(n_estimators=30, criterion='squared_error', max_depth=3)
            models = [('xgb', model_xgb), ('rf', model_rf)]
            model_cv = WeightedAverageEnsemble(models=models, avg_check=False)
        elif model_name == 'tsnn':
            model_cv = CustomTSModel(nprots=y_tr.shape[1], num_categories=7, embedding_dim=10, fs_list=data['fs_list'], bionet=data['bionet_new'])
        elif model_name == 'attention':
            model_cv = CustomAttentionModel(nprots=y_tr.shape[1], num_categories=7, embedding_dim=10)

        if model_type == 'c-ml':
            model_cv.fit(X_tr, y_tr)
        elif model_type == 'nn':
            model_cv.compile(optimizer=Adam(learning_rate=1e-3), loss='mse') 
            history = model_cv.fit(X_tr, y_tr,
                epochs=100,
                batch_size=32,
                shuffle=True,
                validation_data=(X_val, y_val),
                verbose=0)
        
        y_pred_tr = model_cv.predict(X_tr)
        y_pred_val = model_cv.predict(X_val)
        
        train_corr = np.corrcoef(y_pred_tr.flatten(), y_tr.flatten())[0, 1]
        val_corr = np.corrcoef(y_pred_val.flatten(), y_val.flatten())[0, 1]
        cv_train_corr.append(train_corr)
        cv_val_corr.append(val_corr)
        
        print(f"Fold {fold}: Train Corr = {train_corr:.2f}, Val Corr = {val_corr:.2f}")
        break
    
    avg_cv_train_corr = np.mean(cv_train_corr)
    avg_cv_val_corr = np.mean(cv_val_corr)
    
    print(f"\nAverage CV Train Correlation: {avg_cv_train_corr:.2f}")
    print(f"Average CV Validation Correlation: {avg_cv_val_corr:.2f}")
    
    # --- Train Final Model on Full Training Set ---
    X_test = [test_data["drug_vecs"],
                 test_data["time_vecs"],
                 test_data["dose_vecs"],
                 test_data["dim_vecs"],
                 pcas[0].transform(test_data["vectors_cna"]),
                 test_data["vectors_mexp"],
                 pcas[1].transform(test_data["mut_vec_1"]),
                 pcas[2].transform(test_data["mut_vec_2"]),
                 test_data["baselines"]
                 ]
    
    if model_type == 'c-ml':
        X_test = np.concatenate(X_test, axis=1)
    
    print(X_test.shape, X_tr.shape)
    final_model = model_cv
    y_pred_test = final_model.predict(X_test)

    test_corr = np.corrcoef(y_pred_test.flatten(), y_test.flatten())[0, 1]
    
    print(f"\nTest Set Correlation: {test_corr:.2f}")
    
    return final_model, y_pred_test, y_test, avg_cv_train_corr, avg_cv_val_corr


import numpy as np
import pandas as pd
import random
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from metrics import multivariate_r2
from model_stuff import model_shit

def training_machine(dict_list, model_type, model_name):

    targetscores = dict_list["targetscores"]
    ccle_data = dict_list['ccle_data']

    genomics_data, drug2target, dose_dict, dim_dict, time_dict, stimuli_dict = dict_list['dicts']

    baselines = ccle_data.iloc[:,1:].to_numpy()
    labels = targetscores.iloc[:,8:].to_numpy()

    drug_vecs = np.array([drug2target[drug] for drug in targetscores['Drug-Name']])

    stimuli_vecs = np.array([stimuli_dict[sti] for sti in targetscores['Stimuli']])
    stimuli_vecs = np.reshape(stimuli_vecs, (stimuli_vecs.shape[0],1))
    stimuli_vecs = stimuli_vecs.astype(np.float32)

    time_vecs = np.array([time_dict[time] for time in targetscores['Time']])
    time_vecs = np.reshape(time_vecs, (time_vecs.shape[0],1))
    time_vecs = time_vecs.astype(np.float32)

    dose_vecs = np.array([dose_dict[dose] for dose in targetscores['Dose']])
    dose_vecs = np.reshape(dose_vecs, (dose_vecs.shape[0],1))
    dose_vecs = dose_vecs.astype(np.float32)

    dim_vecs = np.array([dim_dict[dim] for dim in targetscores['2D-3D']])
    dim_vecs = np.reshape(dim_vecs, (dim_vecs.shape[0],1))
    dim_vecs = dim_vecs.astype(np.float32)

    vectors_cna = np.array([genomics_data[key]['CNA'] for key in targetscores['CL-Name']])
    vectors_mexp = np.array([genomics_data[key]['mRNA'] for key in targetscores['CL-Name']])

    vectors_mut = [[item for item in genomics_data[key]['Mutation'].to_numpy()] for key in targetscores['CL-Name']]

    mut_vec_1 = np.array(vectors_mut)[:,:,0]
    mut_vec_2 = np.array(vectors_mut)[:,:,1]

    feature_list = [drug_vecs, time_vecs, dose_vecs, dim_vecs, vectors_cna, vectors_mexp, mut_vec_1, mut_vec_2]
    feature_name_list = ['drug', 'time', 'dose', 'dim', 'cna', 'mrna', 'hotspot', 'mut_type']
    feature_dict = {}
    for item1, item2 in zip(feature_name_list, feature_list):
        feature_dict[item1] = item2

    if model_type == 'nn':
        feature_dict['baseline'] = baselines

    trained_model, pcas = model_shit(data=dict_list, model_type=model_type, model_name=model_name, features=feature_dict, labels=labels)

    return trained_model, pcas