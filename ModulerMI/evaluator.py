"""
evaluator.py
Contains functions for evaluating the model performance and plotting results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import r2_score
from metrics import classification_metric, multivariate_r2

def evaluate_model(model_type, trained_model, data_dict, pca_list, print_check=False):

    genomics_data, drug2target, dose_dict, dim_dict, time_dict, stimuli_dict = data_dict['dicts']

    test_targetscores = data_dict['test_targetscores']
    ccle = data_dict['ccle']

    test_ccle_data = pd.DataFrame(columns = ccle.columns)

    order_test_ts = test_targetscores['CL-Name'].value_counts().to_dict()

    count = 0

    for cl_name in set(test_targetscores['CL-Name']):
        
        if cl_name in set(ccle['CL-Name']):

            num = order_test_ts[cl_name]
            temp_df = ccle.loc[ccle['CL-Name']==cl_name,:]  

            if temp_df.shape[0] == 1:
                duplicated_rows = pd.concat([temp_df] * (num), axis=0)
                test_ccle_data = pd.concat([test_ccle_data, duplicated_rows], ignore_index=True)

                count += num

    drug_vecs = np.array([drug2target[drug] for drug in test_targetscores['Drug-Name']])
    stimuli_vecs = np.array([stimuli_dict[sti] for sti in test_targetscores['Stimuli']])
    stimuli_vecs = np.reshape(stimuli_vecs, (stimuli_vecs.shape[0],1))
    time_vecs = np.array([time_dict[time] for time in test_targetscores['Time']])
    time_vecs = np.reshape(time_vecs, (time_vecs.shape[0],1))
    dose_vecs = np.array([dose_dict[dose] for dose in test_targetscores['Dose']])
    dose_vecs = np.reshape(dose_vecs, (dose_vecs.shape[0],1))
    dim_vecs = np.array([dim_dict[dim] for dim in test_targetscores['2D-3D']])
    dim_vecs = np.reshape(dim_vecs, (dim_vecs.shape[0],1))
    dim_vecs = dim_vecs.astype(np.float32)
    vectors_cna = np.array([genomics_data[key]['CNA'] for key in test_targetscores['CL-Name']])
    vectors_mexp = np.array([genomics_data[key]['mRNA'] for key in test_targetscores['CL-Name']])
    vectors_mut = [[item for item in genomics_data[key]['Mutation'].to_numpy()] for key in test_targetscores['CL-Name']]
    mut_vec_1 = np.array(vectors_mut)[:,:,0]
    mut_vec_2 = np.array(vectors_mut)[:,:,1]

    test_baselines = test_ccle_data.iloc[:,1:].to_numpy()
    test_labels = test_targetscores.iloc[:,8:].to_numpy()

    test_data_list = [drug_vecs,
                      time_vecs, dose_vecs, dim_vecs,
                      pca_list[0].transform(vectors_cna),
                      vectors_mexp,
                      pca_list[1].transform(mut_vec_1),
                      pca_list[2].transform(mut_vec_2)]

    if model_type == 'nn':
        test_data_list.append(test_baselines)

    if model_type == 'c-ml':
        test_data_list = np.concatenate(test_data_list, axis=1)

    test_data_pred = trained_model.predict(test_data_list)

    non_NA_mask_test = data_dict['test_na_mask']
    test_data_pred = test_data_pred[non_NA_mask_test]
    test_labels = test_labels[non_NA_mask_test]

    if print_check == True:
        print("\nTest Data Results")
        print("Correlation:",np.corrcoef(test_data_pred.flatten(), test_labels.flatten())[0, 1])
        print("R² Score:",multivariate_r2(y_pred=test_data_pred.flatten(), y_true=test_labels.flatten()))
        print("Accuracy:",classification_metric(predicted=test_data_pred.flatten(),
                                    true=test_labels,
                                    bins=[-np.inf, -1, -0.25,  0.25, 1, np.inf]))

        print('\n')
        print("#Organic Data Points (Test):", np.sum(non_NA_mask_test))
    return(test_targetscores.shape[0], np.corrcoef(test_data_pred.flatten(), test_labels.flatten())[0, 1])