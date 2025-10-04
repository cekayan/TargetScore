
import os
import numpy as np
import pandas as pd
import pickle

def load_all_data():
    # ----- Load Core CSV Files -----
    # Load all_ts.csv and fill missing values with '#'
    all_ts = pd.read_csv('all_ts.csv', sep='\t', index_col=0).fillna('#')
    all_ts = all_ts.fillna('#')
    # Load bionetwork CSV; use the first column as index
    bionetwork = pd.read_csv('./WK_bionetwork.csv', sep=',')
    bionetwork.index = bionetwork['Unnamed: 0']
    
    # Load targetscores CSV and drop unwanted columns
    targetscores = pd.read_csv('Targetscores_v5_MeanImputated.csv', sep=',', index_col=0, low_memory=False)
    targetscores.drop(columns=['badp90rskpt359s363', 'cd496'], inplace=True)
    
    # Load responses data and drop extraneous columns
    resps = pd.read_csv('All_Resps_v5_sanitized.csv', sep='\t', index_col=0, low_memory=False)
    resps.drop(columns=['CL-Name', 'Set-No', 'Drug-Name', 'Drug-Time'], inplace=True)
    
    # Filter resps: remove columns with too many missing values
    thresh = resps.notna().sum().mean() / 2
    resps = resps.loc[:, resps.notna().sum() > thresh]
    
    # Keep only intersecting columns between targetscores and resps
    intersect_cols = sorted(set(targetscores.columns).intersection(resps.columns))
    targetscores = targetscores[intersect_cols]
    resps = resps[intersect_cols]
    
    # ----- Load CCLE and Genomic Data -----
    # Load CCLE data and process the sample_id to get cell line names
    ccle = pd.read_csv('TCPA_CCLE_RPPA500.tsv', sep='\t', low_memory=False)
    ccle['sample_id'] = ccle['sample_id'].apply(lambda x: x.split('_')[0])
    # Rename columns: first column as 'CL-Name', then remaining as-is
    ccle.columns = ['CL-Name'] + list(ccle.columns[1:])
    
    # Load genomics data and experiment dictionaries from pickles
    with open('genomics_data.pkl', 'rb') as f:
        genomics_data = pickle.load(f)
    with open('exp_data_v1.pkl', 'rb') as f:
        loaded_dict_list = pickle.load(f)
    
    # Load feature selection values from fs_korkut.csv
    fs_file = pd.read_csv('fs_korkut.csv', sep='\t', index_col=0)
    fs_list = [int(fs_file.loc[col, 'fs']) for col in targetscores.columns]
    
    # Unpack loaded_dict_list into mappings
    drug2target, dose_dict, dim_dict, time_dict = loaded_dict_list
    # Ensure default values for missing entries
    for d in (dose_dict, dim_dict, time_dict):
        d['#'] = 0
    
    our_drugs = drug2target.keys()
    our_CLs = ccle['CL-Name']
    
    # ----- Build Metadata from all_ts -----
    set_info = []
    cl_info = []
    drug_info = []
    time_info = []
    dim_info = []
    stimuli_info = []
    drug2_info = []
    dose_info = []
    for uid in targetscores.index:
        set_info.append(all_ts.loc[uid, 'set'])
        cl_info.append(all_ts.loc[uid, 'cell_line_name'])
        drug_info.append(all_ts.loc[uid, 'compound_name_1'])
        time_info.append(all_ts.loc[uid, 'time'])
        dim_info.append(all_ts.loc[uid, '2D_3D'])
        stimuli_info.append(all_ts.loc[uid, 'stimuli'])
        drug2_info.append(all_ts.loc[uid, 'compound_name_2'])
        dose_info.append(all_ts.loc[uid, 'dosage_1'])
    
    # Insert metadata columns into targetscores (at specified positions)
    targetscores.insert(0, 'SetNo', set_info)
    targetscores.insert(1, 'CL-Name', cl_info)
    targetscores.insert(2, 'Drug-Name', drug_info)
    targetscores.insert(3, 'Time', time_info)
    targetscores.insert(4, '2D-3D', dim_info)
    targetscores.insert(5, 'Stimuli', stimuli_info)
    targetscores.insert(6, 'Drug-2', drug2_info)
    targetscores.insert(7, 'Dose', dose_info)
    
    # ----- Filter Targetscores -----
    CL_filter = set(our_CLs).intersection(set(genomics_data.keys()))

    targetscores = targetscores.loc[targetscores['Drug-Name'].isin(our_drugs), :]
    targetscores = targetscores.loc[targetscores['CL-Name'].isin(CL_filter), :]
    targetscores = targetscores.loc[targetscores['Time'].isin(['24hr', '24hrs', '12hr', '12h', '12hrs', '48hrs', '48hr', '48h', '#', np.nan, '72hr', '4hr']), :]
    
    # Align responses with the filtered targetscores
    resps = resps.loc[targetscores.index, :]

    test_targetscores = targetscores.sample(frac=0.15)
    targetscores = targetscores.drop(test_targetscores.index)

    test_resps = resps.loc[test_targetscores.index, :]    
    resps = resps.loc[targetscores.index, :]

    test_na_mask = (np.isnan(test_resps.to_numpy()) == 0)
    
    # ----- Build Stimuli Mapping -----
    stimuli_types = {
        '10% serum': 1, '2% serum': 1, '2% serum (IFC) and 2% matrigel': 1, '2% serum (IFC) and 2% matrigel + EGF': 1,
        'FBS': 1, 'SERUM': 1, '2%MG 2%serum': 1,
        'EGF': 2, 'FGF': 2, 'FGF1': 2, 'HGF': 2, 'HGF1': 2, 'IGF1': 2, 'NRG': 2, 'NRG1': 2,
        'INSULIN': 3, 'Insulin': 3,
        'PBS': 4,
        'WIT EGF': 5, 'WIT-YZ': 5,
        'attached': 6, 'suspension': 6,
        'Foxo DBM 4HT ': 7, "No IL3": 7, 'Foxo 3xA EtOH': 7, 'with IL3': 7, 'Foxo 3xA 4HT ': 7, 'Foxo DBM EtOH ': 7,
        '4days+1d induction 4HT+2d drug treatment': 7,
        np.nan: 0, '#': 0,
    }
    stimuli_dict = dict(stimuli_types)    
    
    # ----- Process Baseline Features from CCLE -----
    # Duplicate CCLE rows based on the frequency of each cell line in targetscores
    ccle_data = pd.DataFrame(columns=ccle.columns)
    cl_counts = targetscores['CL-Name'].value_counts().to_dict()
    for cl in set(targetscores['CL-Name']):
        if cl in set(ccle['CL-Name']):
            num = cl_counts[cl]
            temp_df = ccle[ccle['CL-Name'] == cl]
            if temp_df.shape[0] == 1:
                duplicated = pd.concat([temp_df] * num, axis=0)
                ccle_data = pd.concat([ccle_data, duplicated], ignore_index=True)
    
    # ----- Load Additional Network Data -----
    # Load the bionet from WK_bionetwork_New.csv
    bionet_new = pd.read_csv('WK_bionetwork_New.csv', sep=',', index_col=0)
    
    # ----- Package All Data into a Dictionary -----
    data = {
        'all_ts': all_ts,
        'bionetwork': bionet_new,
        'targetscores': targetscores,
        'responses': resps,
        'ccle': ccle,
        'ccle_data': ccle_data,
        'dicts': [genomics_data, drug2target, dose_dict, dim_dict, time_dict, stimuli_dict],
        'fs_list': fs_list,
        'test_targetscores': test_targetscores,
        'test_responses': test_resps,
        'test_na_mask': test_na_mask
    }
    
    return data