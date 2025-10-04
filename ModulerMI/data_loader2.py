# data_loader.py
import pandas as pd
import pickle

def load_data():
    # Load CSV files
    all_ts = pd.read_csv('all_ts.csv', sep='\t', index_col=0)
    resps = pd.read_csv('All_Resps_v5.csv', sep='\t', index_col=0)
    targetscores = pd.read_csv('Targetscores_v5.csv', sep='\t', index_col=0)
    ccle = pd.read_csv('TCPA_CCLE_RPPA500.tsv', sep='\t', low_memory=False)
    bionetwork = pd.read_csv('./WK_bionetwork.csv', sep=',')

    # Load pickle files
    with open('genomics_data.pkl', 'rb') as f:
        genomics_data = pickle.load(f)
        
    with open('exp_data_v1.pkl', 'rb') as f:
        exp_data = pickle.load(f)  # Contains drug2target, dose_dict, dim_dict, time_dict

    return all_ts, resps, targetscores, ccle, bionetwork, genomics_data, exp_data