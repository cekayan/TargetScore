from data_loader import load_all_data
from trainer import training_machine
from evaluator import evaluate_model

def main():
    data_dict = load_all_data()

    model_type, model_name = 'c-ml', 'xgb'

    trained_model, pcas = training_machine(data_dict, model_type=model_type,  model_name=model_name)
    
    evaluate_model(model_type=model_type, trained_model=trained_model, data_dict=data_dict, pca_list=pcas, print_check=True)

if __name__ == '__main__':
    main()
