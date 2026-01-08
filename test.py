import pandas as pd
from sklearn.metrics import root_mean_squared_error
import pickle
import yaml

def load_yaml(file_path: str) -> dict:
    """Load a yaml file and return it as a dictionary.

    Args:
        file_path (str): Path to the yaml file.

    Returns:
        dict: Yaml file as a dictionary.
    """
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def load_model(file_path: str) -> object:
    """Load a pickled model.

    Args:
        file_path (str): Path to the pickled model.

    Returns:
        object: Model object.
    """
    with open(file_path, 'rb') as file:
        return pickle.load(file)
    
def save_model(model: object, file_path: str) -> None:
    """Save a model as a pickle file.

    Args:
        model (object): Model object.
        file_path (str): Path to save the model.
    """
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)
    
def load_dataset(file_path: str) -> pd.DataFrame:
    """Load a dataset.

    Args:
        file_path (str): Path to the dataset.

    Returns:
        pd.DataFrame: Dataset.
    """
    return pd.read_csv(file_path)


if __name__=="__main__":
    config_data = load_yaml('config.yaml')
    test_dataset_path = config_data['test_dataset']
    model_path = config_data['artifacts']['model']
    
    # Optional
    scaler_path = config_data['artifacts']['scaler']
    
    # Load data and model
    test_df = load_dataset(test_dataset_path)
    model = load_model(model_path)

    # Optional
    scaler = load_model(scaler_path)

    # Create X and y
    X = test_df.drop(config_data["target"], axis=1)
    y = test_df[config_data["target"]].values
    
    # Preprocess data (if needed)
    # 1st: Pandas preprocessing
    # 2nd: Sklearn transformations


    # Make predictions
    predictions = None # Add code here to make predictions using the model

    # Calculate metrics (tantas como necesites)
    rmse = root_mean_squared_error(y, predictions)
    print(f"RMSE: {rmse}")