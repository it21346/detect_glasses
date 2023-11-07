import deeplake 
import os 


def create_dataset(token, dataset_dir, Activeloop_repo):
    # Activeloop token
    os.environ['ACTIVELOOP_TOKEN'] = token


    # Directory of the dataset
    data_directory = dataset_dir
    # Activeloop repo URL
    deeplake_path = Activeloop_repo

    deeplake.ingest_classification(data_directory, deeplake_path)
