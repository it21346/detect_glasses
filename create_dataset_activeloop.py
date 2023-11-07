import deeplake 
import os 


def create_dataset():
    # Activeloop token
    os.environ['ACTIVELOOP_TOKEN'] = 'ACTIVELOOP TOKEN'


    # Directory of the dataset
    data_directory = 'Directory of the Dataset images'
    # Github repo URL
    deeplake_path = 'hub://it21346/glasses_or_not_dataset'

    deeplake.ingest_classification(data_directory, deeplake_path)



if __name__ == "__main__":
    create_dataset()