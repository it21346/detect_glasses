import deeplake 
import os 
import argparse

def create_dataset(token, dataset_dir, Activeloop_repo):
    # Activeloop token
    os.environ['ACTIVELOOP_TOKEN'] = token


    # Directory of the dataset (./Images/Images)
    data_directory = dataset_dir
    # Activeloop repo URL
    deeplake_path = Activeloop_repo

    deeplake.ingest_classification(data_directory, deeplake_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create the dataset into Activeloop")
    #Arguments

    parser.add_argument('--token', help="ActiveLoop API Token")
    parser.add_argument('--dir', help= "Dataset directory. (./Images/Images)")
    parser.add_argument('--repo', help= "ActiveLoop repository URL")

    args = parser.parse_args()

    token = args.token
    dir = args.dir
    repo = args.repo
    create_dataset(token, dir, repo)