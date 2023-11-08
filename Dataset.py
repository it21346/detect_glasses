import deeplake 
import os 
import argparse
from deeplake.util.exceptions import TransformError, CheckoutError
import numpy as np
from PIL import Image

class G_Dataset:
    def __init__(self, token = None, Activeloop_repo = None, resize_size = (224,224)):
        self.token = token
        self.ds = None
        self.Activeloop_repo = Activeloop_repo
        self.resize_size = resize_size

    def create_dataset(self, dataset_dir, create_dataset = False):
        if create_dataset:
            # Activeloop token
            os.environ['ACTIVELOOP_TOKEN'] = self.token
            # Directory of the dataset (./Images/Images)
            data_directory = dataset_dir
            # Activeloop repo URL
            deeplake_path = self.Activeloop_repo
            deeplake.ingest_classification(data_directory, deeplake_path)

    def load_dataset(self):
        """
        Load the dataset from ActiveLoop
        """
        self.ds = deeplake.load(self.repo, token = self.token)

    
    def resize_dataset_activeloop(self):
        """
        There is a need to resize the original images of the dataset, which are (1024, 1024, 3) size, to an appropriate size for feeding our later on NN model.
        """
        try: #if the resized dataset branch already exists, do nothing and move on. Otherwise, create a resized dataset branch.
            self.ds = deeplake.load(f'{self.Activeloop_repo}@resized/dataset', token = self.token)
        except CheckoutError as e:

            @deeplake.compute
            def resize(sample_in, sample_out, new_size):
                
                sample_out.labels.append(sample_in.labels.numpy())
                sample_out.images.append(np.array(Image.fromarray(sample_in.images.numpy()).resize(new_size)))
                
                return sample_out

            try:
                resize(new_size = self.resize_size).eval(self.ds, num_workers = 4)
            except TransformError as e:
                failed_idx = e.index
                failed_sample = e.sample
                print(failed_idx)

            
            self.ds.commit(f"Resize Dataset to {self.resize_size}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create the dataset into Activeloop")
    #Arguments
    
    parser.add_argument('--token', help="ActiveLoop API Token")
    parser.add_argument('--dir', help= "Dataset directory. (./Images/Images)")
    parser.add_argument('--repo', help= "ActiveLoop repository URL")
    parser.add_argument('--create_dataset', default = False, help= "Flag to indicate whether to create the dataset initially or not. 'True' to create the dataset.")
    parser.add_argument('--resize_size', help= "Resize size as tuple (x,y) for the dataset resizing function")
    args = parser.parse_args()

    token = args.token
    dir = args.dir
    repo = args.repo
    resize_size = args.resize_size
    create_dataset = args.create_dataset

    #check if Token/Repo have been given as args
    if not token or not repo:
        print("Please provide either token or repo URL as arguments!")
    #check if flag has been given but not the directory of the dataset
    elif not dir:
        print("You need to also provide a path to the dataset!")
    else:
        if resize_size:
            cl = G_Dataset(token, repo, resize_size = resize_size)
        
        cl = G_Dataset(token, repo)
        #create the dataset if create_dataset was True
        cl.create_dataset(dir, create_dataset)
        cl.load_dataset()
        cl.resize_dataset_activeloop()