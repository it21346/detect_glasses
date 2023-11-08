import deeplake 
import os 
import argparse
from deeplake.util.exceptions import TransformError, CheckoutError
import numpy as np
from PIL import Image

class G_Dataset:
    def __init__(self, token, Activeloop_repo = None, resize_size = (224,224)):
        self.token = token
        self.ds = None
        self.Activeloop_repo = Activeloop_repo
        self.resize_size = resize_size

    def create_dataset(self, dataset_dir):
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
        self.ds = deeplake.load('hub://it21346/glasses_or_not_dataset', token = self.token)

    
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
    print("This code will run when the script is executed directly.")
    # parser = argparse.ArgumentParser(description="Create the dataset into Activeloop")
    # #Arguments
    
    # parser.add_argument('--token', help="ActiveLoop API Token")
    # parser.add_argument('--dir', help= "Dataset directory. (./Images/Images)")
    # parser.add_argument('--repo', help= "ActiveLoop repository URL")

    # args = parser.parse_args()

    # token = args.token
    # dir = args.dir
    # repo = args.repo
    # cl = G_Dataset()
    # print(dir)
    # # create_dataset(token, dir, repo)
