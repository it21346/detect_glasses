import deeplake 
import os
from deeplake.util.exceptions import TransformError, CheckoutError
import numpy as np
from PIL import Image

# SET/EXPORT your Activeloop token
api_token = os.environ.get("MY_ACTIVELOOP_API_TOKEN")
resize_size = (224,224)

def load_dataset():
    """
    Load the dataset from ActiveLoop
    """
    ds = deeplake.load('hub://it21346/glasses_or_not_dataset', token = api_token)
    return ds


def resize_dataset_activeloop(ds, resize_size):
    """
    There is a need to resize the original images of the dataset, which are (1024, 1024, 3) size, to an appropriate size for feeding our later on NN model.
    """
    try: #if the resized dataset branch already exists, do nothing and move on. Otherwise, create a resized dataset branch.
        ds = deeplake.load('hub://it21346/glasses_or_not_dataset@resized/dataset', token = api_token)
    except CheckoutError as e:

        

        @deeplake.compute
        def resize(sample_in, sample_out, new_size):
            
            sample_out.labels.append(sample_in.labels.numpy())
            sample_out.images.append(np.array(Image.fromarray(sample_in.images.numpy()).resize(new_size)))
            
            return sample_out

        try:
            resize(new_size = resize_size).eval(ds, num_workers = 4)
        except TransformError as e:
            failed_idx = e.index
            failed_sample = e.sample
            print(failed_idx)

        
        ds.commit(f"Resize Dataset to {resize_size}")

    return ds

if __name__ == "__main__":
    ds = load_dataset()
    ds = resize_dataset_activeloop(ds)
    if ds.branch != 'main':
        #load a specific branch, like the resized one
        ds = deeplake.load(f'hub://it21346/glasses_or_not_dataset@{ds.branch}', token = api_token)