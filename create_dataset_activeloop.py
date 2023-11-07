import deeplake 
import os 

# Activeloop token
os.environ['ACTIVELOOP_TOKEN'] = 'ACTIVELOOP TOKEN'


# Directory of the dataset
data_directory = 'Directory of the Dataset images'
# Github repo URL
deeplake_path = 'hub://it21346/glasses_or_not_dataset'

ds = deeplake.ingest_classification(data_directory, deeplake_path)