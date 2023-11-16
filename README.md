# detect_glasses

## Introduction
This is a side project to detect the presence of glasses on human faces. Using technologies like Convolutional Neural Networks (CNNs), Activeloop Deeplake, Amazon Web Services, to produce an end-to-end pipeline for creating an activeloop dataset, creating and training a CNN model, and finally providing inference capabilities for new test images through AWS API Gateway and Docker.
## Dataset
For this project, the *Glasses or No Glasses* dataset was used from Kaggle, which is a Generative Adversarial Neural Network (GAN) produced dataset, which means these are not real people. Since the original dataset contained a noticeable amount of wrongly labeled images, the community developed a cleaned up version with seperate folders for each class.

## Features
In this project, you can create an Activeloop Deeplake dataset repository, create and train a CNN model (specifically the **MobileNetv3Small** version), and provide inference capabilities to the trained model through AWS and Docker. Each of these steps can be run as a seperate script.

## Getting Started
### Prerequisites and Installation
In order to run this project, you need to install the appropriate prerequisites.

On windows from any CMD:
```python
#Note that you should have already python installed in your system
#Create a python virtual environment named venv
python -m venv venv
#Activate the virtual environment
venv\Scripts\activate
#Install all the dependencies from the path of the requirements.txt file
python -m pip install -r /path/to/requirements.txt
```

On Unix or MacOS:
```python
#Note that you should have already python installed in your system
python -m venv venv
#Activate the virtual environment
source venv/bin/activate
#Install all the dependencies from the path of the requirements.txt file
pip install -r requirements.txt
```

### Usage
 - If you want to create your own Activeloop dataset repository, you need to download the dataset provided in this link https://www.kaggle.com/datasets/jorgebuenoperez/datacleaningglassesnoglasses and run the following script. Otherwise, my Activeloop dataset repository will be used instead. 
`python -m Dataset.py --token 'YOUR_API_TOKEN --dir 'YOUR_DATASET_DIRECTORY' --repo 'YOUR_REPO_URL'` 

    Arguments:
    - **--token** : you need to provide an API token (obtained from Activeloop when you create a profile)
    - **--dir** : the path to your dataset directory and you need to provide it in this level (*./Images/Images*)
    - **--repo** : your ActiveLoop repository URL (*hub://<your_org_name>/<name_of_your_dataset>*)
    - **(Optional) --create_dataset** : is default to False and won't create the dataset, but you can load it (In the case you have already created one and you just want to load)
    - **(Optional) --resize_size** : is referring to the image size which is default to (224,224), and if you want to specify a new one you should as tuple (x,y) for the dataset resizing function
  
 - If you want to train the MobileNetv3Small from scratch, you can do it, with either the Activeloop dataset created or not (You will by default use mine).
  `python -m MobileNetv3Small.py --modelname <YOUR_PREF_MODELNAME.pkl>`
  Arguments:
    - **--modelname** : Name for the model. If argument --train = True is given, the trained model will be named likewise. \
    If not intending training, this modelname will be used to load the corresponding model. \
    Note: Also use .pkl in the end. Refers to any model saved at ./models/*.pkl
    - **(Optional)--activation_func** : Activation function for the model. Default is 'sigmoid'
    - **(Optional) --loss** : Model Loss. Default is 'binary_crossentropy'
    - **(Optional) --lr** : Learning rate. Default is '0.001'
    - **(Optional) --epochs** : Epochs for the model to train. Default is '5'
    - **(Optional) --train** : Flag to indicate whether to train the model or not. 'True' to train the model. Default is 'False'

    **NOTE:** This model's computational capabilities are not big and can be run on CPU, this it has not been optimized for GPU usage.  



