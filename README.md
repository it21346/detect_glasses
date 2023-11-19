# detect_glasses

## Introduction
This is a side project to detect the presence of glasses on human faces. Using technologies like Convolutional Neural Networks (CNNs) for binary classification (*0 : No glasses, 1: Glasses*), Activeloop Deeplake, Amazon Web Services, to produce a pipeline for creating an activeloop dataset, creating and training a CNN model, and finally providing inference capabilities for new test images through AWS and Docker.
## Dataset
For this project, the *Glasses or No Glasses* dataset was used from Kaggle, which is a Generative Adversarial Neural Network (GAN) produced dataset, which means these are not real people. Since the original dataset contained a noticeable amount of wrongly labeled images, the community developed a cleaned up version with seperate folders for each class. In addition, the dataset consists of profile angle pictures of single people.

## Features
In this project, you can create an Activeloop Deeplake dataset repository, create and train a CNN model (specifically the **MobileNetv3Small** version), and provide inference capabilities to the trained model through AWS and Docker. Each of these steps can be run as a seperate script.

## Getting Started
### Prerequisites and Installation
In order to run this project, you need to install the appropriate prerequisites. Note that you should have already python installed in your system.

On windows from any CMD:
```python
#Create a python virtual environment named venv
python -m venv venv
#Activate the virtual environment
venv\Scripts\activate
#Install all the dependencies from the path of the requirements.txt file
python -m pip install -r requirements.txt
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
 - If you want to create your own Activeloop dataset repository, you need to download the dataset provided in this link https://www.kaggle.com/datasets/jorgebuenoperez/datacleaningglassesnoglasses and run the following script. 
`python -m Dataset.py --token 'YOUR_API_TOKEN --dir 'YOUR_DATASET_DIRECTORY' --repo 'YOUR_REPO_URL'` 

    Arguments:
    - **--token** : you need to provide an API token (obtained from Activeloop when you create a profile)
    - **--dir** : the path to your dataset directory and you need to provide it in this level (*./Images/Images*)
    - **--repo** : your ActiveLoop repository URL (*hub://<your_org_name>/<name_of_your_dataset>*)
    - **(Optional) --create_dataset** : is default to False and won't create the dataset, but you can load it (In the case you have already created one and you just want to load)
    - **(Optional) --resize_size** : is referring to the image size which is default to (224,224), and if you want to specify a new one you should as tuple (x,y) for the dataset resizing function
  
 - If you want to train the MobileNetv3Small from scratch, you need to first have created an Activeloop dataset repository and have a token and that repository URL.
  `python -m MobileNetv3Small.py --modelname <YOUR_PREF_MODELNAME.pkl> --`
  Arguments:
    - **--modelname** : Name for the model. If argument --train = True is given, the trained model will be named likewise. \
    If not intending training, this modelname will be used to load the corresponding model. \
    Note: Also use .pkl in the end. Refers to any model saved at ./models/*.pkl
    - **(Optional)--activation_func** : Activation function for the model. Default is 'sigmoid'
    - **(Optional) --loss** : Model Loss. Default is 'binary_crossentropy'
    - **(Optional) --lr** : Learning rate. Default is '0.001'
    - **(Optional) --epochs** : Epochs for the model to train. Default is '5'
    - **(Optional) --train** : Flag to indicate whether to train the model or not. 'True' to train the model. Default is 'False'
    - **(Optional) --repo_dataset** : ActiveLoop API Token. Default is None
    - **(Optional) --token** : ActiveLoop dataset repository URL. For examply, mine is 'hub://it21346/glasses_or_not_dataset'. Default is None

    **NOTE:** This model's computational capabilities are not big and can be run on CPU, this it has not been optimized for GPU usage.  


## Inference
From this point on, to make inferences on the model and make predictions based on image inputs, my already set up AWS, Docker and trained model will be used. In this project, there are files that I used to set up the Docker image and configure the Lambda Function. These files are [Dockerfile](Dockerfile), [lambda_function.py](lambda_function.py), [requirements_docker.txt](requirements_docker.txt). The model I had trained, using the steps mentioned earlier, was saved into an AWS S3 bucket. I created an AWS lambda function, an AWS API Gateway and an AWS Elastic Container Registry (ECR). I uploaded my Docker image into the AWS ECR, and used it in my AWS lambda function, which in turn loads up my model from the AWS S3 bucket and make inferences based on inputs from the http requests in the API Gateway that triggers the Lambda function.
### Usage
Preferably use a profile angle picture of a single person.
`python -m main.py --image <PATH_TO_IMAGE>`

Arguments:
- **--image** : Provide image path that you want to get prediction on
