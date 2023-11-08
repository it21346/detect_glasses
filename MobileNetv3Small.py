import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import time
import argparse
import deeplake
import os
import pickle

class MobileNet():
    def __init__(self, ds, modelname, activation_func = 'sigmoid', loss = 'binary_crossentropy', lr = 0.001, epochs = 5):
        self.activation_func = activation_func
        self.loss =  loss
        self.epochs = epochs
        self.lr = lr
        self.model = None
        self.ds = ds
        self.modelname = modelname
        self.tf_ds = None
    
    def train_model(self):
        self.model = tf.keras.applications.MobileNetV3Small(include_top=False) #import backbone
        #convert Activeloop dataset to tensorflow dataset
        self.tf_ds = self.ds.tensorflow(tensors = ["images", "labels"]).map(lambda d: (d["images"], d["labels"]))
        # Freeze the layers in the pre-trained model to prevent them from being updated during training
        for layer in self.model.layers:
            layer.trainable = False

        x = self.model.output
        x = GlobalAveragePooling2D()(self.model.output)
        # Add sigmoid layer for the classification task of 2 classes
        output_layer  = Dense(1, activation = self.activation_func)(x)

        self.model = Model(inputs=self.model.input, outputs=output_layer)
        self.model.compile(
            loss = self.loss,
            optimizer=tf.keras.optimizers.Adam(learning_rate = self.lr),  # Adjust the learning rate as needed
            metrics=['accuracy']
        )

        validation_percentage = 15
        tf_ds_tmp = self.tf_ds.shuffle(len(self.ds)).batch(16)
        total_batches = len(self.ds) // 16
        num_batches_for_validation = int(total_batches * (validation_percentage / 100))
        tf_ds_valid = tf_ds_tmp.take(num_batches_for_validation)  # Take the calculated number of batches for validation
        tf_ds_train = tf_ds_tmp.skip(num_batches_for_validation)  # Skip the corresponding number of batches for training

        
        start = time.time() 
        self.model.fit(tf_ds_train, epochs=self.epochs, validation_data=tf_ds_valid)

        end = time.time()
        print("Training took {} seconds to complete".format(end - start))

        return self.model
    
    def save_model(self):
        path = "./models/"
        if not os.path.exists(path):
            os.mkdir(path)

        full_path = os.path.join(path, self.modelname)
        pickle.dump(self.model, open(full_path, 'wb'))
        print("Successfully saved model!")

    def load_model(self):
        path = os.path.join("./models/", self.modelname)
        if os.path.exists(path):
            self.model = pickle.load(open(path, 'rb'))
            return self.model
        else:
            print("The model requested, doesn't exist!")
            return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load the MobileNetv3Small backbone, freeze the weights, add binary classification head and train it")
    #Arguments
    
    parser.add_argument('--activation_func', default= 'sigmoid', type = str, help="Activation function for the model. Default is 'sigmoid'. ")
    parser.add_argument('--loss', default = 'binary_crossentropy', type = str, help= "Model Loss. Default is 'binary_crossentropy'. ")
    parser.add_argument('--lr', default = 0.001, type = float,  help= "Learning rate. Default is '0.001'. ")
    parser.add_argument('--epochs', default = 5, type = int, help= "Epochs for the model to train. Default is '5'.")
    parser.add_argument('--train', default = False, help= "Flag to indicate whether to train the model or not. 'True' to train the model.")
    parser.add_argument('--modelname', help= "Name for the model. If argument train = True is given, the trained model will be named likewise. \
                        If not intending training, this modelname will be used to load the corresponding model. \
                        Note: Also use .pkl in the end. Refers to any model saved at ./models/*.pkl")
    args = parser.parse_args()

    activation_func = args.activation_func
    loss = args.loss
    lr = args.lr
    train = args.train
    epochs = args.epochs
    modelname = args.modelname

    #testing purposes
    ds = deeplake.load('hub://it21346/glasses_or_not_dataset', token = os.environ.get('MY_ACTIVELOOP_API_TOKEN'))

    cl = MobileNet(ds, modelname = modelname, activation_func = activation_func, loss = loss, lr = lr, epochs = epochs)

    if train: #if we want to train
        model = cl.train_model()
        cl.save_model()

    model = cl.load_model()
    if model: #if the model existed
        print(f"Successfully loaded model {modelname}")