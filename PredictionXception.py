from email import generator
import os
import argparse
import matplotlib
import numpy as np
from keras.models import Model
from keras.preprocessing import image
from keras.applications.xception import Xception
from keras.losses import binary_crossentropy
from keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use('Agg')
current_directory = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('dataset_root')
parser.add_argument('model_root')
parser.add_argument('result_root')
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--batch_size_fine', type=int, default=32)
parser.add_argument('--lr_pre', type=float, default=5e-3)

#Define fonction to plot the ROC curve
def plot_roc_curve(fper, tper):
    plt.plot(fper, tper, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    plt.show()
    name='Xception'+'GDWCT_fine_'+'ROC.png'
    plt.savefig(os.path.join(args.result_root, name))
    plt.clf()

#Define fonction to plot the accuracy curve
def plot_accuracy(Threshold,Acc):
    plt.plot(Threshold, Acc, color='red', label='ROC')
    plt.xlabel('Threshold')
    plt.ylabel('Acc')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    plt.show()
    name='Xception'+'GDWCT_fine_'+'Accucacy.png'
    plt.savefig(os.path.join(args.result_root, name))
    plt.clf()

def main(args):


    args.dataset_root = os.path.expanduser(args.dataset_root)
    args.model_root= os.path.expanduser(args.model_root)
    args.result_root = os.path.expanduser(args.result_root)
    
    # Build a custom Xception
    # from pre-trained Xception model
    # the default input shape is (256, 256, 3)
    base_model = Xception(
        include_top=False,
        weights='imagenet',
        input_shape=(256, 256, 3))

    # create a custom top classifier
    num_classes=1
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='sigmoid')(x)
    model = Model(inputs=base_model.inputs, outputs=predictions)

    model.summary()
    
    # Train only the top classifier
    # freeze the body layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile model
    model.compile(
        loss=binary_crossentropy,#categorical_crossentropy,
        optimizer=Adam(lr=args.lr_pre),
        metrics=['accuracy']
    )
    
    model.load_weights(args.model_root)

    #Creat generator for prediction
    dataGenerator=ImageDataGenerator(rescale=1./255,validation_split=0.4)

    GenTrain=dataGenerator.flow_from_directory(
        args.dataset_root,
        target_size=(256, 256),
        batch_size=32,
        class_mode='binary',
        shuffle=True,
        subset='training')
    print(GenTrain.class_indices)
    print(len(GenTrain.labels))

    GenVal=dataGenerator.flow_from_directory(
        args.dataset_root,
        target_size=(256, 256),
        batch_size=1,
        class_mode='binary',
        shuffle=True,
        subset='validation')
    print(GenVal.class_indices)
    print(len(GenVal.labels))

    #Create liste for saving the values
    predict=[]
    labels=[]

    #Classification
    # Generating predictions on validation set, storing in separate lists
    for i in range(len(GenVal)-1):
    
        # Loading next picture, generating prediction
        X, y = GenVal.next()
        predict.append(model.predict(X)[0][0])
        labels.append(y[0])
        # Printing status update
        if i % 100 == 0:
            print(i, ' predictions completed.')
    
        if i == len(GenVal.labels)-1:
            print("All", len(GenVal.labels), "predictions completed")
    

    if os.path.exists(args.result_root) is False:
        os.makedirs(args.result_root)
    
    #Save the resultat prediction and label
    resultpd=pd.DataFrame({'predict_proba':predict, 'label':labels})
    resultpd.to_csv(args.result_root+'/'+'GDWCT_fine_resultPred.csv')

    #Compute FPR and TPR   
    FPR, TPR, thresholds = roc_curve(labels, predict)

    #Draw ROC curve.
    plot_roc_curve(FPR, TPR)

    #Computer accuracy and draw accuracy curve.
    TNR=np.ones(len(FPR))-FPR
    Acc=(TNR+TPR)/2
    plot_accuracy(thresholds[1:],Acc[1:])






if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
    



