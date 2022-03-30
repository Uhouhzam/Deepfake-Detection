import os
import argparse
import matplotlib
from keras.models import Model
from keras.applications.xception import Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

matplotlib.use('Agg')
current_directory = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('dataset_root')
parser.add_argument('model_root')
parser.add_argument('result_root')
parser.add_argument('--lr_pre', type=float, default=5e-3)

#Define the ROC curve
def plot_roc_curve(fper, tper):
    plt.plot(fper, tper, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    plt.show()
    name='Meso4'+'Autre'+'ROC.png'
    plt.savefig(os.path.join(args.result_root, name))
    plt.clf()

#Define the accuracy curve
def plot_accuracy(Threshold,Acc):
    plt.plot(Threshold, Acc, color='red', label='ROC')
    plt.xlabel('Threshold')
    plt.ylabel('Acc')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    plt.show()
    name='Meso4'+'Autre'+'Accucacy.png'
    plt.savefig(os.path.join(args.result_root, name))
    plt.clf()
    

def main(args):


    args.dataset_root = os.path.expanduser(args.dataset_root)
    args.model_root= os.path.expanduser(args.model_root)
    args.result_root = os.path.expanduser(args.result_root)
    

    #create Meso4
    image_dimensions = {'height':256, 'width':256, 'channels':3}
    x = Input(shape = (image_dimensions['height'], 
                        image_dimensions['width'],
                        image_dimensions['channels']))
        
    x1 = Conv2D(8, (3, 3), padding='same', activation = 'relu')(x)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
        
    x2 = Conv2D(8, (5, 5), padding='same', activation = 'relu')(x1)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)
        
    x3 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x2)
    x3 = BatchNormalization()(x3)
    x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)
        
    x4 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x3)
    x4 = BatchNormalization()(x4)
    x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)
        
    y = Flatten()(x4)
    y = Dropout(0.5)(y)
    y = Dense(16)(y)
    y = LeakyReLU(alpha=0.1)(y)
    y = Dropout(0.5)(y)
    y = Dense(1, activation = 'sigmoid')(y)

    model=Model(inputs = x, outputs = y)
    model.summary()
    
    model.load_weights(args.model_root)

    dataGenerator=ImageDataGenerator(rescale=1./255,validation_split=0.2)

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

    if os.path.exists(args.result_root) is False:
        os.makedirs(args.result_root)
    #Record result
    predict=[]
    labels=[]
    p=[]
    #Classification
    # Generating predictions on validation set, storing in separate lists
    for i in range(len(GenVal.labels)):
    
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
    
    resultpd=pd.DataFrame({'predict_proba':predict, 'label':labels})
    resultpd.to_csv(args.result_root+'/'+'AutrePredMeso4.csv', index=False)
    
    #Compute FPR ,TPR and thresholds and draw the ROC curve
    FPR, TPR, thresholds = roc_curve(labels, predict)
    plot_roc_curve(FPR, TPR)

    #Compute accuracy and draw the accuracy curve
    TNR=np.ones(len(FPR))-FPR
    Acc=(TNR+TPR)/2
    plot_accuracy(thresholds,Acc)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
    



