import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from keras.losses import binary_crossentropy
import matplotlib
import os
import argparse
import math
from keras.callbacks import ModelCheckpoint
import pickle as pkl

matplotlib.use('Agg')
current_directory = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('dataset_root')
parser.add_argument('result_root')
parser.add_argument('--epochs', type=int, default=8)#10 defore
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr_pre', type=float, default=5e-3)
parser.add_argument('--snapshot_period', type=int, default=1)
parser.add_argument('--split', type=float, default=0.2)

def main(args):

    epochs = args.epochs
    args.dataset_root = os.path.expanduser(args.dataset_root)
    args.result_root = os.path.expanduser(args.result_root)
        # split dataset for training and validation
    dataGenerator=ImageDataGenerator(rescale=1./255,validation_split=args.split)

    GenTrain=dataGenerator.flow_from_directory(
        args.dataset_root,
        target_size=(256, 256),
        batch_size=args.batch_size,
        class_mode='binary',
        shuffle=True,
        subset='training')
    print(GenTrain.class_indices)
    GenVal=dataGenerator.flow_from_directory(
        args.dataset_root,
        target_size=(256, 256),
        batch_size=args.batch_size,
        class_mode='binary',
        shuffle=True,
        subset='validation')
    print(GenVal.class_indices)
    print("Training on %d images and labels" % (len(GenTrain.labels)))
    print("Validation on %d images and labels" % (len(GenVal.labels)))

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

    meso4=Model(inputs = x, outputs = y)
    meso4.summary()

    #compile le model
    meso4.compile(
        loss=binary_crossentropy,
        optimizer=Adam(lr=args.lr_pre),
        metrics=['accuracy']
    )
    if os.path.exists(args.result_root) is False:
        os.makedirs(args.result_root)

    # Train with the dataset
    hist = meso4.fit_generator(
        generator=GenTrain,
        steps_per_epoch=math.ceil(
            len(GenTrain.labels) / args.batch_size),
        epochs=args.epochs,
        validation_data=GenVal,
        validation_steps=math.ceil(
            len(GenVal.labels) / args.batch_size),
        verbose=1,
        callbacks=[
            ModelCheckpoint(
                filepath=os.path.join(
                    args.result_root,
                    'model_ep{epoch}_valloss{val_loss:.3f}.h5'),
                period=args.snapshot_period,
            ),
        ],
    )
    meso4.save(os.path.join(args.result_root, 'model_final.h5'))

    # Create result graphs
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']

    # save graph image
    plt.plot(range(epochs), acc, marker='.', label='accuracy')
    plt.plot(range(epochs), val_acc, marker='.', label='val_accuracy')
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig(os.path.join(args.result_root, 'accuracy.png'))
    plt.clf()

    plt.plot(range(epochs), loss, marker='.', label='loss')
    plt.plot(range(epochs), val_loss, marker='.', label='val_loss')
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(os.path.join(args.result_root, 'loss.png'))
    plt.clf()

    # save plot data
    plot = {
        'accuracy': acc,
        'val_accuracy': val_acc,
        'loss': loss,
        'val_loss': val_loss,
    }
    with open(os.path.join(args.result_root, 'plot.dump'), 'wb') as f:
        pkl.dump(plot, f)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

    
