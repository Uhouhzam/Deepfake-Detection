import math
import os
import argparse
import matplotlib
import pickle as pkl
import matplotlib.pyplot as plt
from keras.applications.xception import Xception
from tensorflow.keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

matplotlib.use('Agg')
current_directory = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('dataset_root')
parser.add_argument('result_root')
parser.add_argument('--epochs_pre', type=int, default=3)#10 defore
parser.add_argument('--epochs_fine', type=int, default=3)#30 defore
parser.add_argument('--batch_size_pre', type=int, default=64)
parser.add_argument('--batch_size_fine', type=int, default=32)
parser.add_argument('--lr_pre', type=float, default=5e-3)
parser.add_argument('--lr_fine', type=float, default=5e-4)
parser.add_argument('--snapshot_period_pre', type=int, default=1)
parser.add_argument('--snapshot_period_fine', type=int, default=1)
parser.add_argument('--split', type=float, default=0.2)


def main(args):

    epochs = args.epochs_pre + args.epochs_fine
    args.dataset_root = os.path.expanduser(args.dataset_root)
    args.result_root = os.path.expanduser(args.result_root)

    # split dataset for training and validation
    dataGenerator=ImageDataGenerator(rescale=1./255,validation_split=args.split)

    GenTrain=dataGenerator.flow_from_directory(
        args.dataset_root,
        target_size=(256, 256),
        batch_size=args.batch_size_pre,
        class_mode='binary',
        shuffle=True,
        subset='training')
    print(GenTrain.class_indices)
    GenVal=dataGenerator.flow_from_directory(
        args.dataset_root,
        target_size=(256, 256),
        batch_size=args.batch_size_pre,
        class_mode='binary',
        shuffle=True,
        subset='validation')
    print(GenVal.class_indices)
    print("Training on %d images and labels" % (len(GenTrain.labels)))
    print("Validation on %d images and labels" % (len(GenVal.labels)))


    if os.path.exists(args.result_root) is False:
        os.makedirs(args.result_root)

    # Build a custom Xception
    # from pre-trained Xception model
    # the default input shape is (299, 299, 3)
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
        loss=binary_crossentropy,
        optimizer=Adam(lr=args.lr_pre),
        metrics=['accuracy']
    )
    

    # train
    hist_pre = model.fit_generator(
        generator=GenTrain,
        steps_per_epoch=math.ceil(
            len(GenTrain.labels) / args.batch_size_pre),
        epochs=args.epochs_pre,
        validation_data=GenVal,
        validation_steps=math.ceil(
            len(GenVal.labels) / args.batch_size_pre),
        verbose=1,
        callbacks=[
            ModelCheckpoint(
                filepath=os.path.join(
                    args.result_root,
                    'model_pre_ep{epoch}_valloss{val_loss:.3f}.h5'),
                period=args.snapshot_period_pre,
            ),
        ],
    )
    model.save(os.path.join(args.result_root, 'model_pre_final.h5'))

    # Train the whole model
    # split dataset for training and validation for training the whole model
    dataGeneratorW=ImageDataGenerator(rescale=1./255,validation_split=args.split)

    GenTrainW=dataGeneratorW.flow_from_directory(
        args.dataset_root,
        target_size=(256, 256),
        batch_size=args.batch_size_fine,
        class_mode='binary',
        shuffle=True,
        subset='training')
    print(GenTrainW.class_indices)
    GenValW=dataGeneratorW.flow_from_directory(
        args.dataset_root,
        target_size=(256, 256),
        batch_size=args.batch_size_fine,
        class_mode='binary',
        shuffle=True,
        subset='validation')
    print(GenValW.class_indices)
    print("Training on %d images and labels" % (len(GenTrainW.labels)))
    print("Validation on %d images and labels" % (len(GenValW.labels)))
    for layer in model.layers:
        layer.trainable = True #all layers are set as trainable

    # recompile
    model.compile(
        optimizer=Adam(lr=args.lr_fine),
        loss=binary_crossentropy,
        metrics=['accuracy'])

    # train
    hist_fine = model.fit_generator(
        generator=GenTrainW,
        steps_per_epoch=math.ceil(
            len(GenTrainW.labels) / args.batch_size_fine),
        epochs=args.epochs_fine,
        validation_data=GenValW,
        validation_steps=math.ceil(
            len(GenValW.labels) / args.batch_size_fine),
        verbose=1,
        callbacks=[
            ModelCheckpoint(
                filepath=os.path.join(
                    args.result_root,
                    'model_fine_ep{epoch}_valloss{val_loss:.3f}.h5'),
                period=args.snapshot_period_fine,
            ),
        ],
    )
    model.save(os.path.join(args.result_root, 'model_fine_final.h5'))

    # Create result graphs
    acc = hist_pre.history['accuracy']
    val_acc = hist_pre.history['val_accuracy']
    loss = hist_pre.history['loss']
    val_loss = hist_pre.history['val_loss']
    acc.extend(hist_fine.history['accuracy'])
    val_acc.extend(hist_fine.history['val_accuracy'])
    loss.extend(hist_fine.history['loss'])
    val_loss.extend(hist_fine.history['val_loss'])

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
