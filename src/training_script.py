import numpy as np
import os
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from cnn import define_model
from PIL import Image

train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        #changed dir for script run (local)
        'galvanize/capstones/capstone2/data/ut-zap50k-data/ut-zap50k-images-square/train',
        target_size=(136, 102),
        batch_size=32,
        class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(
        'galvanize/capstones/capstone2/data/ut-zap50k-data/ut-zap50k-images-square/val',
        target_size=(136, 102),
        batch_size=32,
        class_mode='categorical',
        shuffle = False)
test_generator = test_datagen.flow_from_directory(
        'galvanize/capstones/capstone2/data/ut-zap50k-data/ut-zap50k-images-square/test',
        target_size=(136, 102),
        batch_size=32,
        class_mode='categorical',
        shuffle = False)



def define_model(kernel_size=(4,4), input_shape=[136,102,3], pool_size=2, nb_classes=4):
    model = Sequential() # model is a linear stack of layers (don't change)
    # note: the convolutional layers and dense layers require an activation function
    # see https://keras.io/activations/
    # and https://en.wikipedia.org/wiki/Activation_function
    # options: 'linear', 'sigmoid', 'tanh', 'relu', 'softplus', 'softsign'
    model.add(Conv2D(32, (kernel_size[0], kernel_size[1]),
                        padding='valid', 
                        input_shape=input_shape)) #first conv. layer  KEEP
    model.add(Activation('relu')) # Activation specification necessary for Conv2D and Dense layers
    model.add(Conv2D(64, (kernel_size[0], kernel_size[1]), padding='valid')) #2nd conv. layer KEEP
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size)) # decreases size, helps prevent overfitting
    model.add(Flatten()) # necessary to flatten before going into conventional dense layer  KEEP
    print('Model flattened out to ', model.output_shape)
    # now start a typical neural network
    model.add(Dense(32)) # (only) 32 neurons in this layer, really?   KEEP
    model.add(Activation('relu'))
    model.add(Dropout(0.2)) # zeros out some fraction of inputs, helps prevent overfitting
    model.add(Dense(nb_classes)) # 4 final nodes (one for each class)  KEEP
    model.add(Activation('softmax')) # softmax at end to pick between classes 0-3 KEEP
    
    # many optimizers available, see https://keras.io/optimizers/#usage-of-optimizers
    # suggest you KEEP loss at 'categorical_crossentropy' for this multiclass problem,
    # and KEEP metrics at 'accuracy'
    # suggest limiting optimizers to one of these: 'adam', 'adadelta', 'sgd'
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    return model

model = define_model()
# model = keras.models.load_model("galvanize/capstones/capstone2/src/weights.08-0.93_secondgpusync.hdf5")



root_logdir = os.path.join(os.curdir, "my_logs")
def get_run_logdir(root_logdir):
    import time
    run_id = time.strftime('run_%Y_%m_%d-%H_%M_%S')
    return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir(root_logdir)
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

checkpoint_filepath = 'galvanize/capstones/capstone2/src/weights.{epoch:02d}-{accuracy:.2f}.hdf5'
checkpoint_cb = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True, monitor = 'val_loss')

history = model.fit(
        train_generator,
        steps_per_epoch=35020//32,
        epochs=1000,
        validation_data=validation_generator,
        validation_steps=7504//32,
        callbacks=[checkpoint_cb,early_stopping_cb,tensorboard_cb])