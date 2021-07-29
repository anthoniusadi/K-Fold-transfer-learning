import tensorflow as tf
import keras
import pandas as pd
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D,Flatten,MaxPool2D
from tensorflow.keras.applications import MobileNet
from tensorflow. keras.applications.mobilenet import preprocess_input
import numpy as np
from IPython.display import Image
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.models import load_model
from sklearn.model_selection import KFold
def get_model_name(k):
    return 'saved_models/mobile_'+str(k)+'.h5'
def log(k):
    return  "logs_"+str(k)+"/mobilenetv2"
VALIDATION_ACCURACY = []
VALIDATION_LOSS = []

save_dir = '/home/448690/ASL4/saved_models/'
fold_var = 1


base_model_path = "/home/448690/ASL4/mobilenetv2.h5"

train_path='/home/m448690/ASL4/augmented_data/'
in_img=(224,224,3)

colname=["data","label"]
train_data = pd.read_csv('train_augmented.csv',dtype=str)
Y = train_data['label']
X = train_data['data']
kf = KFold(n_splits = 5,shuffle=True)
#! kfold here
for train_index, val_index in kf.split(X):
    print("==================================================+++++++++++++++++++ TRAINING FOLD",fold_var," +++++++++++++++=================================================")
    training_data = train_data.iloc[train_index]
    validation_data = train_data.iloc[val_index]
    
    base_model = tf.keras.applications.MobileNetV2(input_shape=in_img ,include_top=False, weights="mobilenetv2.h5")

    for layer in (base_model.layers):
        layer.trainable=True
    #for layer in base_model.layers[:70]:
    #    layer.trainable=False
    #for layer in base_model.layers[70:]:
    #    layer.trainable=True

    x=base_model.output
    x=GlobalAveragePooling2D()(x)
    x=Flatten()(x)
    #x=GlobalAveragePooling2D()(x)
    x=Dense(1024,activation='relu')(x) #dense layer 1
    x=Dense(812,activation='relu')(x) #dense layer 2
   # x=Dense(512,activation='relu')(x) #dense layer 3
    preds=Dense(10,activation='softmax')(x) #final layer with softmax activation for N classes

    model=Model(inputs=base_model.input,outputs=preds) #specify the inputs and outputs

    model.summary()

    train_datagen = ImageDataGenerator(
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=0.2
        )


    train_generator=train_datagen.flow_from_dataframe(training_data,directory=train_path,
                                                    x_col = "data", y_col = "label",
                                                    target_size=(224,224),
                                                    color_mode='rgb',
                                                    batch_size=64,
                                                    class_mode='categorical',
                                                    shuffle=True
                                                    )

    val_generator=train_datagen.flow_from_dataframe(validation_data,directory=train_path,
                                                    x_col = "data", y_col = "label",
                                                    target_size=(224,224),
                                                    color_mode='rgb',
                                                    batch_size=64,
                                                    class_mode='categorical',
                                                    shuffle=False
                                                    )

    #set learning rate
    optimizer = tf.keras.optimizers.Adam(lr=0.000001)
    model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
    #keras callbacks
    from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
    from tensorflow.keras.callbacks import TensorBoard
    import os

    #filepath="ckpt/best.hdf5"

    checkpoint = ModelCheckpoint(filepath=get_model_name(fold_var),
                    monitor='val_loss',
                    verbose=1,
                    save_weights_only=False,
                    save_best_only=True,
                    mode='auto')
   # logdir = "logs/mobilenetv2"
    tfboard = TensorBoard(log_dir=log(fold_var))

    earlystop = EarlyStopping(monitor='val_loss',
                min_delta=.0001,
                patience=13,
                verbose=1,
                mode='auto',
                baseline=None,
                restore_best_weights=True)
    callbacks_list = [checkpoint,tfboard,earlystop]
    batch_size=64
    #step_size_train=train_generator.n//train_generator.batch_size
    #step_size_val=val_generator.n//val_generator.batch_size
    #fit model
    history=model.fit_generator(
        train_generator,
        steps_per_epoch = len(train_generator),
    #    steps_per_epoch = train_generator.samples // batch_size,
        validation_data = val_generator,
        validation_steps = len(val_generator),
    #    validation_steps = val_generator.samples // batch_size,
        callbacks=callbacks_list,
        epochs = 30)
    #model.load_model("saved_models/model_"+str(fold_var)+".hdf5")
    
    results = model.evaluate(val_generator)
    results = dict(zip(model.metrics_names,results))
    print(results)
    VALIDATION_ACCURACY.append(results['accuracy'])
    VALIDATION_LOSS.append(results['loss'])
    print("saving model !")
    model_json = model.to_json()
    with open("mobile_"+str(fold_var)+".json", "w") as json_file:
        json_file.write(model_json)
        model.save_weights("mobile_"+str(fold_var)+".h5")     
    tf.keras.backend.clear_session()

    fold_var += 1
###!save model

# print("saving model !")
# model_json = model.to_json()
# with open("mobile.json", "w") as json_file:
#     json_file.write(model_json)
#    model.save_weights("mobile.h5") 
print("Training DONE !")
print("hasil_ACC",VALIDATION_ACCURACY)
print("hasil_LOSS",VALIDATION_LOSS)
#print("====================================FINE TUNNING!====================================")
#for layer in model.layers:
#    layer.trainable=True

#opt=keras.optimizers.Adam(lr=0.00001)
#model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
#model.fit(train_generator,epochs=10,callbacks=callbacks_list,validation_data=val_generator)

#model_json = model.to_json()
#with open("mobile.json", "w") as json_file:
#    json_file.write(model_json)
#    model.save_weights("mobile.h5") # serialize weights to HDF5
#print("DONE!")
