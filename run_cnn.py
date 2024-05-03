import keras
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

from keras import layers
import pickle

dataset = ImageDataGenerator(rescale=1./255, validation_split=.25)
#load the dataset, set data preprocessing

training_dataset = ImageDataGenerator(rescale = 1/255).flow_from_directory(directory='image_dataset/',
                                         target_size=(50,50), shuffle=True, batch_size=32, subset='training')
#split the data into training and validation dataset
validation_dataset = ImageDataGenerator(rescale = 1/255).flow_from_directory(directory='image_dataset/',
                                         target_size=(50,50), shuffle=True, batch_size=32, subset='validation')
                                         
values = list(training_dataset.class_indices.values())
keys = list(training_dataset.class_indices.keys())
print([[values[i], keys[i]] for i in range(len(values))])
num_of_classes = len(values)
#save the keys and values so you can read the model

# Initializing the model
machine = Sequential()
machine.add(layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu', input_shape=(50,50,3)))
#kernal size is grid we analyze. Padding makes a boundary. input shape is 3 because we have three colors. Make sure input_shape matches
machine.add(layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
machine.add(layers.MaxPooling2D(pool_size=(2,2)))
machine.add(layers.Dropout(0.25))

machine.add(layers.Conv2D(filters=64, activation='relu', kernel_size=(3,3)))
machine.add(layers.Conv2D(filters=64, activation='relu', kernel_size=(3,3)))
machine.add(layers.MaxPooling2D(pool_size=(2,2)))
machine.add(layers.Dropout(0.25))
          
machine.add(layers.Flatten())
machine.add(layers.Dense(units=64, activation='relu'))
machine.add(layers.Dense(units=64, activation='relu'))
machine.add(layers.Dropout(0.25))
machine.add(layers.Dense(units=num_of_classes, activation='softmax'))

machine.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')
#use this every time
machine.fit(x=training_dataset, validation_data=validation_dataset, epochs=30) 
#this is the new code. The old code won't give you an error, but it won't work

pickle.dump(machine, open('cnn_image_machine.pickle', 'wb'))
#this whole thing is like one round of kfold