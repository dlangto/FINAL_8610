import keras
import glob
from keras.preprocessing.image import ImageDataGenerator
import pickle
import pandas
import numpy

machine = pickle.load(open('cnn_image_machine.pickle', 'rb'))
new_data = ImageDataGenerator(rescale=1./255)
new_data = new_data.flow_from_directory(directory='part_3/sample_new_data', shuffle=False, target_size=(50,50), batch_size=1)
#batch size has to =1
new_data.reset()
new_data_length = len([i for i in glob.glob('sample_profile_pictures/*.jpg')])
#count the data length
machine.predict(new_data)
#use new data to make prediction
prediction = numpy.argmax(machine.predict(new_data, steps=new_data_length), axis=1)
#this is our prediction, steps is how many new images in our dataset
print(prediction)
print(new_data.filenames)

results = [[new_data.filenames[i], prediction[i]] for i in range(new_data_length)]
results_dataframe = pandas.DataFrame(results, columns=['image', 'prediction'])
results_dataframe.to_csv('prediction.csv', index=False)
#This program predicts how good the programmer will be by their picture. 2 is the best, 0 is the worst.