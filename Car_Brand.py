import warnings
warnings.filterwarnings('ignore')
from keras.models import Sequential,Model
from keras.layers import Dense,Flatten,Convolution2D,MaxPooling2D,ReLU,Softmax,BatchNormalization,Dropout
from keras.applications.vgg16 import VGG16,preprocess_input
from glob import glob

train_path='D:/INeuron/DLCVNLP/Project/data/car brand/Train'
test_path='D:/INeuron/DLCVNLP/Project/data/car brand/Test'

vgg16=VGG16(input_shape=[224,224,3],weights='imagenet',include_top=False)

for layer in vgg16.layers:
    layer.trainable=False

folders=glob(r'D:\INeuron\DLCVNLP\Project\data\car brand\Train\*')
print(len(folders))

x=Flatten()(vgg16.output)
prediction=Dense(len(folders),activation='softmax')(x)
model=Model(inputs=vgg16.input,outputs=prediction)
model.summary()

model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

train=ImageDataGenerator(rescale=1/255,preprocessing_function=preprocess_input,rotation_range=0.4,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,vertical_flip=True,horizontal_flip=True,fill_mode='nearest')
test=ImageDataGenerator(rescale=1/255)

train_set=train.flow_from_directory(train_path,batch_size=32,target_size=(224,224),class_mode='categorical')
test_set=test.flow_from_directory(test_path,batch_size=32,target_size=(224,224),class_mode='categorical')


model.fit_generator(train_set,validation_data=test_set,epochs=1,steps_per_epoch=200,validation_steps=100)
model.save("carbrandclassifier.h5")

import numpy as np
test_image=image.load_img(r'C:\Users\sarie\Downloads\Other pictures\mercedes.jpg',target_size=(224,224))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=model.predict(test_image)
print(result)
if result[0][0]==1:
    print('audi')
elif result[0][1]==1:
    print('Lamborghini')
else:
    print('Mercedes Benz')

