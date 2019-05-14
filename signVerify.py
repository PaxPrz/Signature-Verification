import os
from tqdm import tqdm
import cv2
import csv
import ast
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from skimage.morphology import skeletonize, thin

from keras import layers
from keras import models
from keras import optimizers
#from keras.preprocessing.image import ImageDataGenerator
#from keras.preprocessing.image import img_to_array, load_img

train_path = './trainingset'
x_genuine_path = './genuine'
x_forgery_path = './forgery'
x_random_path = './random'
x_random_csv_path = './randomcsv'
x_test_path = './test'
save_path = './modelsave'


def preProcessImage(train_path, final_img_size = (300,300)):
  train_batch = os.listdir(train_path)
  #print(train_batch)
  x_train = []
  train_data = [x for x in train_batch if x.endswith('png') or x.endswith('PNG') or x.endswith('jpg') or x.endswith('JPG')]

  for sample in tqdm(train_data):
    img_path = os.path.join(train_path, sample)
    #importing images from drive
    #x = image.load_img(img_path)
    #img = image.img_to_array(x)
    img = cv2.imread(img_path)

    #denoising the colored image
    #img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)

    #changing RGB to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #thresholding image
    '''
    img2 = []
    for i in img:
      temp = []
      x=0
      for j in i:
        if(j>10):
          x=255
        else:
          x=0
        temp.append(x)
      img2.append(np.array(temp))
    image = np.array(img2)
    print(type(image), image.shape)
    '''
    avg = np.average(img)
    _,image = cv2.threshold(img, int(avg)-40, 255, cv2.THRESH_BINARY)

    #padding to make image into square
    lp, rp, tp, bp = (0,0,0,0)
    if(image.shape[0]>image.shape[1]):
      lp = int((image.shape[0]-image.shape[1])/2)
      rp = lp
    elif(image.shape[1]>image.shape[0]):
      tp = int((image.shape[1]-image.shape[0])/2)
      bp = tp
    image_padded = cv2.copyMakeBorder(image, tp, bp, lp, rp, cv2.BORDER_CONSTANT, value=255)

    #resizing the image
    img = cv2.resize(image_padded, final_img_size)

    #denoising the grayscale image
    img = cv2.fastNlMeansDenoising(img, None, 10, 21)

    #producing image negative
    img = 255-img

    #skeletonizing image
    img = thin(img/255)

    #appending it in list
    x_train.append(img)

  #converting it into np-array  
  x_train = np.array(x_train)
  return x_train


def convertToInt(arr):
  t1=[]
  for x in arr:
    t2=[]
    for y in x:
      t3=[]
      for z in y:
        if(z==True):
          t3.append(1)
        else:
          t3.append(0)
      t2.append(np.array(t3))
    t1.append(np.array(t2))
  return np.array(t1).astype('uint8')

def convertToBool(arr):
  t1=[]
  for x in arr:
    t2=[]
    for y in x:
      t3=[]
      for z in y:
        if(z==1):
          t3.append(True)
        else:
          t3.append(False)
      t2.append(np.array(t3))
    t1.append(np.array(t2))
  return np.array(t1)


def csvWriter(fil_name, nparray):
  example = nparray.tolist()
  with open(fil_name+'.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows(example)


def csvReader(fil_name):
  with open(fil_name+'.csv', 'r') as f:
    reader = csv.reader(f)
    examples = list(reader)
    examples = np.array(examples)
  
  t1=[]
  for x in examples:
    t2=[]
    for y in x:
      z= ast.literal_eval(y)
      t2.append(np.array(z))
    t1.append(np.array(t2))
  ex = np.array(t1)
  return ex


mod = models.Sequential(name='CNN for Signature Verification')
mod.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(300,300,1), name='Input_Conv2D'))
mod.add(layers.MaxPooling2D((2,2), name='Hidden_MaxPooling_1'))
mod.add(layers.Conv2D(64, (3,3), activation='relu', name='Hidden_Conv2D_1'))
mod.add(layers.MaxPooling2D((2,2), name='Hidden_MaxPooling_2'))
mod.add(layers.Conv2D(128, (3,3), activation='relu', name='Hidden_Conv2D_2'))
mod.add(layers.MaxPooling2D((2,2), name='Hidden_MaxPooling_3'))
# mod.add(layers.Conv2D(128, (3,3), activation='relu'))
# mod.add(layers.MaxPooling2D((2,2)))
mod.add(layers.Flatten(name='Hidden_Flattening'))
mod.add(layers.Dropout(0.5, name='Hidden_Dropout_1'))
mod.add(layers.Dense(256, activation='relu', name='Hidden_Dense_1'))
mod.add(layers.Dropout(0.5, name='Hidden_Dropout_2'))
mod.add(layers.Dense(128, activation='relu', name='Hidden_Dense_2'))
mod.add(layers.Dense(1, activation='sigmoid', name='Output_Layer'))

mod.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=1e-3), metrics=['acc'])

if 'mysign_weights.h5' not in os.listdir(save_path):
    x_genuine = preProcessImage(x_genuine_path)
    x_random = csvReader(os.path.join(x_random_csv_path,'random'))
    #another training
    mod.fit(np.concatenate((x_genuine, x_random)).reshape(x_genuine.shape[0]+x_random.shape[0], 300,300,1), np.concatenate((np.full(x_genuine.shape[0],1),np.full(x_random.shape[0],0))), epochs=5, verbose=2, shuffle=True, validation_split=0.1)
    
    evaluated = mod.evaluate(x_genuine.reshape(x_genuine.shape[0],300,300,1), np.full((x_genuine.shape[0]),1))
    print('Accuracy: ', evaluated[1]*100, '%')
else:
    mod.load_weights(os.path.join(save_path, 'mysign_weights.h5'))

mod.save_weights(os.path.join(save_path,'mysign_weights.h5'))

x_test = preProcessImage(os.path.join(x_test_path))
x_test = x_test.reshape(300,300)
predicted_acc = mod.predict(np.array([x_test.reshape(300,300,1),]))
print("\n\nSuranjan sign check: ", predicted_acc*100 ,"%")
'''
history = mod.history.history
plt.plot(history['acc'], marker='o', linewidth=3, color='blue', label='Accuracy')
plt.plot(history['loss'], marker='X', linewidth=3, color='red', label='Loss')
plt.plot(history['val_acc'], marker='o', linewidth=3, color='green', label='Val_Acc')
plt.plot(history['val_loss'], marker='X', linewidth=3, color='brown', label='Val_Loss',)
plt.xlabel('Epochs')
plt.ylabel('Accuracy/Loss')
plt.legend()
plt.show()
plt.savefig('history3.png', bbox_inches='tight', dpi=200)
'''
#mod.save(os.path.join(save_path,'mysign.h5'))
