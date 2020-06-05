import os, shutil
original_dataset_dir="/home/ram/Desktop/task2_data_prepro"
base_dir="/home/ram/Desktop/task2_classfication_data"
os.mkdir(base_dir)



train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)

train_anger_dir = os.path.join(train_dir, 'anger')
os.mkdir(train_anger_dir)

train_disgust_dir = os.path.join(train_dir, 'disgust')
os.mkdir(train_disgust_dir)

train_happy_dir = os.path.join(train_dir, 'happy')
os.mkdir(train_happy_dir)

train_neutral_dir = os.path.join(train_dir, 'neutral')
os.mkdir(train_neutral_dir)

train_sad_dir = os.path.join(train_dir, 'sad')
os.mkdir(train_sad_dir)

train_surprise_dir = os.path.join(train_dir, 'surprise')
os.mkdir(train_surprise_dir)

validation_anger_dir = os.path.join(validation_dir, 'anger')
os.mkdir(validation_anger_dir)

validation_disgust_dir = os.path.join(validation_dir, 'disgust')
os.mkdir(validation_disgust_dir)

validation_happy_dir = os.path.join(validation_dir, 'happy')
os.mkdir(validation_happy_dir)

validation_neutral_dir = os.path.join(validation_dir, 'neutral')
os.mkdir(validation_neutral_dir)

validation_sad_dir = os.path.join(validation_dir, 'sad')
os.mkdir(validation_sad_dir)

validation_surprise_dir = os.path.join(validation_dir, 'surprise')
os.mkdir(validation_surprise_dir)

fnames = ['{}.jpg'.format(i) for i in range(1, 501)]
anger = original_dataset_dir + '/' + 'anger'
for fname in fnames:
    src = os.path.join(anger, fname)
    dst = os.path.join(validation_anger_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['{}.jpg'.format(i) for i in range(1,501)]
disgust=original_dataset_dir+'/'+'disgust'
for fname in fnames:
    src = os.path.join(disgust, fname)
    dst = os.path.join(validation_disgust_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['{}.jpg'.format(i) for i in range(1, 501)]
happy = original_dataset_dir + '/' + 'happy'
for fname in fnames:
    src = os.path.join(happy, fname)
    dst = os.path.join(validation_happy_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['{}.jpg'.format(i) for i in range(1, 501)]
neutral = original_dataset_dir + '/' + 'neutral'
for fname in fnames:
    src = os.path.join(neutral, fname)
    dst = os.path.join(validation_neutral_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['{}.jpg'.format(i) for i in range(1, 501)]
sad = original_dataset_dir + '/' + 'sad'
for fname in fnames:
    src = os.path.join(sad, fname)
    dst = os.path.join(validation_sad_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['{}.jpg'.format(i) for i in range(1, 501)]
surprise = original_dataset_dir + '/' + 'surprise'
for fname in fnames:
    src = os.path.join(surprise, fname)
    dst = os.path.join(validation_surprise_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['{}.jpg'.format(i) for i in range(501, 2340)]
anger = original_dataset_dir + '/' + 'anger'
for fname in fnames:
    src = os.path.join(anger, fname)
    dst = os.path.join(train_anger_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['{}.jpg'.format(i) for i in range(501, 2363)]
disgust = original_dataset_dir + '/' + 'disgust'
for fname in fnames:
    src = os.path.join(disgust, fname)
    dst = os.path.join(train_disgust_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['{}.jpg'.format(i) for i in range(501, 2665)]
happy = original_dataset_dir + '/' + 'happy'
for fname in fnames:
    src = os.path.join(happy, fname)
    dst = os.path.join(train_happy_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['{}.jpg'.format(i) for i in range(501, 2851)]
neutral = original_dataset_dir + '/' + 'neutral'
for fname in fnames:
    src = os.path.join(neutral, fname)
    dst = os.path.join(train_neutral_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['{}.jpg'.format(i) for i in range(501,1858)]
sad=original_dataset_dir+'/'+'sad'
for fname in fnames:
    src = os.path.join(sad, fname)
    dst = os.path.join(train_sad_dir, fname)
    shutil.copyfile(src, dst)


fnames = ['{}.jpg'.format(i) for i in range(501, 2281)]
surprise = original_dataset_dir + '/' + 'surprise'
for fname in fnames:
    src = os.path.join(surprise, fname)
    dst = os.path.join(train_surprise_dir, fname)
    shutil.copyfile(src, dst)

from keras.applications import VGG16

conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

conv_base.summary()

from keras import models
from keras import layers

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(6, activation='softmax'))

model.summary()

from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150, 150),
                                                    batch_size=50,
                                                    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        target_size=(150, 150),
                                                        batch_size=50,
                                                        class_mode='categorical')

batch_size = 50 * 6
history = model.fit_generator(train_generator,
                              steps_per_epoch=11352 // batch_size,
                              epochs=30,
                              validation_data=validation_generator,
                              validation_steps=3000 // batch_size)


import matplotlib.pyplot as plt
acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']
epochs=range(1,len(acc)+1)
plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title("training and validation accuracy")
plt.legend()

plt.figure()
plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title("training and validation loss")
plt.legend()

plt.show()

model.save("/home/ram/Desktop/facial_emotion_classification")
model.save("/home/ram/Desktop/facial_emotion_classification.h5")