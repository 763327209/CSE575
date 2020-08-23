import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from matplotlib import pyplot as plt

# Plotting of Graph
def results(model, value):
    # Plot 1
    plt.title('Loss Vs Epoch Plot')
    plt.plot(value.history['val_loss'])
    plt.plot(value.history['loss'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Testing'],loc='center right')
    plt.show()

    # Plot 2
    error = [1 - a for a in value.history['accuracy']]
    value_error = [1 - a for a in value.history['val_accuracy']]
    plt.plot(error)
    plt.plot(value_error)
    plt.title('Error Vs Epoch Plot')
    plt.ylabel('Error')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Testing'], loc='center right')
    plt.show()

    # Plot 3
    plt.title('Accuracy Vs Epoch Plot')
    plt.plot(value.history['val_accuracy'])
    plt.plot(value.history['accuracy'])
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')  
    plt.legend(['Training', 'Testing'], loc='center right')
    plt.show()

    # Output 
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test Error:', value_error[-1])
    print('Test Loss:', score[0])
    print('Test Accuracy:', score[1])

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#Question 1:
#Run the baseline code and report the accuracy.
model = Sequential()
model.add(Conv2D(6, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dense(84, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))


# https://keras.io/optimizers/ 
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(lr=0.1, rho=0.95, epsilon=1e-07, decay=0.0),
              metrics=['accuracy'])

value = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
results(model, value)

#Question 2:
#Change the kernel size to 5*5, redo the experiment, plot the learning errors along with the epoch, and report the testing error and accuracy on the test set.
model1 = Sequential()
model1.add(Conv2D(6, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=input_shape))
model1.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding="same"))
model1.add(Conv2D(16, (5, 5), activation='relu'))
model1.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding="same"))
model1.add(Flatten())
model1.add(Dense(120, activation='relu'))
model1.add(Dense(84, activation='relu'))

model1.add(Dense(num_classes, activation='softmax'))


# https://keras.io/optimizers/ 
model1.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(lr=0.1, rho=0.95, epsilon=1e-07, decay=0.0),
              metrics=['accuracy'])

value_1 = model1.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
results(model1, value_1)

#Question 3:
#CNN with kernel-size as 3*3 and feature maps as 8,32 for 2 convolution layers respectively
model2 = Sequential()
model2.add(Conv2D(8, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding="same"))
model2.add(Conv2D(32, (3, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding="same"))
model2.add(Flatten())
model2.add(Dense(120, activation='relu'))
model2.add(Dense(84, activation='relu'))

model2.add(Dense(num_classes, activation='softmax'))


# https://keras.io/optimizers/ 
model2.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(lr=0.1, rho=0.95, epsilon=1e-07, decay=0.0),
              metrics=['accuracy'])

value_2 = model2.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
results(model2, value_2)