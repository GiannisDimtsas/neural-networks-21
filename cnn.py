import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# Loading the MNIST data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#data preprocessing
#normalize x_train and y_train to scale down between 0 and 1 because their values are between 0 and 255
#we do not normalize y_train and y_test because these are the labels 0,1,2,3,4,5,6,7,8,9
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Merge train data and test data
inputs = np.concatenate((x_train, x_test), axis=0)
targets = np.concatenate((y_train, y_test), axis=0)

# Define the K-fold Cross Validator K=5
kfold = KFold(n_splits=5, shuffle=True)

# K-fold Cross Validation model evaluation
fold_number = 1

for train, test in kfold.split(inputs, targets):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(28,kernel_size=(3,3), input_shape=(28,28,1)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

    # Compiling and optimizing model

    #create loss
    scce = tf.keras.losses.SparseCategoricalCrossentropy()
    mse = tf.keras.losses.MeanSquaredError()

    #create optimizer with different learning rates and momentum
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.6)

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=[scce,mse,'accuracy'])


    # Training the model
    #earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=40, min_delta=0.1)
    history = model.fit(inputs[train], targets[train], validation_data=(inputs[test], targets[test]), epochs=10) #callbacks=[earlyStopping]

    # Evaluating the model
    results = model.evaluate(inputs[test], targets[test])
    print(results)
    print(history.history)


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()