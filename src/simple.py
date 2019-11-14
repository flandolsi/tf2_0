import tensorflow as tf
import logging
import os


# set level for logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.basicConfig(level=logging.INFO)



(x_train,y_train), (x_test,y_test) = tf.keras.datasets.mnist.load_data()
logging.info('Data Loaded')

x_train,x_test = x_train / 255.0 , x_test / 255.0


model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10,activation='softmax')
    ])


model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
logging.info('Model Compiled')


model.fit(x_train,y_train,epochs=5)
logging.info('Model trained')


model.evaluate(x_test,y_test,verbose=2)



