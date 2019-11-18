import tensorflow as tf
import numpy as np
import logging
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# set level for logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.basicConfig(level=logging.INFO)


logging.info('Loading & Preprocessing Data')
(x_train,y_train), (x_test,y_test) = tf.keras.datasets.mnist.load_data()
x_train,x_test = x_train / 255.0 , x_test / 255.0


# check weither or not  model exists

if(not os.path.isdir("/tmp/test/1")):
    logging.info('Definig Model')
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
    logging.info('Model Training')
    model.fit(x_train,y_train,epochs=5)
    
    
    logging.info('Model test')
    model.evaluate(x_test,y_test,verbose=2)

    logging.info('Save Model')
    tf.saved_model.save(model, "/tmp/test/1/")

else:
    logging.info('loading saved model')
    model = tf.saved_model.load("/tmp/test/1/")
    
    logging.info('Evaluating Model')
    
    N=11
    indexes = np.random.randint(100,size=N) 
    
    x = model(tf.cast(x_test[indexes].reshape(N,28,28),tf.float32))
    
    fig=plt.figure(figsize=(8, 8))
    columns = 5
    rows = 2

    for i in range(1, columns*rows+1):
        img = x_test[indexes[i]]
        ax = fig.add_subplot(rows, columns, i)

        ax.set_title(tf.math.argmax(x,axis=1).numpy()[i])
        
        plt.imshow(img, alpha=0.25)
    plt.show()

