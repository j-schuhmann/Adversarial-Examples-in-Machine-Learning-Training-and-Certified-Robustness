import tensorflow as tf


# model for the adversarial training experiments
def ground_model(input_shape):
    model = tf.keras.models.Sequential([
       
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    return model



# models for the certified robustness experiments

def ground_model_delta_experiments3(input_shape):

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(1024, activation=tf.nn.sigmoid),  
        tf.keras.layers.Dense(512, activation=tf.nn.sigmoid),  
        tf.keras.layers.Dense(128, activation=tf.nn.sigmoid),                        
        tf.keras.layers.Dense(1)                               
    ])
    return model


def ground_model_delta_experiments2(input_shape):

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(128, activation=tf.nn.sigmoid),  
        tf.keras.layers.Dense(16, activation=tf.nn.sigmoid),                        
        tf.keras.layers.Dense(1)                               
    ])
    return model


def ground_model_delta_experiments1(input_shape):

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(16, activation=tf.nn.sigmoid),                       
        tf.keras.layers.Dense(1)                               
    ])
    return model

def ground_model_delta_experiments4(input_shape):

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(2048, activation=tf.nn.sigmoid),
        tf.keras.layers.Dense(1024, activation=tf.nn.sigmoid),  
        tf.keras.layers.Dense(512, activation=tf.nn.sigmoid),  
        tf.keras.layers.Dense(128, activation=tf.nn.sigmoid),                        
        tf.keras.layers.Dense(1)                                 
    ])
    return model


