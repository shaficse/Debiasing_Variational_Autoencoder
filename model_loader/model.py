import tensorflow as tf

def make_standard_classifier(n_filters=12, n_outputs=1):

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(filters=2*n_filters,kernel_size=5,strides=2,padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2D(filters=2*n_filters,kernel_size=5,strides=2,padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2D(filters=4*n_filters,kernel_size=3,strides=2,padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2D(filters=6*n_filters,kernel_size=3,strides=2,padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(n_outputs, activation=None)

        ]
    )
    return model
if __name__ == '__main__':
    ml = make_standard_classifier()
    print(ml)