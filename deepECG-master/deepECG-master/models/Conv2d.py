from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, GlobalAveragePooling2D, MaxPooling2D, Flatten


def conv2d(input_dim, output_dim=4):
    model = Sequential()
    # model.load_weights('my_model_weights.h5')

    # 64 conv
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_dim, padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # 128 conv
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # #256 conv
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # #512 conv
    # model.add(Conv2D(512, (3, 3), activation='relu'))
    # model.add(Conv2D(512, (3, 3), activation='relu'))
    # model.add(Conv2D(512, (3, 3), activation='relu'))
    # model.add(Conv2D(512, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Conv2D(512, (3, 3), activation='relu'))
    # model.add(Conv2D(512, (3, 3), activation='relu'))
    # model.add(Conv2D(512, (3, 3), activation='relu'))
    # model.add(Conv2D(512, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Dense part
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model
