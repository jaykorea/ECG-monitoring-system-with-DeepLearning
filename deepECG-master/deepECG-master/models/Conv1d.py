import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, GlobalAveragePooling1D, MaxPooling1D


def conv1d(input_dim, output_dim=4):
    model = Sequential()
    
    model.add(Conv1D(128, 55, activation='relu', input_shape=(input_dim, 1)))
    model.add(MaxPooling1D(10))
    model.add(Dropout(0.5))
    
    model.add(Conv1D(128, 25, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Dropout(0.5))
    
    model.add(Conv1D(128, 10, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Dropout(0.5))
    
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(GlobalAveragePooling1D())

    # model.add(Flatten())
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(output_dim, kernel_initializer='normal', activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
