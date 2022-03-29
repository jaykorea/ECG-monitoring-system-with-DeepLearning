import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM


def deepLSTM(input_dim, output_dim=4):
    model = Sequential()

    model.add(LSTM(512, return_sequences=True, input_shape=input_dim))
    # model.add(Dropout(0.25))

    model.add(LSTM(256, return_sequences=True))
    # model.add(Dropout(0.25))

    model.add(LSTM(128, return_sequences=True))
    # model.add(Dropout(0.25))

    model.add(LSTM(64, return_sequences=True))
    # model.add(Dropout(0.25))

    model.add(LSTM(32))

    model.add(Dense(output_dim, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model