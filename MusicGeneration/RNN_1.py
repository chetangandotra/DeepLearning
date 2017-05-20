import matplotlib.pyplot as plt
import numpy as np
import time
import csv
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.recurrent import SimpleRNN
np.random.seed(1234)


def read_data(path_to_dataset='./data/input.txt', sequence_length=25):
    file = open('raw_music_text.txt','w')
    
    with open(path_to_dataset) as f:
        data = []
        for line in f.readlines():
            if line.startswith('<start>'):
                continue
            elif line.startswith('<end>'):
                continue
            elif ':' in line:
                continue
            
            data.append(line.strip())
    
    print(data[1])
    file.write(str(' '.join(data)))
    file.close()
    return str(' '.join(data))
        
def transform_data(data, sequence_length=25):
    X = []
    length = len(data)
    end = 0
    start = 0
    while (end+sequence_length) < length:
        start = end
        end += sequence_length
        X.append([c for c in data[start:end]])
    return np.array(X)

def get_train_test_split(transformed_data):
    row = round(0.8 * transformed_data.shape[0])
    np.random.shuffle(transformed_data)
    train = transformed_data[:row, :]
    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = transformed_data[row:, :-1]
    y_test = transformed_data[row:, -1]
    X_train1 = []
    X_test1 = []
    for i in range(len(X_train[0])):
        X_train1.append(get_one_hot(X_train[0]))
        X_test1.append(get_one_hot(X_test[0]))
    return X_train1, get_one_hot(y_train), X_test1, get_one_hot(y_test)

def get_model(sequence_length):
    model = Sequential()
    model.add(SimpleRNN(100, input_shape=(96, 24)))
    model.add(Dense(59))
    model.add(Activation('softmax'))
    return model
 
def get_one_hot(images_names):
    all_categories = list(sorted(set(images_names)))
    C = len(all_categories)    
    one_hot = np.zeros((len(images_names), C))
    for i in range(len(images_names)):
        index1 = all_categories.index(images_names[i])
        one_hot[i, index1] = 1.0
    return one_hot 
    
if __name__ == '__main__':
    data = read_data()
    
    sequence_length = 25
    transformed_data = transform_data(data, sequence_length)
    
    X_train, y_train, X_test, y_test = get_train_test_split(transformed_data)
    
    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(X_train, (1, 96, 78))
    #testX = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    model = get_model(sequence_length)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.summary()
    model.fit(trainX, y_train, nb_epoch=100, batch_size=1, verbose=2)