from os import walk
from os.path import join

import numpy as np
import math

from sklearn.utils import shuffle
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from keras.models import Sequential,Model
from keras.layers import Dense,LSTM,SimpleRNN
from keras.optimizers import RMSprop

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator


def generateData2(path_to_dataset='./data/input.txt',batch_Size=25):

    print('Loading Data ...............................................\n')
    # Create List of Unique Characters in the Music    
    fHandle = open('./data/input.txt')
    text = fHandle.read()
    chars=sorted(list(set(text)))
    print('Number of Different Characters in Music:\t',len(chars))
    split_lines = text.split("<end>\n")
    split_result = ['{}{}'.format(a,'<end>\n') for a in split_lines]
    fHandle.close()
    # Create index number for all the characters
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))    

    file = open('./data/raw_music_text.txt','w')
    for line in split_result:
        file.write(line)
        file.write("\n")
    file.close()
    
    # Create training Data X and Y
    sentences = [];     next_chars = [];
    for i in range(len(split_result)):
        text = split_result[i]
        nBatch = math.ceil(len(text)/batch_Size)-1
        for i in range(math.floor(nBatch/2)):
            sentences.append(text[i*batch_Size: (i+1)*batch_Size])
            next_chars.append(text[i*batch_Size+1: (i+1)*batch_Size+1])
            #next_chars.append(text[i+batch_Size])
        for i in range(math.ceil(nBatch/2),-1,-1):
            sentences.append(text[len(text)-(i+1)*batch_Size-1:len(text)-i*batch_Size-1])
            next_chars.append(text[len(text)-(i+1)*batch_Size:len(text)-i*batch_Size])    

    print('Total number of batches: \t',len(sentences))
    
    print('Vectorization..............')
    X = np.zeros((len(sentences), batch_Size, len(chars)), dtype=np.bool)
    #y = np.zeros((len(sentences), batch_Size, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), batch_Size, len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
            #y[i,t, char_indices[next_chars[i,t]]] =1
        for t, char in enumerate(next_chars[i]):
            y[i,t,char_indices[char]] = 1;
        
    #[X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size=0.2)    

    print('Number of Training Examples: \t',X.shape[0])
    #print('Number of Test Examples: \t',X_test.shape[0])
    
    print('\nComplete.')
    return(X,y,char_indices,indices_char, len(chars),split_result)

    
    

def read_data(path_to_dataset='./data/input.txt', sequence_length=25):

#    train_data = []
    file = open('./data/raw_music_text.txt','w')
    with open(path_to_dataset) as f:
        data = []
        
        for line in f.readlines():
            data.append(line.strip())
            if line.startswith('<end>'):
                file.write(str(' '.join(data)))
                file.write("\n")
                data = []
                continue
    file.write(str(' '.join(data)))
    file.close()

def generateData(path_to_dataset='./data/input.txt',batch_Size=25):
    print('Loading Data ...............................................\n')
    read_data(path_to_dataset,batch_Size)

    # Create List of Unique Characters in the Music    
    fHandle = open('./data/raw_music_text.txt')
    text = fHandle.read()
    chars=sorted(list(set(text)))
    print('Number of Different Characters in Music:\t',len(chars))
    fHandle.close()

    # Create index number for all the characters
    chars = sorted(list(set(text)))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    
    
    # Create training Data X and Y
    sentences = [];     next_chars = [];
#    with open('raw_music_text.txt') as fHandle:
#        for text in fHandle.readlines():
#             for i in range(0, len(text) - batch_Size-1):
#                 sentences.append(text[i: i+batch_Size])
#                 #next_chars.append(text[i+1: i+batch_Size+1])
#                 next_chars.append(text[i+batch_Size])

    with open('./data/raw_music_text.txt') as fHandle:
        for lines in fHandle.readlines():
            text = lines.strip()
            nBatch = math.ceil(len(text)/batch_Size)-1
            for i in range(math.floor(nBatch/2)):
                sentences.append(text[i*batch_Size: (i+1)*batch_Size])
                next_chars.append(text[i*batch_Size+1: (i+1)*batch_Size+1])
                #next_chars.append(text[i+batch_Size])
            for i in range(math.ceil(nBatch/2),-1,-1):
                sentences.append(text[len(text)-(i+1)*batch_Size-1:len(text)-i*batch_Size-1])
                next_chars.append(text[len(text)-(i+1)*batch_Size:len(text)-i*batch_Size])    

        
    print('Total number of batches: \t',len(sentences))
    
    print('Vectorization..............')
    X = np.zeros((len(sentences), batch_Size, len(chars)), dtype=np.bool)
    #y = np.zeros((len(sentences), batch_Size, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), batch_Size, len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
            #y[i,t, char_indices[next_chars[i,t]]] =1
        for t, char in enumerate(next_chars[i]):
            y[i,t,char_indices[char]] = 1;
        
    #[X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size=0.2)    

    print('Number of Training Examples: \t',X.shape[0])
    #print('Number of Test Examples: \t',X_test.shape[0])
    
    print('\nComplete.')
    return(X,y,char_indices,indices_char, len(chars))


# In[ ]:

def buildModel(batch_Size,uniqueChar):
    print('\nBuilding model.......................................')
    model = Sequential()
    model.add(SimpleRNN(100, input_shape=(batch_Size, uniqueChar), return_sequences=True))
    model.add(Dense(uniqueChar,activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.005),metrics=['acc'])
    print('Complete.')
    model.summary()
    return(model)


# In[ ]:

def generateSequence(model,batch_Size,uniqueChar,seedIndex,char_indices,indices_char, temp, maxLength,split_sequence):
    #    fileHandle = open('raw_music_text.txt')
#    seedSentence = fileHandle.readlines()[seedIndex-1].strip()
    seedSentence = split_sequence[seedIndex-1]
    seedSentence = seedSentence[0:batch_Size]
    generatedSequence = seedSentence
    
    print('Seed Sentence: \n',seedSentence)
    for i in range(maxLength):
        predict_next_char = predictNextChar(model,batch_Size,uniqueChar,seedSentence,char_indices,indices_char,temp);
        generatedSequence = generatedSequence + predict_next_char
        seedSentence = seedSentence[1:] + predict_next_char
        #print(i,seedSentence)
    print('Generated Sequence: \n',generatedSequence)
    
    
def predictNextChar(model,batch_Size,uniqueChar,sentence,char_indices,indices_char,temp):
    X = np.zeros((1,batch_Size,uniqueChar))

    #print('prediction for seed: ', sentence)
    for i,c in enumerate(sentence):
        X[0,i,char_indices[c]] = 1

    pred = model.predict(X,verbose = 0)[0,batch_Size-1]

    #print('Probability = ')
    #print(pred)
    
    preds = np.asarray(pred).astype('float64')
    preds = np.log(preds) / temp
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    #print(preds.shape)
    probas = np.random.multinomial(1, preds, 1)
    char_predict = indices_char[np.argmax(probas)]
    #print(char_predict)
    return(char_predict)


# In[ ]:

batch_Size = 50
[X,y,char_indices,indices_char, uniqueChar,split_sequence] = generateData2('./data/input.txt',batch_Size)
model = buildModel(batch_Size,uniqueChar)


# In[ ]:

#model = buildModel(batch_Size,uniqueChar)
model.fit(X,y, batch_size=1024, nb_epoch=100,verbose=1)
#model.load_weights('RNN_Text_generation_30_v1.h5')
#model.save('RNN_Text_generation_30_v2.h5')


# In[ ]:

temp = 1; maxLength = 2500; seedIndex = 7; ## Seed Index < 100

s = generateSequence(model,batch_Size,uniqueChar,seedIndex,char_indices,indices_char, temp,maxLength,split_sequence)

