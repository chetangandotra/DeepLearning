# In[ ]:

import numpy as np
import seaborn as sns; sns.set()

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pylab as pl
from keras.models import Sequential
from keras.layers import Dense,SimpleRNN,Dropout
from keras.optimizers import RMSprop, Adagrad

# In[ ]:
def generateData(path_to_dataset='input.txt',batch_Size=25):

    print('Loading Data ...............................................\n')
    
    # Create List of Unique Characters in the Music    
    fHandle = open('input.txt')
    text = fHandle.read()
    chars=sorted(list(set(text)))
    print('Number of Different Characters in Music:\t',len(chars))
    split_lines = text.split("<end>\n")
    split_result = ['{}{}'.format(a,'<end>\n') for a in split_lines]
    fHandle.close()
    
    # Create index number for all the characters
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))    

    # Create training Data X and Y
    sentences = [];     next_chars = [];
    for i in range(len(split_result)):
        text = split_result[i]
        for j in range(len(text)-batch_Size-1):
            sentences.append(text[j:j+batch_Size])
            next_chars.append(text[j+batch_Size])
            
    print('Total number of batches: \t',len(sentences))
    
    print('Vectorization..............')
    X = np.zeros((len(sentences), batch_Size, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i,char_indices[next_chars[i]]] = 1;
        
    [X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size=0.2,
                                            random_state=1)    

    print('Number of Training Examples: \t',X.shape[0])
    print('Number of Test Examples: \t',X_test.shape[0])
    
    print('\nComplete.')
    return(X_train,y_train,X_test,y_test,char_indices,indices_char, len(chars),
           split_result)


def readDataFromGeneratedMusic(path_to_dataset,batch_Size, char_indices,
                               indices_char, uniqueChar):

    print('Loading Data ...............................................\n')
    
    # Create List of Unique Characters in the Music    
    fHandle = open(path_to_dataset)
    text = fHandle.read()
    split_lines = text.split("<end>\n")
    split_result = ['{}{}'.format(a,'<end>\n') for a in split_lines]
    fHandle.close()
    

    # Create training Data X and Y
    sentences = [];     next_chars = [];
    for i in range(len(split_result)):
        text = split_result[i]
        for j in range(len(text)-batch_Size-1):
            sentences.append(text[j:j+batch_Size])
            next_chars.append(text[j+batch_Size])
            
    print('Total number of batches: \t',len(sentences))
    
    print('Vectorization..............')
    X = np.zeros((len(sentences), batch_Size, uniqueChar), dtype=np.bool)
    y = np.zeros((len(sentences), uniqueChar), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i,char_indices[next_chars[i]]] = 1;
        
    print('Number of Training Examples: \t',X.shape[0])
    print('Number of Test Examples: \t',X_test.shape[0])
    
    print('\nComplete.')
    return(X,y,char_indices,indices_char, uniqueChar,
           split_result)

# In[ ]:
def buildModel(batch_Size,uniqueChar,nHiddenNeuron=100,percentDropout=0,
               optimizerUsed='RMSprop'):
    print('\nBuilding model.......................................')
    model = Sequential()
    model.add(SimpleRNN(nHiddenNeuron,input_shape=(batch_Size, uniqueChar), 
                        return_sequences=False))
    model.add(Dropout(percentDropout))
    model.add(Dense(uniqueChar,activation='softmax'))
    
    if(optimizerUsed == 'RMSprop'):
        model.compile(loss='categorical_crossentropy', 
                      optimizer=RMSprop(lr=0.01,decay=0.02),metrics=['acc'])
    if(optimizerUsed == 'Adagrad'):
        model.compile(loss='categorical_crossentropy', 
                      optimizer=Adagrad(lr=0.01, epsilon=1e-08, decay=0.02),
                        metrics=['acc'])
    
    print('Dropout Percentage: ',percentDropout,'%')
    print('Optimizer Used: ',optimizerUsed)
    print('Complete.')
    model.summary()
    return(model)

# In[ ]:
def buildTruncatedModel(original_model, batch_Size,uniqueChar,
                        nHiddenNeuron=100):
    print('\nBuilding model.......................................')
    model = Sequential()
    model.add(SimpleRNN(nHiddenNeuron,input_shape=(batch_Size, uniqueChar),
                        weights = original_model.layers[0].get_weights(), 
                        return_sequences=False))
    model.compile(loss='categorical_crossentropy', 
                      optimizer=RMSprop(lr=0.01,decay=0))
#    model.summary()
    return(model)

# In[ ]:
def reshape_into_3d(data):
    return np.reshape(data, (1,data.shape[0], data.shape[1]))

def reshape_into_2d(data):
    return np.reshape(data, (1,data.shape[0]))


## Load Data
# In[ ]:
np.random.seed(1)
batch_Size = 50
[X_train,y_train,X_test,y_test,char_indices,indices_char, 
 uniqueChar,split_sequence] = generateData('input.txt',batch_Size)

# # Initialize Model

# In[ ]:

nHiddenNeuron = 100
percentDropout = 0
optimizerList = ['RMSprop','Adagrad']
optimizerUsed = optimizerList[0]

model = buildModel(batch_Size,uniqueChar,nHiddenNeuron,
                   percentDropout,optimizerUsed)

# # Train Model
# In[ ]:

history = model.fit(X_train,y_train, batch_size=1024, nb_epoch=25,
                    verbose=1,validation_data=(X_test, y_test))

## Generating heat map
# In[ ]:
trunc_model = buildTruncatedModel(model, 1, uniqueChar, nHiddenNeuron)

np.random.seed(1)
batch_Size = 1
[X_train_t,y_train_t,char_indices,indices_char, 
 uniqueChar,split_sequence] = readDataFromGeneratedMusic('generated_input.txt',
                            batch_Size, char_indices,indices_char, uniqueChar)

inputs = []
outputs = []

for n in range(nHiddenNeuron):
    for i in range(len(X_train_t)-4):
        char=indices_char[np.argmax(X_train_t[i].flatten().astype(int))];
        if char=='\n':
            char='nl'
        elif char==' ':
            char='sp'
        elif char=='\t':
            char='tb'
        elif char=='\r':
            char='rt'
    
        inputs.append(char)
        outputs.append(trunc_model.predict(reshape_into_3d(X_train_t[i]))
        .flatten()[n])
        sns.heatmap(np.array(outputs).reshape((25,20)), 
                    annot=np.array(inputs).reshape((25,20)), fmt='', 
                    cmap='RdBu_r')