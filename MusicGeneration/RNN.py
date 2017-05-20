# In[ ]:

import numpy as np

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

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
                      optimizer=RMSprop(lr=0.01,decay=0.01),metrics=['acc'])
    if(optimizerUsed == 'Adagrad'):
        model.compile(loss='categorical_crossentropy', 
                      optimizer=Adagrad(lr=0.01, epsilon=1e-05, decay=0.0),
                        metrics=['acc'])
    
    print('Dropout Percentage: ',percentDropout,'%')
    print('Optimizer Used: ',optimizerUsed)
    print('Complete.')
    model.summary()
    return(model)

# In[ ]:
def generateSequence(fHandle, model,batch_Size,uniqueChar,seedIndex,
                     char_indices,indices_char, temp, maxLength,
                     split_sequence,count):
    
    seedSentence = split_sequence[seedIndex-1]
    seedSentence = seedSentence[0:batch_Size]
    generatedSequence = seedSentence
    
    fHandle.write(str(count)+'. \n\n')
    fHandle.write('Temperature: '+str(temp)+'\n')
    fHandle.write('Seed Sentence: '+str(seedSentence)+'\n\n')
    for i in range(maxLength):
        if(seedSentence[batch_Size-5:batch_Size] == '<end>'):
            break
        predict_next_char = predictNextChar(model,batch_Size,uniqueChar,
                                            seedSentence,char_indices,
                                            indices_char,temp);
        generatedSequence = generatedSequence + predict_next_char
        seedSentence = seedSentence[1:] + predict_next_char
    fHandle.write('Generated Sequence: \n'+str(generatedSequence)+'\n\n\n')
        
def predictNextChar(model,batch_Size,uniqueChar,sentence,
                    char_indices,indices_char,temp):
    X = np.zeros((1,batch_Size,uniqueChar))

    for i,c in enumerate(sentence):
        X[0,i,char_indices[c]] = 1

    pred = model.predict(X,verbose = 0)[0]
    preds = np.asarray(pred).astype('float64')
    preds = np.log(preds) / temp
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    char_predict = indices_char[np.argmax(probas)]
    return(char_predict)

# In[ ]:

def plotGraph(history, percentDropout, nHiddenNeuron,optimizerUsed):
    plt.plot(history.history['loss'],'r-', label='Train Loss')
    plt.plot(history.history['val_loss'],'b-', label='Validation Loss')
    plt.tick_params(labelright = True)
    plt.title('"Train/Validation Loss vs Epoch"')
    plt.ylabel('Train/Validation Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train Loss', 'Validation Loss'], loc='upper left', 
               shadow=True)
    
    xCoord = int(0.5*len(history.history['loss']));
    ran = (max(history.history['loss']+history.history['val_loss']) 
            - min(history.history['loss']+history.history['val_loss']))
    st = min(history.history['loss']+history.history['val_loss'])
    
    plt.text(xCoord,st+ran*0.85, 'Dropout : '+str(percentDropout))
    plt.text(xCoord,st+ran*0.9,'Neurons : '+str(nHiddenNeuron))
    plt.text(xCoord,st+ran*0.95, 'Optimier: '+optimizerUsed )
    
    fileName = ('trainPlot_Dropout_'+str(percentDropout)
                +'_Neuron_'+str(nHiddenNeuron)+'_'+optimizerUsed)
    plt.savefig(fileName)
    plt.show()

# # Load Data

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

history = model.fit(X_train,y_train, batch_size=1024, nb_epoch=45,
                    verbose=1,validation_data=(X_test, y_test))
plotGraph(history, percentDropout, nHiddenNeuron,optimizerUsed)

# # Generate Music

# In[ ]:

temp = 0.5; 
tempList = [0.5]#[0.5, 1, 2]
maxLength = 1000
seedIndex = [12,15,21,71,89,53, 55, 22, 42,11,8,1,2,3,4,5,6,7]
count = 1

fHandle = open('GeneratedMusic.txt','w')
for temp in tempList:
    for i in range(100):
        generateSequence(fHandle,model,batch_Size,uniqueChar,i,#seedIndex[i],
                         char_indices,indices_char, temp,maxLength,
                         split_sequence,count)
        count = count+1
fHandle.close()
    
print('Music Generated in File: GeneratedMusic.txt')

