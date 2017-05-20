
import numpy
from LoadMNIST import load_mnist
from sklearn.utils import shuffle
import plotly.plotly as py1
import plotly.graph_objs as go
import copy
#import bigfloat

#----------------------------------------Utility functions--------------------------------------#

def get_data(N=60000, N_test=10000, validationReqd = True):
    # Load MNIST data using libraries available
    training_data, training_labels = load_mnist('training')    
    test_data, test_labels = load_mnist('testing')
    
    # Training_data is N x 784 matrix
    training_data = flattenAndNormalize(N, 784, training_data) 
    training_labels = training_labels[:N]
    test_data = flattenAndNormalize(N_test, 784, test_data)
    test_labels = test_labels[:N_test]

    # Adding column of 1s for bias
    training_data = addOnesColAtStart(training_data)
    test_data = addOnesColAtStart(test_data)
    
    if (validationReqd):
        # Last 10% of training data size will be considered as the validation set
        N_validation = int (N / 6.0)
        validation_data = training_data[N-N_validation:N]
        validation_labels = training_labels[N-N_validation:N]
        N=N-N_validation
    else:
        validation_data = []
        validation_labels = []
    #update training data to remove validation data
    training_data = training_data[:N]
    training_labels = training_labels[:N]    

    return training_data, training_labels, test_data, test_labels, validation_data, validation_labels
    
def flattenAndNormalize(rows, cols, twoDArr):
    flattened_arr = numpy.zeros(shape=(rows, cols))
    for row in range(0, rows):
        i=0
        for element in twoDArr[row]:
            for el1 in element:
                flattened_arr[row][i] = el1
                i = i+1
    return flattened_arr

def addOnesColAtStart(matrix):
    Ones = numpy.ones(len(matrix))
    newMatrix = numpy.c_[Ones, matrix]
    return newMatrix
    
# Single method for calculating sigmoid (logisitic) activation 
# and its derivative
def f_sigmoid(X, derivativeReqd=False):
    if not derivativeReqd:
        return 1.0 / (1 + numpy.exp(-X))
        #return 1.0 / (1 + bigfloat.exp(-X, bigfloat.precision(100)))
    return numpy.multiply(f_sigmoid(X), (1 - f_sigmoid(X)))

# Method to calculate the softmax activation
def f_softmax(X):
    Z = numpy.sum(numpy.exp(X), axis=1)
    Z = Z.reshape(Z.shape[0], 1)
    return numpy.exp(X) / Z

# Single method for tanh sigmoid and its derivative
def f_tanh(x, a=0, derivativeReqd = False):
    mul_factor = 1.7159
    div_factor = 2.0/3.0
    tanh_term = numpy.tanh(div_factor*x)
    if not derivativeReqd:
        return (mul_factor * tanh_term + a*x)
    return (div_factor * mul_factor * (1 - (tanh_term*tanh_term))) + a

# Method to return batches from data and labels after shuffling        
def get_batches(X, Y, batch_size):
    N = len(X)#X.shape[0]
    batch_X = []
    batch_Y = []
    count = 0
    X, Y = shuffle(X, Y, random_state=0)
    while count + batch_size <= N:
        batch_X.append(X[count:count+batch_size, :])
        one_hot = get_one_hot_representation(Y[count:count+batch_size], 10)
        batch_Y.append(one_hot)
        count += batch_size
    return batch_X, batch_Y

# Convert numberical y value (0-9) to a one-hot representation    
def get_one_hot_representation(Y, C=10):
    one_hot = numpy.zeros((Y.shape[0], C))
    for i in range(Y.shape[0]):
        one_hot[i, Y[i]] = 1.0
    return one_hot

# Layer specific forward propogation
def forward_prop(W, Z_prev, layer_no, num_layers, 
                 batch_size, layer_config, activation=f_sigmoid):
    #Fprime = numpy.zeros((layer_config[layer_no], batch_size))   
    Z = activation(numpy.dot(Z_prev, W))
    if layer_no == num_layers - 1:
        return Z, []
    else:
        # Hidden layers need to have their Fprime values computed
        Fprime = activation(Z, derivativeReqd=True).T
        # Add bias terms for the hidden layers
        Z = numpy.append(numpy.ones((Z.shape[0], 1)), Z, axis=1)
        return Z, Fprime

# Forward propagation begins
def forward_prop_for_all_layers(W, Z, train_data, num_layers,  
                                batch_size, layer_config):
    Z[0] = train_data
    Fprime = []
    for i in range(1, num_layers-1):
        Z[i], Fprime1 = forward_prop(W[i-1], Z[i-1], i, num_layers, 
                                        batch_size, layer_config)
        Fprime.append(Fprime1)
    # Separate call to send f_softmax as the activation function parameter
    Z[-1], Fprime1 = forward_prop(W[-1], Z[-2], num_layers-1, 
                            num_layers, batch_size, layer_config, f_softmax)
    return Z, Fprime

# Backpropagation step
def backprop(y, t, num_layers, Fprime, delta, W):
    delta[-1] = (t - y).T
    for i in range(num_layers-2, 0, -1):
        # Remove the bias column before operating on W
        W1 = W[i][1:, :]
        temp = numpy.dot(W1, delta[i])
        delta[i-1] = numpy.multiply(temp, Fprime[i-1])
    return delta

# Update the weight vectors
def update_weights(learning_rate, num_layers, Z, delta, W, 
                   momentum, prev_del_w, size):
    ret_val_W_prev = []
    gradient_bp = []
    for i in range(0, num_layers-1):
        W_grad = (learning_rate*(numpy.dot(delta[i], Z[i])).T + momentum*prev_del_w[i])/float(size);
        gradient_bp.append(W_grad)
        W[i] += W_grad
        ret_val_W_prev.append(-1*W_grad)
    return W, ret_val_W_prev, gradient_bp

# Update the weight vectors
def update_weights_analytical_gradient(learning_rate, num_layers, Z, delta, W, t, activation_hidden = f_sigmoid, activation_output = f_softmax):
    e = 0.01
    gradient = []
    W_save = copy.deepcopy(W)
    for x in range(0, num_layers-1):
        gradient.append(numpy.zeros((W[x].shape[0], W[x].shape[1])))
        for i in range(0,W[x].shape[0]):
            for j in range(0,W[x].shape[1]):
                W_pos = copy.deepcopy(W_save)
                W_neg = copy.deepcopy(W_save)
                
                W_pos[x][i][j] = W_save[x][i][j] + e
                W_neg[x][i][j] = W_save[x][i][j] - e
                #print(W_save[x][i][j] - W_pos[x][i][j])
                Z_pos = copy.deepcopy(Z)
                Z_neg = copy.deepcopy(Z)
                for k in range(1,num_layers-1):
                    Z_pos[k] = activation_hidden(numpy.dot(Z_pos[k-1], W_pos[k-1]))
                    Z_neg[k] = activation_hidden(numpy.dot(Z_neg[k-1], W_neg[k-1]))
                    Z_pos[k] = numpy.append(numpy.ones((Z_pos[k].shape[0], 1)), Z_pos[k], axis=1)
                    Z_neg[k] = numpy.append(numpy.ones((Z_neg[k].shape[0], 1)), Z_neg[k], axis=1)                    
        
                Z_pos[-1] = activation_output(numpy.dot(Z_pos[-2], W_pos[-1]))
                Z_neg[-1] = activation_output(numpy.dot(Z_neg[-2], W_neg[-1]))

                gradient[x][i][j] = ( learning_rate * (numpy.dot(t, numpy.log(Z_pos[-1].T)) - numpy.dot(t, numpy.log(Z_neg[-1].T))) / 2.0*e)                   
                
    return gradient
    
# Hard coded forward propagation for 2 layer NN - only for testing 
def hard_code_forward_prop(W, Z, num_layers, 
                           batch_size, layer_config):
    A = []
    A.append(numpy.dot(Z[0], W[0]))
    Z[1] = f_sigmoid(A[0])
    Fprime = []
    Fprime.append(f_sigmoid(Z[1], derivativeReqd=True).T)
    Z[1] = numpy.append(numpy.ones((Z[1].shape[0], 1)), Z[1], axis=1)
    A.append(numpy.dot(Z[1], W[1]))
    Z[2] = f_softmax(A[1])
    return Z, Fprime

# train the model
def fit(X, y, X_test, y_test, X_val, y_val, iterations, learning_rate, 
        num_layers, W, Z, Z_val, Z_test, delta,   
        batch_size, layer_config, batch_size_val, batch_size_test,
        momentum, prev_del_W):
            
    train_data_batches, train_label_batches = get_batches(X, y, 
                                                          batch_size)        
    val_data_batches, val_label_batches = get_batches(X_val, y_val, 
                                                      batch_size_val)
    test_data_batches, test_label_batches = get_batches(X_test, y_test, 
                                                       batch_size_test)
    
    test_error = 0.0
    
    for t in range(iterations):
        # Train for particular iteration
        for i in range(len(train_label_batches)):
            batch_data = train_data_batches[i]
            batch_labels = train_label_batches[i]
            # Forward Propagation   
                     
            Z_updated, Fprime = forward_prop_for_all_layers(W, Z,
                                                             batch_data, 
                                                             num_layers, 
                                                             batch_size, 
                                                             layer_config)
            # Back-propagation                                                 
            delta = backprop(Z_updated[-1], batch_labels, num_layers,
                             Fprime, delta, W)
            # Update the weights
            W_true, prev_del_W, gradient_bp = update_weights(learning_rate, num_layers,
                               Z_updated, delta, W, momentum, prev_del_W, len(batch_labels))
            
            # Update the weights
            gradient_ana = update_weights_analytical_gradient(learning_rate, num_layers,
                               Z_updated, delta, W, batch_labels)
            
            gradient_bp = [gradient_bp[i] - gradient_ana[i] for i in range(0,len(gradient_bp))]
            print(gradient_bp)            
            
            diff_grad_ij = numpy.fabs(gradient_bp[0] - gradient_ana[0])
            diff_grad_jk = numpy.fabs(gradient_bp[1] - gradient_ana[1])

            max_diff_ij = numpy.amax(diff_grad_ij)
            max_diff_jk = numpy.amax(diff_grad_jk)    
            mean_diff_ij = numpy.mean(diff_grad_ij)
            mean_diff_jk = numpy.mean(diff_grad_jk)
            print("max_diff_ij: " + str(max_diff_ij))
            print("max_diff_jk: " + str(max_diff_jk))
            
            print("mean_diff_ij: " + str(mean_diff_ij))
            print("mean_diff_jk: " + str(mean_diff_jk))

            numpy.savetxt("Gradient_diff.csv", gradient_bp[0], delimiter=",")
            numpy.savetxt("Gradient_diff2.csv", gradient_bp[1], delimiter=",")
                
    print ("Test error = " + str(test_error/len(y_test)))
    
def plotly_graphs(percent_correct_train, percent_correct_test):
    py1.sign_in('chetang', 'vil7vTAuCSWt2lEZvaH9')
    trace = []
    graph_y = []
    graph_y.append(percent_correct_train)
    graph_y.append(percent_correct_test)
    for i in range(2):
        name = "Training Data"
        if i == 1:
            name = "Test Data"
        y1 = graph_y[i]
        x1 = [j+1 for j in range(len(y1))]
        trace1 = go.Scatter(
            x=x1,
            y=y1,
            name = name,
            connectgaps=True
        )
        trace.append(trace1)
    data = trace
    fig = dict(data=data)
    py1.iplot(fig, filename='percentCorrect-connectgaps_1')
        
#---------------------------------------Main function------------------------------------------------------
        
if __name__ == "__main__":     
    numpy.random.seed(0)
    learning_rate = 0.001
    momentum = 0.0
    N = 1
    N_test = 1
    iteration = 1
    X, y, X_test, y_test, X_validation, y_validation = get_data(N, 
                                                                N_test, 
                                                                False)
    X = X/255.0
    #X_test = X_test/255.0
    #X_validation = X_validation/255.0
    #X = X - numpy.mean(X, axis=0)[numpy.newaxis, :]
    for r in range(0,X.shape[0]):
        X[r] = X[r] - sum(X[r])/X.shape[1]
    #X_test = X_test - numpy.mean(X_test, axis=0)[numpy.newaxis,: ]
    #X_validation = X_validation - numpy.mean(X_validation, axis=0)[numpy.newaxis,:]
    
    batch_size = 1
    batch_size_val = 1
    batch_size_test = 1
    
    # Layers - 784, 100, 10
    layer_config = [784, 300, 10]
    num_layers = len(layer_config)
    W = []
    Z = []
    Z_test = []
    Z_val = []
    delta = []
    prev_del_W = []
        
    for i in range(num_layers): 
        # common for all layers except input layer
        if i != 0:
            Z.append(numpy.zeros((batch_size, layer_config[i])))
            Z_test.append(numpy.zeros((batch_size_test, layer_config[i])))
            Z_val.append(numpy.zeros((batch_size_val, layer_config[i])))
            delta.append(numpy.zeros((batch_size, layer_config[i])))
        else:
            Z.append(numpy.zeros((batch_size, layer_config[i]+1)))
            Z_test.append(numpy.zeros((batch_size_test, layer_config[i]+1)))
            Z_val.append(numpy.zeros((batch_size_val, layer_config[i]+1)))
        # For all layers except output
        if i != num_layers - 1:
            W.append(numpy.random.normal(size=[layer_config[i]+1, 
                                               layer_config[i+1]], 
                                               loc = 0.0, 
                                               scale=0.001))
            prev_del_W.append(numpy.zeros((layer_config[i]+1, 
                                               layer_config[i+1])))
                                               
    fit(X, y, X_test, y_test, X_validation, y_validation, iteration, 
        learning_rate, num_layers, W, 
        Z, Z_val, Z_test, delta,  
        batch_size, layer_config,
        batch_size_val, batch_size_test, momentum, prev_del_W)