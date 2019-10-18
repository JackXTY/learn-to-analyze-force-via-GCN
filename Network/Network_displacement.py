# This is a a GCN network model with convolution,
# which is rewrited from PointGCN.
# Let's have a look at it.

import numpy as np
import tensorflow as tf
import random
import scipy
from scipy.sparse.linalg import eigsh


N_ = 165
M =  1600
N_out = 3
test_size = 400
'''
M = Number of Files
N = Number of Nodes
N_ = Number of Nodes which need output
'''
file_object = open('adjacency.txt')
A = [[0 for i in range(0,N_)]for j in range(0,N_)]
while True:
    line = file_object.readline().split()
    #print(line)
    if (len(line) == 0):
        break
    A[int(line[0])][int(line[1])] = 1
#print (A)
file_object.close()




def normalize_adj(adj):
    adj = scipy.sparse.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = scipy.sparse.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def scaled_laplacian(adj):
    adj_normalized = normalize_adj(adj)
    laplacian = scipy.sparse.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - scipy.sparse.eye(adj.shape[0])
    return scaled_laplacian

A = np.mat(A)

temp_A = scaled_laplacian(A)
new_A = temp_A.A
#print(new_A)




#####################
###// read data //###
#####################
# store the inputdata(figure) in the 'all_x'
# x[i] means the result of i's test
file_object = open('train_data.txt')
x = []
y = []
for i in range(0,M):
    temp_x = []
    temp_y = []

    for j in range(0,N_):
        temp_temp_x = []
        temp_temp_y = []      
        line = file_object.readline().split()
        for k in range(0,7):    
            temp_temp_x += [float(line[k])]
        temp_x += [temp_temp_x]
        for k in range(16-N_out,16):
            temp_temp_y += [float(line[k])]
        temp_y += [temp_temp_y]

    x += [temp_x]
    y += [temp_y]

file_object.close()
#print(len(x))print(len(y))print(len(x[0]))print(len(y[0]))


test_file_object = open('test_data.txt')
test_x = []
test_y = []
for i in range(0,test_size):
    temp_x = []
    temp_y = []

    for j in range(0,N_):
        temp_temp_x = []
        temp_temp_y = []    
        line = test_file_object.readline().split()
        
        for k in range(0,7):
            temp_temp_x += [float(line[k])]
        temp_x += [temp_temp_x]
        for k in range(16-N_out,16):
            temp_temp_y += [float(line[k])]
        temp_y += [temp_temp_y]
    test_x += [temp_x]
    test_y += [temp_y]
        
test_file_object.close()


#####################
###// Parameter //###
#####################
class Parameters():

    def __init__(self):
    	
        self.neighborNumber = 8
        self.outputClassN = N_out
        self.pointNumber = N_

        self.gcn_1_filter_n = 1120
        self.gcn_2_filter_n = 560
        self.gcn_3_filter_n = 280
        self.gcn_4_filter_n = 140

        self.fc_1_n = 140
        self.chebyshev_1_Order = 6
        self.chebyshev_2_Order = 5
        self.chebyshev_3_Order = 4
        self.chebyshev_4_Order = 3
        self.keep_prob_1 = 1 #0.9 original
        self.keep_prob_2 = 1
        
        '''
        self.batchSize = 28
        self.testBatchSize = 1
	    self.max_epoch = 260
        self.learningRate = 12e-4
        self.dataset = 'ModelNet40'
        self.weighting_scheme = 'weighted'
        self.modelDir = '/raid60/yingxue.zhang2/ICASSP_code/global_pooling/model/'
        self.logDir = '/raid60/yingxue.zhang2/ICASSP_code/global_pooling/log/'
        self.fileName = '0112_1024_40_cheby_4_3_modelnet40_max_var_first_second_layer'
        self.weight_scaler = 40#50
        '''
para = Parameters()

'''
print(x[273][15])
print(x[1589][15])
print(y[273][15])
print(y[1589][15])

print(test_x[73][10])
print(test_x[289][10])
print(test_y[73][10])
print(test_y[289][10])
'''


#####################
###// Main Part //###
#####################
# activation function: tanh & identity(y=x)
# aggregate = D**-0.5 * A *  D**-0.5 * X

# prepare the value
sess = tf.InteractiveSession()

def weightVariables(shape, name):
    initial = tf.truncated_normal(shape=shape, mean=0, stddev=0.000001)
    #initial = tf.ones(shape=shape)
    return tf.Variable(initial, name=name)


def chebyshevCoefficient(chebyshevOrder, inputNumber, outputNumber):
    chebyshevWeights = dict()
    for i in range(chebyshevOrder):
        initial = tf.truncated_normal(shape=[inputNumber, outputNumber], mean=0, stddev=0.000001)
        chebyshevWeights['w_' + str(i)] = tf.Variable(initial)
    return chebyshevWeights


def gcnLayer(inputPC, scaledLaplacian, pointNumber, inputFeatureN, outputFeatureN, chebyshev_order):
    biasWeight = weightVariables([outputFeatureN], name='bias_w')
    chebyshevCoeff = chebyshevCoefficient(chebyshev_order, inputFeatureN, outputFeatureN)
    chebyPoly = []
    cheby_K_Minus_1 = tf.matmul(scaledLaplacian, inputPC)
    cheby_K_Minus_2 = inputPC
    chebyPoly.append(cheby_K_Minus_2)
    chebyPoly.append(cheby_K_Minus_1)
    for i in range(2, chebyshev_order):
        chebyK = 2 * tf.matmul(scaledLaplacian, cheby_K_Minus_1) - cheby_K_Minus_2
	    #chebyK = tf.matmul(scaledLaplacian, cheby_K_Minus_1)
        chebyPoly.append(chebyK)
        cheby_K_Minus_2 = cheby_K_Minus_1
        cheby_K_Minus_1 = chebyK
        #cheby_K_Minus_2, cheby_K_Minus_1 = cheby_K_Minus_1, chebyK
    chebyOutput = []

    for i in range(chebyshev_order):
        
        weights = chebyshevCoeff['w_' + str(i)]
        chebyPolyReshape = tf.reshape(chebyPoly[i], [-1, inputFeatureN])
        output = tf.matmul(chebyPolyReshape, weights)
        
        '''
        print(i)
        print(chebyPoly[i]) # (72,7)
        print(chebyPolyReshape) # (168,3)
        print(weights) # (3,100)
        '''
        #output = tf.reshape(output, [-1, pointNumber, outputFeatureN])
        output = tf.reshape(output, [pointNumber, outputFeatureN])
        chebyOutput.append(output)

    gcnOutput = tf.add_n(chebyOutput) + biasWeight
    gcnOutput = tf.nn.relu(gcnOutput)
    return gcnOutput


def globalPooling(gcnOutput, featureNumber): 

    # There is some bugs with pooling.
    # So this part is omitted now.

    #l2_max_pooling_pre = tf.reshape(gcnOutput, [-1, 1024, featureNumber, 1])
    #max_pooling_output_1=tf.nn.max_pool(l2_max_pooling_pre,ksize=[1,1024,1,1],strides=[1,1,1,1],padding='VALID')
    #max_pooling_output_1=tf.reshape(max_pooling_output_1,[-1,featureNumber])
    #mean, var = tf.nn.moments(gcnOutput, axes=[1])
    #poolingOutput = tf.concat([max_pooling_output_1, var], axis=1)
    #print("\ngcnOutput:")
    #print(gcnOutput)

    '''
    gcnOutput = tf.reshape(gcnOutput,[-1,N_,para.gcn_1_filter_n])
    mean, var = tf.nn.moments(gcnOutput, axes=[1])
    max_f = tf.reduce_max(gcnOutput, axis=[1])
    #print("max_f:")print(max_f)print("var:")print(var)print()
    poolingOutput = tf.concat([max_f, var], axis=1)
    #return max_f
    return poolingOutput
    '''
    return gcnOutput

#fully connected layer without relu activation
def fullyConnected(features, inputFeatureN, outputFeatureN):
    weightFC = weightVariables([inputFeatureN, outputFeatureN], name='weight_fc')
    biasFC = weightVariables([outputFeatureN], name='bias_fc')
    outputFC = tf.matmul(features,weightFC)+biasFC
    return outputFC


inputPC = tf.placeholder(tf.float32, [para.pointNumber, 7]) # PC = Point Cloud
inputGraph = tf.placeholder(tf.float32, [para.pointNumber , para.pointNumber])
outputLabel = tf.placeholder(tf.float32, [para.pointNumber, para.outputClassN])

#scaledLaplacian = tf.reshape(inputGraph, [para.pointNumber, para.pointNumber])
scaledLaplacian = inputGraph

#weights = tf.placeholder(tf.float32, [None])
#lr = tf.placeholder(tf.float32)
keep_prob_1 = tf.placeholder(tf.float32)
keep_prob_2 = tf.placeholder(tf.float32)

# gcn layer 1
gcn_1 = gcnLayer(inputPC, scaledLaplacian, pointNumber=para.pointNumber, inputFeatureN=7,
                    outputFeatureN=para.gcn_1_filter_n,
                    chebyshev_order=para.chebyshev_1_Order)
gcn_1_output = tf.nn.dropout(gcn_1, keep_prob=keep_prob_1)
gcn_1_pooling = globalPooling(gcn_1_output, featureNumber=para.gcn_1_filter_n)
print("\nThe output of the first gcn layer is {}".format(gcn_1_pooling))
#print (gcn_1_pooling)

# gcn_layer_2

gcn_2 = gcnLayer(gcn_1_output, scaledLaplacian, pointNumber=para.pointNumber, inputFeatureN=para.gcn_1_filter_n,
                    outputFeatureN=para.gcn_2_filter_n,
                    chebyshev_order=para.chebyshev_2_Order)
gcn_2_output = tf.nn.dropout(gcn_2, keep_prob=keep_prob_1)
gcn_2_pooling = globalPooling(gcn_2_output, featureNumber=para.gcn_2_filter_n)
print("The output of the second gcn layer is {}".format(gcn_2_pooling))

#gcn_layer_3

gcn_3 = gcnLayer(gcn_2_output, scaledLaplacian, pointNumber=para.pointNumber, inputFeatureN=para.gcn_2_filter_n,
                    outputFeatureN=para.gcn_3_filter_n,
                    chebyshev_order=para.chebyshev_3_Order)
gcn_3_output = tf.nn.dropout(gcn_3, keep_prob=keep_prob_1)
gcn_3_pooling = globalPooling(gcn_3_output, featureNumber=para.gcn_3_filter_n)
print("The output of the second gcn layer is {}".format(gcn_3_pooling))

#gcn_layer_3

gcn_4 = gcnLayer(gcn_3_output, scaledLaplacian, pointNumber=para.pointNumber, inputFeatureN=para.gcn_3_filter_n,
                    outputFeatureN=para.gcn_4_filter_n,
                    chebyshev_order=para.chebyshev_4_Order)
gcn_4_output = tf.nn.dropout(gcn_4, keep_prob=keep_prob_1)
gcn_4_pooling = globalPooling(gcn_4_output, featureNumber=para.gcn_4_filter_n)
print("The output of the second gcn layer is {}".format(gcn_4_pooling))

# concatenate global features
#globalFeatures = gcn_3_pooling

globalFeatures = tf.concat([gcn_1_pooling, gcn_2_pooling], axis=1)
globalFeatures = tf.concat([globalFeatures, gcn_3_pooling], axis=1)
globalFeatures = tf.concat([globalFeatures, gcn_4_pooling], axis=1)
globalFeatures = tf.nn.dropout(globalFeatures, keep_prob=keep_prob_2)
print("The global feature is {}".format(globalFeatures))
#globalFeatureN = para.gcn_2_filter_n*2
#globalFeatureN = (para.gcn_1_filter_n + para.gcn_2_filter_n)*2 
globalFeatureN = para.gcn_1_filter_n + para.gcn_2_filter_n + para.gcn_3_filter_n + para.gcn_4_filter_n


# fully connected layer 1
fc_layer_1 = fullyConnected(globalFeatures, inputFeatureN=globalFeatureN, outputFeatureN=para.fc_1_n)
fc_layer_1 = tf.nn.relu(fc_layer_1)
fc_layer_1 = tf.nn.dropout(fc_layer_1, keep_prob=keep_prob_2)
print("The output of the first fc layer is {}".format(fc_layer_1))


# fully connected layer 2
fc_layer_2 = fullyConnected(fc_layer_1, inputFeatureN=para.fc_1_n, outputFeatureN=para.outputClassN)
print("The output of the second fc layer is {}".format(fc_layer_2))
print()

#loss = tf.reduce_mean(outputLabel - fc_layer_2)
#loss = loss*loss
loss_MSE = tf.losses.mean_squared_error(labels=outputLabel,predictions=fc_layer_2)
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss_MSE) 

#cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits( labels = outputLabel, logits = fc_layer_2 )
#train_step = tf.train.AdamOptimizer(1e-6).minimize(cross_entropy) 
#train_step = tf.train.GradientDescentOptimizer(1e-9).minimize(cross_entropy) #(99.3%) times:13900
#train_step = tf.train.AdadeltaOptimizer(1e-3).minimize(cross_entropy)
#train_step = tf.train.AdagradOptimizer(1e-4).minimize(cross_entropy)

saver = tf.train.Saver()
saver.restore(sess, "/tmp/Network_v1_stress_model.ckpt")
#sess.run(tf.initialize_all_variables())

# /tmp/Network_v1_stress_model.ckpt  For AdamOptimizer with loss_MSE

old_real_evaluation = 0
real_evaluation = 0
for k in range(0,20):
    times = 1000

    for j in range(0,times):
        ran = random.randint(0,M-1)
        train_step.run(feed_dict = {inputPC: x[ran], 
            inputGraph: new_A, outputLabel: y[ran], 
            keep_prob_1: para.keep_prob_1, keep_prob_2: para.keep_prob_2})

    print("Epochs " + str(times*(k+1)) +":")

    '''
    if(k%3==0):
        ran = random.randint(0,M-1)
        out_gcn_1_output = gcn_1_output.eval(session=sess,feed_dict = {inputPC: x[ran],
            inputGraph: new_A, outputLabel: y[ran], keep_prob_1: para.keep_prob_1, keep_prob_2: para.keep_prob_2})
        print("\nout_gcn_1_output:")
        print(out_gcn_1_output)
        out_gcn_2_output = gcn_2_output.eval(session=sess,feed_dict = {inputPC: x[ran],
            inputGraph: new_A, outputLabel: y[ran], keep_prob_1: para.keep_prob_1, keep_prob_2: para.keep_prob_2})
        print("out_gcn_2_output:")
        print(out_gcn_2_output)
        out_gcn_3_output = gcn_3_output.eval(session=sess,feed_dict = {inputPC: x[ran],
            inputGraph: new_A, outputLabel: y[ran], keep_prob_1: para.keep_prob_1, keep_prob_2: para.keep_prob_2})
        print("out_gcn_3_output:")
        print(out_gcn_3_output)
        out_fc_layer_1 = fc_layer_1.eval(session=sess,feed_dict = {inputPC: x[ran],
            inputGraph: new_A, outputLabel: y[ran], keep_prob_1: para.keep_prob_1, keep_prob_2: para.keep_prob_2})
        print("out_fc_layer_1:")
        print(out_fc_layer_1)
        #out_fc_layer_2 = fc_layer_2.eval(session=sess,feed_dict = {inputPC: x[ran],
        #    inputGraph: new_A, outputLabel: y[ran], keep_prob_1: para.keep_prob_1, keep_prob_2: para.keep_prob_2})
        #print("out_fc_layer_2:")
        #print(out_fc_layer_2)
    '''

    # evaluate
    evaluation = 0
    total_loss = 0
    for k in range(0,test_size):

        loss_now = loss_MSE.eval(session=sess,feed_dict = {inputPC: test_x[k],
            inputGraph: new_A, outputLabel: test_y[k],
            keep_prob_1: 1, keep_prob_2: 1})
        total_loss += loss_now

        temp = 0
        y_out = fc_layer_2.eval(session=sess,feed_dict = {inputPC: test_x[k],
            inputGraph: new_A, outputLabel: test_y[k],
            keep_prob_1: 1, keep_prob_2: 1})
        y_out = y_out.tolist()
        for i in range(0,N_):
            for j in range(0,N_out):
                if y[ran][i][j] != 0 :
                    temp += ( 1 - abs( ( y_out[i][j] - test_y[k][i][j] ) / test_y[k][i][j] ) )
        evaluation += temp / (N_out*N_)

    print("loss: "+str(total_loss/test_size))

    evaluation = evaluation / test_size
    print("accuracy: "+str(evaluation))

    old_real_evaluation = real_evaluation
    real_evaluation = evaluation
    if(old_real_evaluation > 0 and old_real_evaluation < real_evaluation):
        print("Overfit!!!")
        break
    '''
    for k in range(0,test_size):
        temp = 0
        y_out = fc_layer_2.eval(session=sess,feed_dict = {inputPC: test_x[k],
            inputGraph: new_A, outputLabel: test_y[k], keep_prob_1: para.keep_prob_1, keep_prob_2: para.keep_prob_2})
        y_out = y_out.tolist()
        for i in range(0,N_):
            for j in range(0,N_out):
                if y[ran][i][j] != 0 :
                    temp += abs( ( y_out[i][j] - test_y[k][i][j] ) / test_y[k][i][j] )
        evaluation += temp / (N_out*N_)

    evaluation = evaluation / test_size
    old_real_evaluation = real_evaluation
    real_evaluation = evaluation
    print("Test Accuracy: "+str(real_evaluation))
    '''

    #if (old_real_evaluation > real_evaluation and old_real_evaluation>0):
    #    print("Overfit!")
    #    break

save_path = saver.save(sess, "/tmp/Network_v1_stress_model.ckpt")
