# This is a a GCN network model with convolution,
# which is rewrited from PointGCN.
# Let's have a look at it.

import numpy as np
import tensorflow as tf
import random
import scipy
from scipy.sparse.linalg import eigsh
import pyansys
from PIL import Image,ImageDraw
import os

N_ = 567   # Number of nodes need output
M =  240  # Number of train cases
test_size = 240 # Number of test cases
batch_size = 1

file_object = open('chair_adjacency.txt')
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
    adj_normalized = normalize_adj(adj) # =(D**-0.5)W(D**-0.5)
    laplacian = scipy.sparse.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - scipy.sparse.eye(adj.shape[0])
    return scaled_laplacian

A = np.mat(A)
temp_A = scaled_laplacian(A)
new_A = temp_A.A
batch_A = [new_A for i in range(batch_size)]


#####################
###// read data //###
#####################
# store the inputdata(figure) in the 'all_x'
# x[i] means the result of i's test

test_file_object = open('test_chair_data.txt')
test_x = []
test_y_stress = []
test_y_displacement = []
for i in range(0,test_size):
    temp_x = []
    temp_y_stress = []
    temp_y_displacement = []
    for j in range(0,N_):
        temp_temp_x = []
        temp_temp_y_stress = []
        temp_temp_y_displacement = []    
        line = test_file_object.readline().split()
        
        for k in range(0,7):
            temp_temp_x += [float(line[k])]
        temp_x += [temp_temp_x]
        for k in range(7,10):
            temp_temp_y_stress += [float(line[k])]
        for k in range(10,13):
            temp_temp_y_displacement += [float(line[k])*1e11]
        temp_y_stress += [temp_temp_y_stress]
        temp_y_displacement += [temp_temp_y_displacement]
    test_x += [temp_x]
    test_y_stress += [temp_y_stress]
    test_y_displacement += [temp_y_displacement]
test_file_object.close()

#y_stress = np.array(y_stress)
#y_displacement = np.array(y_displacement)
#test_y_stress = np.array(test_y_stress)
#test_y_displacement = np.array(test_y_displacement)
#print(y_stress.shape)
#print(y_displacement)
#print(test_y_stress.shape)
#print(test_y_displa'tcement.shape)

#####################
###// Parameter //###
#####################
class Parameters():

    def __init__(self):
    	
        self.neighborNumber = 8
        self.outputClassN = 6
        self.pointNumber = N_

        self.gcn_1_filter_n = 70
        self.gcn_2_filter_n = 35
        #self.gcn_3_filter_n = 35
        #self.gcn_4_filter_n = 70

        self.fc_1_n = 35
        self.chebyshev_1_Order = 3
        self.chebyshev_2_Order = 2
        self.chebyshev_3_Order = 2
        #self.chebyshev_4_Order = 3
        self.keep_prob_1 = 1 #0.9 original
        self.keep_prob_2 = 1
        
para = Parameters()


#####################
###// Main Part //###
#####################
# activation function: tanh & identity(y=x)
# aggregate = D**-0.5 * A *  D**-0.5 * X

# prepare the value
sess = tf.InteractiveSession()

def weightVariables(shape, name):
    initial = tf.truncated_normal(shape=shape, mean=0, stddev=0.0001)
    #initial = tf.ones(shape=shape)
    return tf.Variable(initial, name=name)


def chebyshevCoefficient(chebyshevOrder, inputNumber, outputNumber):
    chebyshevWeights = dict()
    for i in range(chebyshevOrder):
        initial = tf.truncated_normal(shape=[inputNumber, outputNumber], mean=0, stddev=0.001)
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
        output = tf.reshape(output, [-1, pointNumber, outputFeatureN])
        #output = tf.reshape(output, [pointNumber, outputFeatureN])
        chebyOutput.append(output)

    gcnOutput = tf.add_n(chebyOutput) + biasWeight
    gcnOutput = tf.nn.relu(gcnOutput)
    return gcnOutput


#fully connected layer without relu activation
def fullyConnected(features, inputFeatureN, outputFeatureN):
    features = tf.reshape(features,[-1,inputFeatureN*N_])
    weightFC = weightVariables([inputFeatureN*N_, outputFeatureN*N_], name='weight_fc')
    biasFC = weightVariables([outputFeatureN*N_], name='bias_fc')
    outputFC = tf.reshape(tf.matmul(features,weightFC)+biasFC,[-1,N_,outputFeatureN])
    return outputFC


inputPC = tf.placeholder(tf.float32, [None, para.pointNumber, 7]) # PC = Point Cloud
inputGraph = tf.placeholder(tf.float32, [None, para.pointNumber , para.pointNumber])
outputLabel_stress = tf.placeholder(tf.float32, [None, para.pointNumber, 3])
outputLabel_displacement = tf.placeholder(tf.float32, [None, para.pointNumber, 3])

#scaledLaplacian = tf.reshape(inputGraph, [-1, para.pointNumber, para.pointNumber])
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
print("\nThe output of the first gcn layer is {}".format(gcn_1_output))
#print (gcn_1_pooling)

# gcn_layer_2
gcn_2 = gcnLayer(gcn_1_output, scaledLaplacian, pointNumber=para.pointNumber, inputFeatureN=para.gcn_1_filter_n,
                    outputFeatureN=para.gcn_2_filter_n,
                    chebyshev_order=para.chebyshev_2_Order)
gcn_2_output = tf.nn.dropout(gcn_2, keep_prob=keep_prob_1)
print("The output of the second gcn layer is {}".format(gcn_2_output))
'''
#gcn_layer_3
gcn_3 = gcnLayer(gcn_2_output, scaledLaplacian, pointNumber=para.pointNumber, inputFeatureN=para.gcn_2_filter_n,
                    outputFeatureN=para.gcn_3_filter_n,
                    chebyshev_order=para.chebyshev_3_Order)
gcn_3_output = tf.nn.dropout(gcn_3, keep_prob=keep_prob_1)
print("The output of the second gcn layer is {}".format(gcn_3_output))
'''
'''
#gcn_layer_4
gcn_4 = gcnLayer(gcn_3_output, scaledLaplacian, pointNumber=para.pointNumber, inputFeatureN=para.gcn_3_filter_n,
                    outputFeatureN=para.gcn_4_filter_n,
                    chebyshev_order=para.chebyshev_4_Order)
gcn_4_output = tf.nn.dropout(gcn_4, keep_prob=keep_prob_1)
print("The output of the second gcn layer is {}".format(gcn_4_output))
'''
'''
# concatenate global features
#globalFeatures = gcn_3_pooling
globalFeatures = tf.concat([gcn_1_output, gcn_2_output], axis=2)
#globalFeatures = tf.concat([globalFeatures, gcn_3_output], axis=2)
#globalFeatures = tf.concat([globalFeatures, gcn_4_output], axis=2)
globalFeatures = tf.nn.dropout(globalFeatures, keep_prob=keep_prob_2)
print("The global feature is {}".format(globalFeatures))
#globalFeatureN = para.gcn_2_filter_n*2
#globalFeatureN = (para.gcn_1_filter_n + para.gcn_2_filter_n)*2 
globalFeatureN = para.gcn_1_filter_n + para.gcn_2_filter_n #+ para.gcn_3_filter_n + para.gcn_4_filter_n
'''
'''
# fully connected layer 1
#fc_layer_1 = fullyConnected(globalFeatures, inputFeatureN=globalFeatureN, outputFeatureN=para.fc_1_n)
fc_layer_1 = fullyConnected(gcn_2_output, inputFeatureN=para.gcn_2_filter_n, outputFeatureN=para.fc_1_n)
fc_layer_1 = tf.nn.relu(fc_layer_1)
fc_layer_1 = tf.nn.dropout(fc_layer_1, keep_prob=keep_prob_2)
print("The output of the first fc layer is {}".format(fc_layer_1))
'''

# fully connected layer 2
'''
Stress

fc_layer_2_stress = fullyConnected(fc_layer_1, inputFeatureN=para.fc_1_n, outputFeatureN=3)
print("The output of the second fc layer for stress is {}".format(fc_layer_2_stress))
print()
loss_MSE_stress = tf.losses.mean_squared_error(labels=outputLabel_stress,predictions=fc_layer_2_stress)
#train_step = tf.train.AdamOptimizer(1e-4).minimize(loss_MSE) 
train_step_stress = tf.train.AdagradOptimizer(0.002).minimize(loss_MSE_stress)
'''

'''
Displacement
'''
#fc_layer_2_displacement = fullyConnected_displacement(fc_layer_1, inputFeatureN=para.fc_1_n, outputFeatureN=3)
#fc_layer_2_displacement = fullyConnected(fc_layer_1, inputFeatureN=para.fc_1_n, outputFeatureN=3)
fc_layer_2_displacement = fullyConnected(gcn_2_output, inputFeatureN=para.gcn_2_filter_n, outputFeatureN=3)
print("The output of the second fc layer for displacement is {}".format(fc_layer_2_displacement))
print()
loss_MSE_displacement = tf.losses.mean_squared_error(labels=outputLabel_displacement,predictions=fc_layer_2_displacement)
#train_step = tf.train.AdamOptimizer(1e-4).minimize(loss_MSE) 
train_step_displacement = tf.train.AdagradOptimizer(0.005).minimize(loss_MSE_displacement)


V = 0.1125*(0.3*0.1-0.16*0.04)
gravity=  (9.8 * 487 * V)/N_ 

input_x = [test_x[0][:]]
#input_y_stress = test_y_stress[0][:]
#input_y_displacement = test_y_displacement[0][:]

saver = tf.train.Saver()

path = 'D:\\summer_research\\chair\\test_rst'

all_dis_error = []

i_file = 0
ave = 0
for file in os.listdir(path):
    print(i_file)
    i_file += 1
    
    result = pyansys.read_binary(os.path.join(path,file))
    ndnum, nodal_dof=result.nodal_solution(0)

    ip = file.split('_')
    #print(ip)
    #print(ip[6][:-4])
    #print(ip)
    x = float(ip[1])
    y = float(ip[2])
    z = float(ip[3][:-4])
    
    x_ = -x/1.41421 + y/1.41421
    y_ = z - x/1.41421 - y/1.41421
    s_ = (x_**2+y_**2)**0.5
    x_ = x_*200/s_
    y_ = y_*200/s_
    for n in range(0,N_):
        if input_x[0][n][4]!=0 and input_x[0][n][5]!=0:
            input_x[0][n][4] = x
            input_x[0][n][5] = y
            input_x[0][n][6] = z + gravity
    
  
    #saver.restore(sess, "/summer_research/chair/ckpt/Network_CNN_v2.4.2_chair_str_model.ckpt") # stress
    #predict_stress = fc_layer_2_stress.eval(session=sess,feed_dict = {inputPC: input_x, inputGraph: new_A, keep_prob_1: 1, keep_prob_2: 1})
    

    saver.restore(sess, "/summer_research/chair/ckpt/Network_CNN_v2.4.2_chair_dis_temp_model.ckpt") # displacement
    predict_displacement = fc_layer_2_displacement.eval(session=sess,feed_dict = {inputPC: input_x,
        inputGraph: batch_A, 
        #outputLabel_displacement: input_y_displacement,
        keep_prob_1: 1, keep_prob_2: 1})
    #evaluate()
    predict_displacement = [[_i[0]/1e11]+[_i[1]/1e11]+[_i[2]/1e11] for _i in predict_displacement[0]]
    
    
    
    a = np.array(predict_displacement)
    #print(a)
    #print(len(a))
    #print(len(a[0]))
    scalars = a[ :, :3]
    scalars = (scalars*scalars).sum(1)**0.5
    B = np.zeros((N_,3))
    nodal_dof = nodal_dof[:,:3]
    nodal_dof = (nodal_dof*nodal_dof).sum(1)**0.5
    ave += sum(nodal_dof)/N_
    for i in range(N_):
        B[i][0] = scalars[i] - nodal_dof[i]
    all_dis_error += [B]
    

plot_range = [0,ave/test_size]

print
error = [0 for _i in range(N_)]
for t in range(N_):
    err = 0
    for j in range(test_size):
        err += all_dis_error[j][t][0]
    error[t] = err/test_size
E = np.zeros((N_,3))
for i in range(N_):
    E[i][0] = error[t]
error_pngpath = 'D:\\summer_research\\chair\\png\\error_plot.png'
result._plot_point_scalars(E.transpose(),screenshot=error_pngpath,interactive=False,stitle='Error Plot',rng=plot_range)

