# This is a a GCN network model with convolution,
# which is rewrited from PointGCN.
# Let's have a look at it.

import numpy as np
import tensorflow as tf
import random
import scipy
from scipy.sparse.linalg import eigsh
import pyansys



N_ = 567   # Number of nodes need output
M =  1600  # Number of train cases
test_size = 400 # Number of test cases
batch_size = 20

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
file_object = open('train_chair_data.txt')
x = []
y_stress = []
y_displacement = []
#break_result = []
for i in range(0,M):
    temp_x = []
    temp_y_stress = []
    temp_y_displacement = []
    #break_result += [file_object.readline().split()[0]]
    for j in range(0,N_):
        temp_temp_x = []
        #temp_temp_y_stress = []  
        #temp_temp_y_displacement = []
        line = file_object.readline().split()
        for k in range(0,7):    
            temp_temp_x += [float(line[k])]
        temp_x += [temp_temp_x]
        
        temp_y_stress += [[float(line[7]),float(line[8]),float(line[9])]]
        temp_y_displacement += [[float(line[10])*1e11,float(line[11])*1e11,float(line[12])*1e11]]
    x += [temp_x]
    y_stress += [temp_y_stress]
    y_displacement += [temp_y_displacement]
file_object.close()

test_x = x
test_y_stress = y_stress
test_y_displacement = y_displacement

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
train_step_displacement = tf.train.AdagradOptimizer(0.004).minimize(loss_MSE_displacement)


def evaluate():
    #saver = tf.train.Saver()
    '''
    #saver.restore(sess, "/summer_research/chair/ckpt/Network_CNN_v2.4.2_chair_str_model.ckpt")
    total_loss_stress = 0
    for k in range(0,test_size):
        loss_now_stress = loss_MSE_stress.eval(session=sess,feed_dict = {inputPC: test_x[k],
            inputGraph: new_A, outputLabel_stress: test_y_stress[k],
            keep_prob_1: 1, keep_prob_2: 1})
        total_loss_stress += loss_now_stress
    loss_stress = total_loss_stress/test_size
    print("loss_stress: "+str(loss_stress))
    '''
    
    #saver.restore(sess, "/summer_research/chair/ckpt/Network_CNN_v2.4.2_chair_dis_model.ckpt")
    
    total_loss_displacement = 0
    
    #for k in range(int(test_size/batch_size)):
        #batch_x = []
        #batch_y_displacement = []
        #for t in range(batch_size):
            #batch_x += [test_x[count+t]]
            #batch_y_displacement += [test_y_displacement[count+t]]
        #loss_now_displacement = loss_MSE_displacement.eval(session=sess,feed_dict = {inputPC: batch_x,
        #    inputGraph: batch_A, outputLabel_displacement: batch_y_displacement,
        #    keep_prob_1: 1, keep_prob_2: 1})
        #count += batch_size
    for t in range(test_size):
        loss_now_displacement = loss_MSE_displacement.eval(session=sess,feed_dict = {inputPC: [test_x[t]],
            inputGraph: [new_A], outputLabel_displacement: [test_y_displacement[t]],
            keep_prob_1: 1, keep_prob_2: 1})    
        total_loss_displacement += loss_now_displacement
    loss_displacement = total_loss_displacement/test_size
    print("loss_displacement: "+str(loss_displacement))
    
    #return [loss_stress,loss_displacement]


def accuracy():
    '''
    saver = tf.train.Saver()
    saver.restore(sess,"/tmp/Network_CNN_v2.4.2_chair_str_model.ckpt")
    a = 0
    for k in range(0,test_size):
        stress = fc_layer_2_stress.eval(session=sess,feed_dict = {inputPC: test_x[k],
            inputGraph: new_A, outputLabel_displacement: test_y_stress[k],
            keep_prob_1: 1, keep_prob_2: 1})
        for t in range(N_):
            #a += (stress[t][0]/test_y_stress[k][t][0] + stress[t][1]/test_y_stress[k][t][1] + stress[t][2]/test_y_stress[k][t][2])
            #a +=( abs((stress[t][0]-test_y_stress[k][t][0])/test_y_stress[k][t][0]) + abs((stress[t][1]-test_y_stress[k][t][1])/test_y_stress[k][t][1]) + abs((stress[t][2]-test_y_stress[k][t][2])/test_y_stress[k][t][2]) )
            a +=( abs((abs(stress[t][0])-abs(test_y_stress[k][t][0]))/test_y_stress[k][t][0]) + abs((abs(stress[t][1])-abs(test_y_stress[k][t][1]))/test_y_stress[k][t][1]) + abs((abs(stress[t][2])-abs(test_y_stress[k][t][2]))/test_y_stress[k][t][2]) )
    print(str(a/(test_size*3*N_)))
    '''
    saver = tf.train.Saver()
    saver.restore(sess, "/summer_research/chair/ckpt/Network_CNN_v2.4.2_chair_dis_temp_model.ckpt")
    acc = 0
    #acc_list = []
    for k in range(0,int(test_size/batch_size)):
        temp = 0
        real_count = 0
        batch_x = []
        batch_y_displacement = []
        count = 0
        for t in range(batch_size):
            batch_x += [test_x[real_count+t]]
            batch_y_displacement += [test_y_displacement[real_count+t]]
        displacement = fc_layer_2_displacement.eval(session=sess,feed_dict = {inputPC: batch_x,
            inputGraph: batch_A, outputLabel_displacement: batch_y_displacement,
            keep_prob_1: 1, keep_prob_2: 1})
        for b in range(batch_size):
            for t in range(N_):
                if batch_y_displacement[b][t][0] != 0 :
                    #temp+=( abs(displacement[b][t][0]/batch_y_displacement[b][t][0]) + abs(displacement[b][t][1]/batch_y_displacement[b][t][1]) + abs(displacement[b][t][2]/batch_y_displacement[b][t][2]) )
                    #temp+=( abs((displacement[b][t][0]-batch_y_displacement[b][t][0])/batch_y_displacement[b][t][0]) + abs((displacement[b][t][1]-batch_y_displacement[b][t][1])/batch_y_displacement[b][t][1]) + abs((displacement[b][t][2]-batch_y_displacement[b][t][2])/batch_y_displacement[b][t][2]) )
                    temp+=( abs((displacement[b][t][0]-batch_y_displacement[b][t][0])/batch_y_displacement[b][t][0]) + abs((displacement[b][t][1]-batch_y_displacement[b][t][1])/batch_y_displacement[b][t][1]) + abs((displacement[b][t][2]-batch_y_displacement[b][t][2])/batch_y_displacement[b][t][2]) )
                    count += 3
        real_count += batch_size
        temp = temp/count
        #acc_list += [temp]
        acc += temp
    print(str(acc/test_size))
    #print(acc_list)



def train():
    saver = tf.train.Saver()
    saver.restore(sess,"/summer_research/chair/ckpt/Network_CNN_v2.4.2_chair_dis_temp_model.ckpt")
    #saver.restore(sess,"summer_research/chair/ckpt/Network_CNN_v2.4.2_chair_str_model.ckpt")
    #sess.run(tf.initialize_all_variables())

    for k in range(0,10):
        
        times = 600
        for j in range(0,times):

            batch_x = []
            batch_y_displacement = []
            for t in range(batch_size):
                ran = random.randint(0,M-1)
                batch_x += [x[ran]]
                batch_y_displacement += [y_displacement[ran]]
            train_step_displacement.run(feed_dict = {inputPC: batch_x, 
                inputGraph: batch_A, outputLabel_displacement: batch_y_displacement, 
                keep_prob_1: para.keep_prob_1, keep_prob_2: para.keep_prob_2})
            
            #train_step_stress.run(feed_dict = {inputPC: batch_x, 
            #    inputGraph: batch_A, outputLabel_stress: batch_y_stress, 
            #    keep_prob_1: para.keep_prob_1, keep_prob_2: para.keep_prob_2})
                
        print("Stress Times " + str(times*(k+1)) +":")
        #print("Displacement Times " + str(times*(k+1)) +":")
 
       
        evaluate()
        print()
        
        saver.save(sess,"/summer_research/chair/ckpt/Network_CNN_v2.4.2_chair_dis_temp_model.ckpt")
     
        
        #saver.save(sess,"/summer_research/chair/ckpt/Network_CNN_v2.4.2_chair_str_model.ckpt")
    #output_txt_for_test_data()

accuracy()  
#evaluate()

#train()

# external force:bottom face
# fix: behind face
