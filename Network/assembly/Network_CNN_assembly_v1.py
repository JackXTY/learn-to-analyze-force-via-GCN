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


N_ = 550   # Number of nodes need output
M =  60  # Number of train & test cases

file_object = open('assembly_adjacency.txt')
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



#####################
###// read data //###
#####################
# store the inputdata(figure) in the 'all_x'
# x[i] means the result of i's test
file_object = open('assembly_data_temp.txt')
x = []
y_stress = []
y_displacement = []
assembly_result = []

for i in range(0,M):
    temp_x = []
    temp_y_stress = []
    temp_y_displacement = []
    
    assembly_result += [[1-float(file_object.readline().split()[0])]]
    
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

#print(assembly_result)

#####################
###// Parameter //###
#####################
class Parameters():

    def __init__(self):
    	
        self.neighborNumber = 8
        self.outputClassN = 6
        self.pointNumber = N_

        self.gcn_1_filter_n = 560
        self.gcn_2_filter_n = 140
        self.gcn_3_filter_n = 280
        #self.gcn_4_filter_n = 140

        self.fc_1_n = 140
        self.chebyshev_1_Order = 6
        self.chebyshev_2_Order = 5
        self.chebyshev_3_Order = 4
        self.chebyshev_4_Order = 3
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
    initial = tf.truncated_normal(shape=shape, mean=0, stddev=0.001)
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
        #output = tf.reshape(output, [-1, pointNumber, outputFeatureN])
        output = tf.reshape(output, [pointNumber, outputFeatureN])
        chebyOutput.append(output)

    gcnOutput = tf.add_n(chebyOutput) + biasWeight
    gcnOutput = tf.nn.relu(gcnOutput)
    return gcnOutput


#fully connected layer without relu activation
def fullyConnected(features, inputFeatureN, outputFeatureN):
    weightFC = weightVariables([inputFeatureN, outputFeatureN], name='weight_fc')
    biasFC = weightVariables([outputFeatureN], name='bias_fc')
    outputFC = tf.matmul(features,weightFC)+biasFC
    return outputFC



inputPC = tf.placeholder(tf.float32, [para.pointNumber, 7]) # PC = Point Cloud
inputGraph = tf.placeholder(tf.float32, [para.pointNumber , para.pointNumber])
outputLabel_stress = tf.placeholder(tf.float32, [para.pointNumber, 3])
outputLabel_displacement = tf.placeholder(tf.float32, [para.pointNumber, 3])
outputLabel_result = tf.placeholder(tf.float32,[1])

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


# concatenate global features
#globalFeatures = gcn_3_pooling
globalFeatures = tf.concat([gcn_1_output, gcn_2_output], axis=1)
#globalFeatures = tf.concat([globalFeatures, gcn_3_output], axis=1)
globalFeatures = tf.nn.dropout(globalFeatures, keep_prob=keep_prob_2)
print("The global feature is {}".format(globalFeatures))
#globalFeatureN = para.gcn_2_filter_n*2
#globalFeatureN = (para.gcn_1_filter_n + para.gcn_2_filter_n)*2 
globalFeatureN = para.gcn_1_filter_n + para.gcn_2_filter_n # + para.gcn_3_filter_n 


# fully connected layer 1
fc_layer_1 = fullyConnected(globalFeatures, inputFeatureN=globalFeatureN, outputFeatureN=para.fc_1_n)
fc_layer_1 = tf.nn.relu(fc_layer_1)
fc_layer_1 = tf.nn.dropout(fc_layer_1, keep_prob=keep_prob_2)
print("The output of the first fc layer is {}".format(fc_layer_1))



fc_layer_2_result = fullyConnected(fc_layer_1, inputFeatureN=para.fc_1_n, outputFeatureN=1)
new_fc_layer_2_result = ((fc_layer_2_result*1e6)%10)/10
#soft_result = tf.nn.softmax(new_fc_layer_2_result)
#result = tf.reduce_mean(soft_result)
result = tf.reduce_mean(new_fc_layer_2_result)

#print(result)
#print(outputLabel_result)
#loss_MSE_result = tf.losses.mean_squared_error(labels=outputLabel_result,predictions=result)
loss_MSE_result = (result-outputLabel_result[0])**2
#train_step = tf.train.AdamOptimizer(1e-4).minimize(loss_MSE) 
train_step_result = tf.train.AdagradOptimizer(0.001).minimize(loss_MSE_result)



# fully connected layer 2
fc_layer_2_stress = fullyConnected(fc_layer_1, inputFeatureN=para.fc_1_n, outputFeatureN=3)
print("The output of the second fc layer for stress is {}".format(fc_layer_2_stress))
print()

loss_MSE_stress = tf.losses.mean_squared_error(labels=outputLabel_stress,predictions=fc_layer_2_stress)
#train_step = tf.train.AdamOptimizer(1e-4).minimize(loss_MSE) 
train_step_stress = tf.train.AdagradOptimizer(0.001).minimize(loss_MSE_stress)


#fc_layer_2_displacement = fullyConnected_displacement(fc_layer_1, inputFeatureN=para.fc_1_n, outputFeatureN=3)
fc_layer_2_displacement = fullyConnected(fc_layer_1, inputFeatureN=para.fc_1_n, outputFeatureN=3)
print("The output of the second fc layer for displacement is {}".format(fc_layer_2_displacement))
print()

loss_MSE_displacement = tf.losses.mean_squared_error(labels=outputLabel_displacement,predictions=fc_layer_2_displacement)
#train_step = tf.train.AdamOptimizer(1e-4).minimize(loss_MSE) 
train_step_displacement = tf.train.AdagradOptimizer(0.001).minimize(loss_MSE_displacement)




#cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits( labels = outputLabel, logits = fc_layer_2 )
#train_step = tf.train.AdamOptimizer(1e-6).minimize(cross_entropy) 
#train_step = tf.train.GradientDescentOptimizer(1e-9).minimize(cross_entropy) #(99.3%) times:13900 (in old version of data)
#train_step = tf.train.AdadeltaOptimizer(1e-3).minimize(cross_entropy)
#train_step = tf.train.AdagradOptimizer(1e-4).minimize(cross_entropy)


def evaluate_result(): 
    accuracy = 0
    test_size = M
    loss = 0
    for k in range(0,test_size):
        result_now = result.eval(session=sess,feed_dict = {inputPC: x[k],
            inputGraph: new_A, outputLabel_result: assembly_result[k],
            keep_prob_1: 1, keep_prob_2: 1})
        if (result_now>0.5 and assembly_result[k]==[1]) or (result_now<0.5 and assembly_result[k]==[0]):
            accuracy += 1
        loss_now = loss_MSE_result.eval(session=sess,feed_dict = {inputPC: x[k],
            inputGraph: new_A, outputLabel_result: assembly_result[k],
            keep_prob_1: 1, keep_prob_2: 1})
        loss += loss_now
        #print("result:"+str(k)+": "+str(result.eval(session=sess,feed_dict = {inputPC: x[k],
        #    inputGraph: new_A, outputLabel_result: assembly_result[k],
        #    keep_prob_1: 1, keep_prob_2: 1})))
    accuracy = accuracy/test_size
    print("accuracy: "+str(accuracy))
    print("loss:"+str(loss))

    return accuracy


def evaluate_stress(): 
    total_loss_stress = 0
    test_size = M
    for k in range(0,test_size):
        loss_now_stress = loss_MSE_stress.eval(session=sess,feed_dict = {inputPC: x[k],
            inputGraph: new_A, outputLabel_stress: y_stress[k],
            keep_prob_1: 1, keep_prob_2: 1})
        total_loss_stress += loss_now_stress
    loss_stress = total_loss_stress/test_size
    print("loss_stress: "+str(loss_stress))
    return loss_stress
    
def evaluate_displacement():
    total_loss_displacement = 0
    test_size = M
    for k in range(0,test_size):
        loss_now_displacement = loss_MSE_displacement.eval(session=sess,feed_dict = {inputPC: x[k],
            inputGraph: new_A, outputLabel_displacement: y_displacement[k],
            keep_prob_1: 1, keep_prob_2: 1})
        total_loss_displacement += loss_now_displacement
    loss_displacement = total_loss_displacement/test_size
    print("loss_displacement: "+str(loss_displacement))
    return loss_displacement
    
'''
def output_txt_for_test_data():
    write_file_object = open('predict_assembly_data.txt','w')
    test_size = M
    for k in range(0,test_size):
        predict_stress = fc_layer_2_stress.eval(session=sess,feed_dict = {inputPC: x[k],
            inputGraph: new_A, outputLabel_stress: y_stress[k],
            keep_prob_1: 1, keep_prob_2: 1})
        predict_displacement = fc_layer_2_displacement.eval(session=sess,feed_dict = {inputPC: x[k],
            inputGraph: new_A, outputLabel_displacement: y_displacement[k],
            keep_prob_1: 1, keep_prob_2: 1})
        for t in range(0,N_):
            write_file_object.write(str(predict_stress[t][0])+' '+str(predict_stress[t][1])+' '+str(predict_stress[t][2])+' '+
                str(predict_displacement[t][0]/1e11)+' '+str(predict_displacement[t][1]/1e11)+' '+str(predict_displacement[t][2]/1e11)+'\n')
    write_file_object.close()
'''

def train_result():
    saver = tf.train.Saver()
    #saver.restore(sess, "/tmp/Network_CNN_assembly_v1_model.ckpt")
    sess.run(tf.initialize_all_variables())

    for k in range(0,40):
        times = 100
        for j in range(0,times):
            ran = random.randint(0,M-1)
            train_step_result.run(feed_dict = {inputPC: x[ran], 
                inputGraph: new_A, outputLabel_result: assembly_result[ran], 
                keep_prob_1: para.keep_prob_1, keep_prob_2: para.keep_prob_2})
        print("Times " + str(times*(k+1)) +":")

        # evaluate
        evaluate_result()
        print()
    saver.save(sess, "/tmp/Network_CNN_assembly_v1_model.ckpt")

def train_stress():
    saver = tf.train.Saver()
    #saver.restore(sess, "/tmp/Network_CNN_assembly_v1_stress_model.ckpt")
    sess.run(tf.initialize_all_variables())

    for k in range(0,10):
        times = 500
        for j in range(0,times):
            ran = random.randint(0,M-1)
            train_step_stress.run(feed_dict = {inputPC: x[ran], 
                inputGraph: new_A, outputLabel_stress: y_stress[ran], 
                keep_prob_1: para.keep_prob_1, keep_prob_2: para.keep_prob_2})
        print("Stress Times " + str(times*(k+1)/5) +":")

        # evaluate
        evaluate_stress()
        print()
    saver.save(sess, "/tmp/Network_CNN_assembly_v1_stress_model.ckpt")


def train_displacement():
    saver = tf.train.Saver()
    #saver.restore(sess, "/tmp/Network_CNN_assembly_v1_displacement_model.ckpt")
    sess.run(tf.initialize_all_variables())
    for k in range(0,10):
        times = 500
        for j in range(0,times):
            ran = random.randint(0,M-1)
            train_step_displacement.run(feed_dict = {inputPC: x[ran], 
                inputGraph: new_A, outputLabel_displacement: y_displacement[ran], 
                keep_prob_1: para.keep_prob_1, keep_prob_2: para.keep_prob_2})

        print("Displacement Times " + str(times*(k+1)) +":")
        evaluate_displacement()
        print()
    saver.save(sess, "/tmp/Network_CNN_assembly_v1_displacement_model.ckpt")


train_result()
#output_txt_for_test_data()




def run_with_input(x,y,z):
    gravity=  (9.8 * 487 *0.1*0.3*0.06)/N_

    input_x = x[0][:]
    #input_y_stress = y_stress[0][:]
    #input_y_displacement = y_displacement[0][:]
    
    for n in range(0,N_):
        if input_x[n][4]!=0 and input_x[n][5]!=0:
            input_x[n][4] = x
            input_x[n][5] = y
            input_x[n][6] = z + gravity
    
    #print(input_x)
    saver = tf.train.Saver()
    saver.restore(sess, "/tmp/Network_CNN_assembly_v1_model.ckpt") # stress
    predict_stress = fc_layer_2_stress.eval(session=sess,feed_dict = {inputPC: x,
        inputGraph: new_A, outputLabel_stress: y_stress,
        keep_prob_1: 1, keep_prob_2: 1})

    saver.restore(sess, "/tmp/Network_CNN_assembly_v1_model.ckpt") # displacement
    predict_displacement = fc_layer_2_displacement.eval(session=sess,feed_dict = {inputPC: x,
        inputGraph: new_A, outputLabel_displacement: y_displacement,
        keep_prob_1: 1, keep_prob_2: 1})
    #evaluate()
    predict_displacement = [[_i[0]/1e11]+[_i[1]/1e11]+[_i[2]/1e11] for _i in predict_displacement]
    
    print(predict_stress)
    print(predict_displacement)
    rst_path = '1_-22.228909891907985_-57.001098696703046_84.36130192338905.rst'
    result=pyansys.read_binary(rst_path)
    result.plot_nodal_solution(0)
    

    a = np.array(predict_stress)
    scalars = a[:, :3]
    scalars = (scalars*scalars).sum(1)**0.5
    B = np.zeros((557,3))
    for i in range(165):
        B[i][0]=scalars[i]
    pngpath = 'D:\\summer_research\\png\\test.png'
    result._plot_point_scalars(B.transpose(),screenshot=pngpath,interactive=True,stitle='Label Stress')

    TINT_COLOR = (0, 0, 0)
    image = Image.open(pngpath)
    image = image.convert("RGBA")
    tmp = Image.new('RGBA', image.size, TINT_COLOR+(0,))
    
    draw = ImageDraw.Draw(tmp)
    draw.polygon(((680,140),(665,330),(750,350),(775,180)), fill=TINT_COLOR+(127,))#topleft bottom left bottom right top right
    image = Image.alpha_composite(image, tmp)
    image = image.convert("RGB")
    circle='D:\\summer_research\\png\\testcircle'+str(i)+'.png'
    image.save(circle)

#train()
#run_with_input(-22.228909891907985,-57.001098696703046,84.36130192338905)
#run_with_input(0,0,200)
# external force:bottom face
# fix: behind face

# draw.line((100,200, 150,300), fill=128)