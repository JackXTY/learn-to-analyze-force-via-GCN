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
M =  1600  # Number of train cases
test_size = 400 # Number of test cases
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
train_step_displacement = tf.train.AdagradOptimizer(0.005).minimize(loss_MSE_displacement)



def run_with_input(x,y,z):
    V = 0.1125*(0.3*0.1-0.16*0.04)
    gravity=  (9.8 * 487 * V)/N_ 

    input_x = [test_x[0][:]]
    #input_y_stress = test_y_stress[0][:]
    #input_y_displacement = test_y_displacement[0][:]
    
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
    
    
    saver = tf.train.Saver()
    #saver.restore(sess, "/summer_research/chair/ckpt/Network_CNN_v2.4.2_chair_str_model.ckpt") # stress
    #predict_stress = fc_layer_2_stress.eval(session=sess,feed_dict = {inputPC: input_x, inputGraph: new_A, keep_prob_1: 1, keep_prob_2: 1})
    

    saver.restore(sess, "/summer_research/chair/ckpt/Network_CNN_v2.4.2_chair_dis_temp_model.ckpt") # displacement
    predict_displacement = fc_layer_2_displacement.eval(session=sess,feed_dict = {inputPC: input_x,
        inputGraph: batch_A, 
        #outputLabel_displacement: input_y_displacement,
        keep_prob_1: 1, keep_prob_2: 1})
    #evaluate()
    predict_displacement = [[_i[0]/1e11]+[_i[1]/1e11]+[_i[2]/1e11] for _i in predict_displacement[0]]
    
    result = pyansys.read_binary("RST.rst")
    '''
    a = np.array(predict_stress)
    scalars = a[:, :3]
    scalars = (scalars*scalars).sum(1)**0.5
    B = np.zeros((557,3))
    for i in range(165):
        B[i][0]=scalars[i]

    #mi = min(scalars)
    ma = max(scalars)
    plot_range = [round(500,-2),round(ma*0.25,-3)]

    #print(predict_stress)
    #print(predict_displacement)
    #rst_path = '1_-22.228909891907985_-57.001098696703046_84.36130192338905.rst'
    
    
    label_pngpath = 'D:\\summer_research\\png\\stress\\original_stress'+str(x)+'_'+str(y)+'_'+str(z)+'.png'
    #result.plot_nodal_solution(rnum = 0,screenshot=pngpath,interactive=False)
    result.plot_principal_nodal_stress(rnum=0,stype='INT',screenshot=label_pngpath,interactive=False,rng=plot_range)

    
    predict_pngpath = 'D:\\summer_research\\png\\stress\\prediction_stress'+str(x)+'_'+str(y)+'_'+str(z)+'.png'
    result._plot_point_scalars(B.transpose(),screenshot=predict_pngpath,interactive=False,stitle='Label Stress',rng=plot_range)

    TINT_COLOR = (0, 0, 0)
    image = Image.open(predict_pngpath)
    image = image.convert("RGBA")
    tmp = Image.new('RGBA', image.size, TINT_COLOR+(0,))
    
    
    draw = ImageDraw.Draw(tmp)

    draw.line([(518,388),(518+x_,388+y_)],fill=TINT_COLOR+(127,),width=5)
    draw.polygon([(518+x_*1.1,388+y_*1.1),(518+x_+y_*0.1,388+y_-x_*0.1),(518+x_-y_*0.1,388+y_+x_*0.1)],fill=TINT_COLOR+(127,))
    draw.polygon(((850,390),(830,497),(393,250),(383,158)), fill=TINT_COLOR+(127,))#topleft bottom left bottom right top right
    image = Image.alpha_composite(image, tmp)
    image = image.convert("RGB")
    circle='D:\\summer_research\\png\\stress\\prediction_circle_stress'+str(x)+'_'+str(y)+'_'+str(z)+'.png'
    image.save(circle)

    predict_png = Image.open(predict_pngpath)
    label_png = Image.open(label_pngpath)
    combined_png=Image.new('RGB', (1024*2,745))
    combined_png.paste(predict_png,(0,0))
    combined_png.paste(label_png,(1024,0))
    filename='D:\\summer_research\\png\\stress\\combine_stress'+str(x)+'_'+str(y)+'_'+str(z)+'.png'
    combined_png.save(filename)
    '''

    ####################
    ####################
    
    a = np.array(predict_displacement)
    #print(a)
    #print(len(a))
    #print(len(a[0]))
    scalars = a[ :, :3]
    scalars = (scalars*scalars).sum(1)**0.5
    B = np.zeros((N_,3))
    for i in range(N_):
        B[i][0]=scalars[i]

    plot_range = [round(min(scalars+1e-9),9),round(max(scalars),9)]

    #label_pngpath = 'D:\\summer_research\\chair\\png\\displacement\\original_displacement'+str(x)+'_'+str(y)+'_'+str(z)+'.png'
    #result.plot_nodal_solution(rnum = 0,screenshot=pngpath,interactive=False)
    #result.plot_nodal_solution(rnum=0,screenshot=label_pngpath,interactive=False,rng=plot_range)

    
    predict_pngpath = 'D:\\summer_research\\chair\\png\\displacement_2\\prediction_displacement_'+str(x)+'_'+str(y)+'_'+str(z)+'.png'
    result._plot_point_scalars(B.transpose(),screenshot=predict_pngpath,interactive=False,stitle='Label displacement',rng=plot_range)

    TINT_COLOR = (0, 0, 0)
    image = Image.open(predict_pngpath)
    image = image.convert("RGBA")
    tmp = Image.new('RGBA', image.size, TINT_COLOR+(0,))
    draw = ImageDraw.Draw(tmp)
    draw.line([(502,263),(502+x_,263+y_)],fill=TINT_COLOR+(127,),width=10)
    draw.polygon([(502+x_*1.2,263+y_*1.2),(502+x_+y_*0.2,263+y_-x_*0.2),(502+x_-y_*0.2,263+y_+x_*0.2)],fill=TINT_COLOR+(127,))
    #draw.polygon(((850,390),(830,497),(393,250),(383,158)), fill=TINT_COLOR+(127,))#topleft bottomleft bottomright topright
    image = Image.alpha_composite(image, tmp)
    image = image.convert("RGB")
    circle='D:\\summer_research\\chair\\png\\displacement_2\\prediction_circle_displacements_'+str(x)+'_'+str(y)+'_'+str(z)+'.png'
    image.save(circle)
    '''
    #predict_png = Image.open(predict_pngpath)
    label_png = Image.open(label_pngpath)
    label_png = label_png.convert("RGBA")
    tmp = Image.new('RGBA', label_png.size, TINT_COLOR+(0,))
    draw = ImageDraw.Draw(tmp)
    draw.line([(502,263),(502+x_,263+y_)],fill=TINT_COLOR+(127,),width=10)
    draw.polygon([(502+x_*1.2,263+y_*1.2),(502+x_+y_*0.2,263+y_-x_*0.2),(502+x_-y_*0.2,263+y_+x_*0.2)],fill=TINT_COLOR+(127,))
    #draw.polygon(((850,390),(830,497),(393,250),(383,158)), fill=TINT_COLOR+(127,))#topleft bottomleft bottomright topright
    label_png = Image.alpha_composite(label_png, tmp)
    label_png = label_png.convert("RGB")
    combined_png=Image.new('RGB', (1024*2,745))
    combined_png.paste(image,(0,0))
    combined_png.paste(label_png,(1024,0))
    filename='D:\\summer_research\\chair\\png\\displacement\\combine_displacement'+str(x)+'_'+str(y)+'_'+str(z)+'.png'
    combined_png.save(filename)
    '''
'''
path = 'D:\\summer_research\\chair\\test_rst'
size = 5
path_list = []
N = 0
i = 0
for file in os.listdir(path):
    i+=1
    probability = random.random()
    
    #print(probability)
    if probability < (size/400)+0.05 :
        path_list += [path+'\\'+file]
        N += 1
        #print(i)
        #print(path_list[N-1])
    if N >= size:  
        break
for i in range(N):
    ip = path_list[i].split('_')
    #print(ip)
    #print(ip[6][:-4])
    run_with_input(float(ip[3]),float(ip[4]),float(ip[5][:-4]),path_list[i])
'''
while True:
    a_ = float(input("X:"))
    b_ = float(input("Y:"))
    c_ = float(input("Z:"))
    run_with_input(a_,b_,c_)
# external force:bottom face
# fix: behind face

# draw.line((100,200, 150,300), fill=128)