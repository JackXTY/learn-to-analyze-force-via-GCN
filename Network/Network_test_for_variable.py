
import numpy as np
#import tensorflow as tf
#import random
import scipy
from scipy.sparse.linalg import eigsh
import pyansys
#from PIL import Image,ImageDraw


N_ = 165   # Number of nodes need output
M =  1600  # Number of train cases
N_out = 6  # Number of outputs for each node
test_size = 400 # Number of test cases

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
    print(d_mat_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def scaled_laplacian(adj):
    #print("Adjacancy Matrix")
    #print(sum(adj))

    adj_normalized = normalize_adj(adj) # =(D**-0.5)W(D**-0.5)

    print("Normalised Adjacency Matrix")
    print((adj_normalized.toarray())[0])

    laplacian = scipy.sparse.eye(adj.shape[0]) - adj_normalized

    print("Laplacian")
    print(laplacian)
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - scipy.sparse.eye(adj.shape[0])
    return scaled_laplacian

A = np.mat(A)
temp_A = scaled_laplacian(A)
new_A = temp_A.A

print("Scaled Laplacian Matirx")
print(new_A)

'''
#####################
###// read data //###
#####################
# store the inputdata(figure) in the 'all_x'
# x[i] means the result of i's test
file_object = open('train_data.txt')
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


test_file_object = open('test_data.txt')
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
'''