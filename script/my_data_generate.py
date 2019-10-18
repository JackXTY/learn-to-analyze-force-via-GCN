##############################
## FOR MODEL IN REAL FOLDER ##
##############################
import pyansys
import numpy as np
import os
import math


path='D:\\summer_research\\rst_data\\break_rst_7_22'
inputdata=open('D:\\summer_research\\big_data_7.22.txt','w')
i=0

# for NEW model (7.3)
force_p = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 0, 0, 0, 0, 1, 1, 1]


for file in os.listdir(path):
    
    i = i+1
    #compute the file name:node,force
    fname,ext=os.path.splitext(file)
    #print(fname)
    
    ipTable=fname.split('_')
    #print(ipTable)
    x = float(ipTable[1])
    y = float(ipTable[2])
    z = float(ipTable[3])
    #print(x)print(y)print(z)
    #x = y = z = 0.1

    #open rst file
    result = pyansys.read_binary(os.path.join(path,file))
    #print(result)
    
    nsnum, nodal_stress=result.nodal_stress(0)
    ndnum, nodal_dof=result.nodal_solution(0)

    # Number of Nodes
    N = nsnum.shape[0]
    #print("There are "+str(N)+" Nodes.")

    N_ = 165

    geometry=result.geometry
    nodespos=geometry['nodes']

    ####################
    ###    bug!!!!   ###
    ####################
    gravity=  (9.8 * 487 *0.1*0.3*0.06)/N_   #old:0.02...

    #print('File.' + str(i))

    #fix_n = 0
    #fix_p = []

    inputdata.write(ipTable[0])
    inputdata.write("\n")

    for n in range(1,N_+1):

        #inputdata.write('Node.' + str(n) + ':\n')

        if(nodal_dof[n-1][0]==0 and nodal_dof[n-1][1]==0 and nodal_dof[n-1][2]==0):
            inputdata.write(str(1))
            #fix_p += [n]
            #print("Node."+str(n)+" is fixed.")
            #fix_n += 1
        else:
            inputdata.write(str(0))        
        inputdata.write(' ')

        #inputdata.write('(position:) ')
        #nodes pos
        for j in range(0,3):
            inputdata.write(str(nodespos[n-1][j]))
            inputdata.write(' ')
        #print(str(n)+": "+str(nodespos[n-1][0])+" , "+str(nodespos[n-1][1])+" , "+str(nodespos[n-1][2]))
        
        #inputdata.write('(force:) ')
        #nodes force
        if(force_p[n-1]==1):
            inputdata.write(str(x))
            inputdata.write(' ')
            inputdata.write(str(y))
            inputdata.write(' ')
            inputdata.write(str(z + gravity))
            inputdata.write(' ')
        
        else:
            inputdata.write(str(0.0))
            inputdata.write(' ')
            inputdata.write(str(0.0))
            inputdata.write(' ')
            inputdata.write(str(gravity))
            inputdata.write(' ')

        #inputdata.write('(stress and displacement) ')
        #result stress and displacement
        for j in range(0,3):
            inputdata.write(str(nodal_stress[n-1][j]))
            inputdata.write(' ')

        for j in range(0,3):
            inputdata.write(str(nodal_dof[n-1][j]))
            inputdata.write(' ')
        inputdata.write('\n')
        
        
        #print("Node."+str(n)+": "+str(nodal_stress[n-1][0])+" , "+str(nodal_stress[n-1][1])+
        #    " , "+str(nodal_stress[n-1][2])+" , "+str(nodal_stress[n-1][3])+" , "
        #    +str(nodal_stress[n-1][4])+" , "+str(nodal_stress[n-1][5]))
        
        #print(n)

    #print("fix number = "+str(fix_n))
    #print(fix_p)
    #break
    print(i)
inputdata.close()


'''
Now is the second model.
N_ = 165 (totally 557)
Fix point (33):
[99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124,
125, 126, 127, 128, 129, 130, 131]

First model:
Fix Point (45): 
[35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187,
188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 204]
force: (29)
[25, 26, 29, 35, 49, 50, 61, 62, 69, 70, 71, 72, 155, 156, 157, 158, 159, 166, 177, 201, 202, 203, 221, 222, 223, 231, 232, 233, 234]
Node.1-72 has stress.
force_p = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
    0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
'''
