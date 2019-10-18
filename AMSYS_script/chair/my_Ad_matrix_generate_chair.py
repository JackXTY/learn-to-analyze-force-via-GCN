import pyansys
import numpy as np
import os
import math

'''
Totally 1211 nodes, so there should be at least 2400 edges around(considering iteration).

'''


path='D:\\summer_research\\chair\\chair_rst'
inputdata=open('D:\\summer_research\\chair\\assembly_chair_adjacency.txt','w')
j=0


for file in os.listdir(path):
    
    j = j+1
    #compute the file name:node,force
    fname,ext=os.path.splitext(file)
    
    
    ipTable=fname.split('_')
    
    #x=float(ipTable[1])
    #y=float(ipTable[2])
    #z=float(ipTable[3])
  
    print(fname)
    #print(x)
    #print(y)
    #print(z)
    
    #open rst file
    result = pyansys.read_binary(os.path.join(path,file))
    print(result)
    
    nsnum, nodal_stress = result.nodal_stress(0)

       
    #ndnum, nodal_dof=result.nodal_solution(0)
    
    
    # Number of Nodes
    #N = nsnum.shape[0]
    #print("There are "+str(N)+" Nodes.")
    N = 567
    
    #nodes pos
    geometry=result.geometry
    nodespos=geometry['nodes']
    #print(nodespos)

    print('File.' + str(j))
    
    
    result = [[0 for _i in range(0,N)] for _j in range(0,N)]
    
    for n in range(0,N):
        
        #print(str(n)+": "+str(nodespos[n][0])+" , "+str(nodespos[n][1])+" , "+str(nodespos[n][2]))
        xd = 0
        yd = 0
        zd = 0
        for i in range(0,N):
            temp_xd = abs(nodespos[i][0] - nodespos[n][0])
            temp_yd = abs(nodespos[i][1] - nodespos[n][1])
            temp_zd = abs(nodespos[i][2] - nodespos[n][2])
            if (temp_xd<xd or xd==0) and temp_xd>xd*0.3:
                xd = temp_xd
            if (temp_yd<yd or yd==0) and temp_yd>yd*0.3:
                yd = temp_yd
            if (temp_zd<zd or zd==0) and temp_zd>zd*0.3:
                zd = temp_zd   
        
        #print()print("xd="+str(xd))print("yd="+str(yd))print("zd="+str(zd))print()
        
        for i in range(0,N):
            _xd = abs(nodespos[i][0] - nodespos[n][0])
            _yd = abs(nodespos[i][1] - nodespos[n][1])
            _zd = abs(nodespos[i][2] - nodespos[n][2])
            
            if  (  ( _xd<xd*1.2 and _xd>xd*0.01 and _yd<yd*0.25 and _zd<zd*0.25)
                or ( _yd<yd*1.2 and _yd>yd*0.01 and _xd<xd*0.25 and _zd<zd*0.25)
                or ( _zd<zd*1.2 and _zd>zd*0.01 and _xd<xd*0.25 and _yd<yd*0.25) ):
                result[n][i] = 1
                result[i][n] = 1

                        
    break

#print(len(result))
#print(len(result[0]))
count=0
for n in range(0,N):
    for i in range(0,N):
        if (result[n][i]==1):
            inputdata.write(str(n)+" "+str(i)+"\n")
            count += 1
print("Edges: "+str(count))    
inputdata.close()
