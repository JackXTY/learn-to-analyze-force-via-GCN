##############################
## FOR MODEL IN REAL FOLDER ##
##############################
import pyansys
import numpy as np
import os
import math

# Totally 615 nodes


path='D:\\summer_research\\T\\rst'

for file in os.listdir(path):
    result = pyansys.read_binary(os.path.join(path,file))
    print(result)
    geometry=result.geometry
    nodespos=geometry['nodes']
    nsnum, nodal_stress=result.nodal_stress(0)
    ndnum, nodal_dof=result.nodal_solution(0)
    print()
    
    li = []
    N = 615
    #for i in range(N):
    #    if nodal_stress[i][0]>=0 or nodal_stress[i][0]<0:
    #        li += [i]
    #for i in range(N):
    #    if nodespos[i][1]>=0.44 :
    #        print(nodespos[i][1])
    #        li += [i]
    for i in range(N):
        if nodal_dof[i][0]==0 and nodal_dof[i][1]==0 and nodal_dof[i][2]==0:
                li+=[i]
    print(li)
    print(len(li))
    
    break
'''
force_p = []
data = open('D:\\summer_research\\\\Real\\assembly\\assembly_fix_and_force_point_data.txt','r')
a = data.read()
a = a.split('\n')
print(len(a))
for i in range(0,190):
    temp = int(a[i])
    force_p += [temp]
data.close()

new_data = open('D:\\summer_research\\\\Real\\assembly\\assembly_fix_point_data.txt','r')
a = new_data.read()
a = a.split('\n')
print(len(a))
for i in range(0,130):
    temp = int(a[i])
    force_p.remove(temp)

new_data.close()
print(len(force_p))
print(force_p)
'''
'''
new_data = open('D:\\summer_research\\\\Real\\assembly\\assembly_fix_point_data.txt','r')
a = new_data.read()
a = a.split('\n')

for i in range(0,130):
    a[i] = int(a[i])

new_data.close()
print(len(a))
print(a)
'''


'''
inputdata=open('D:\\summer_research\\\\Real\\assembly\\assembly_fix_and_force_point_data.txt','w')
name = 'Test_1_15.19341303139982_-18.777791325821205_26.351899966674583.rst'
result = pyansys.read_binary(os.path.join('D:\\summer_research\\Real\\assembly',name))
print(result)
ndnum, nodal_dof=result.nodal_solution(0)
    
N = 1211
count = 0

for n in range(0,N):
    if(nodal_dof[n-1][0]==0 and nodal_dof[n-1][1]==0 and nodal_dof[n-1][2]==0):
        inputdata.write(str(n))
        inputdata.write('\n')
        count += 1
inputdata.close()
print(count)
'''


'''
inputdata=open('D:\\summer_research\\\\Real\\assembly\\assembly_fix_point_data.txt','w')
for file in os.listdir(path):

    fname,ext=os.path.splitext(file)
    print(fname)

    result = pyansys.read_binary(os.path.join(path,file))
    print(result)
    ndnum, nodal_dof=result.nodal_solution(0)
    
    N = 1211
    count = 0

    for n in range(0,N):
        if(nodal_dof[n-1][0]==0 and nodal_dof[n-1][1]==0 and nodal_dof[n-1][2]==0):
            inputdata.write(str(n))
            inputdata.write('\n')
            count += 1
    inputdata.close()
    print(count)

    break
'''