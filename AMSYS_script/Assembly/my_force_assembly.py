##############################
## FOR MODEL IN REAL FOLDER ##
##############################
import pyansys
import numpy as np
import os
import math

# Totally 1211 nodes
'''
force_p = [1045, 1046, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1152, 
    1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1166,
    1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180,
    1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191, 1192, 1193, 1194,
    1195, 1196, 1197, 1198, 1199]

path='D:\\summer_research\\Real\\assembly\\rst'

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