import pyansys
import numpy as np
import os
import math

#D:\summer_research\script\trash
path='D:\\summer_research\\project_0\\rst'
#inputdata=open('D:\\summer_research\\project_0\\inputdata_v0.txt','w')
i=0
#inputdata.write("Test information 1\n")

real_file=os.path.join(path+'\\1_0.1_0.1_0.1.rst')
#real_file=os.path.join(path+'\\OK.rst')
print(real_file)
     
result = pyansys.read_binary(real_file)


for file in os.listdir(path):

    i=i+1

    #compute the file name:node,force
    fname,ext=os.path.splitext(file)
    #print(fname)
    #print(ext)
    print(file)
    
    ipTable=fname.split('_')
    print(ipTable)
    x=float(ipTable[1])
    y=float(ipTable[2])
    z=float(ipTable[3])

    print("x="+str(x)+", y="+str(y)+", z="+str(z))
    inputdata.write("Test information 2\n")
    
    #open rst file
    real_file=os.path.join(path+'\\OK.rst')

    print(real_file)
     
    result = pyansys.read_binary(real_file)
    
    print("result"+str(result))
    inputdata.write("Test information 3\n")
    
    #Beam natural frequencies
    freqs = result.time_values
    print(freqs)
    
    

    nsnum, nodal_stress=result.nodal_stress(0)
    ndnum, nodal_dof=result.nodal_solution(0)

    print(nsnum)
    print(ndnum)
    print('\n')
    geometry=result.geometry
    nodespos=geometry['nodes']
    gravity= (9.8 * 9.1312e-002)/42.0
    
    
    #inputdata.write('\nFile.' + str(i) + ':\n')
    

    print("Here")

    
    for m in range(1,43):

        #inputdata.write('Node.' + str(m) + ':\n')

        if(m==24 or m==25 or m==26 or m==27 or m==28 or m==29):
            inputdata.write(str(1))
        else:
            inputdata.write(str(0))        
        inputdata.write(' ')

        #inputdata.write('(position:) ')
        #nodes pos
        for n in range(0,3):
            inputdata.write(str(nodespos[m-1][n]))
            inputdata.write(' ')

        #inputdata.write('(force:) ')
        #nodes force
        if(m==23 or m== 32):
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
        for n in range(0,6):
            inputdata.write(str(nodal_stress[m-1][n]))
            inputdata.write(' ')
        inputdata.write(str(nodal_dof[m-1][0]))
        for n in range(1,3):
            inputdata.write(' ')
            inputdata.write(str(nodal_dof[m-1][n]))
        inputdata.write('\n')
    
    print(i)
print("Last Test")
inputdata.write("Last Test\n")
inputdata.close()
'''
