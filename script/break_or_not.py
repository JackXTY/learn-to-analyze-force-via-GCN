##############################
## FOR MODEL IN REAL FOLDER ##
##############################
import pyansys
import numpy as np
import os
import math


#path='D:\\summer_research\\Real\\test_for_break\\break_point_rst'
path='D:\\summer_research\\rst_data\\break_rst_7_22'


N_ = 165  # totally 921

i = 0

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
    

    #open rst file
    result = pyansys.read_binary(os.path.join(path,file))
    #print(result)
    
    nsnum, nodal_stress=result.nodal_stress(0)

    geometry=result.geometry
    nodespos=geometry['nodes']
    
    gravity=  (9.8 * 487 *0.1*0.3*0.06)/N_   #old:0.02...
    
    real_max_stress = 0
    
    # Point
    for n in range(0,N_):
        total_stress = max(nodal_stress[n][0], nodal_stress[n][1], nodal_stress[n][2])
        if(total_stress>real_max_stress):
            real_max_stress = total_stress

    print(real_max_stress)
    
    if(real_max_stress>2300000):
        os.rename(os.path.join(path,file), os.path.join(path,"0_"+ipTable[1]+"_"+ipTable[2]+"_"+ipTable[3]+".rst"))
        print("Rename: "+str(ipTable))

    '''
    if (int(ipTable[0])==0 and real_max_stress>2300000) or (int(ipTable[0])==1 and real_max_stress<2300000):
        print("Right")
    else:
        print("Wrong")
    '''


    print(i)
    #print()
    
