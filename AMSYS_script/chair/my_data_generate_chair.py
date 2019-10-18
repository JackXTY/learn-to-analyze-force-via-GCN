##############################
## FOR MODEL IN REAL FOLDER ##
##############################
import pyansys
import numpy as np
import os
import math

path='D:\\summer_research\\chair\\test_rst'
inputdata=open('D:\\summer_research\\chair\\test_chair_data.txt','w')
i=0
# Totally 567 nodes.

old_fix_p = [251, 252, 253, 254, 255, 269, 270, 271, 272, 273, 309, 310, 311, 312, 313, 314, 315, 
    316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333,
    334, 335, 336, 337, 338, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429,
    430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446] # 70
old_force_p = [231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 278, 
    303, 304, 305, 306, 307, 308, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483,
    484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502,
    503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521,
    522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540,
    541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 
    560, 561, 562, 563, 564, 565, 566] # 119

N_ = 567

force_p = [0 for _i in range(N_)]
for i in range(len(old_force_p)):
    force_p[old_force_p[i]] = 1

fix_p = [0 for _i in range(N_)]
for i in range(len(old_fix_p)):
    fix_p[old_fix_p[i]] = 1


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
    #print(x)print(y)print(z)
    #x = y = z = 0.1

    #open rst file
    result = pyansys.read_binary(os.path.join(path,file))
    #print(result)
    

    geometry=result.geometry
    nodespos=geometry['nodes']

    ####################
    ###    bug!!!!   ###
    ####################
    V = 0.1125*(0.3*0.1-0.16*0.04)
    gravity=  (9.8 * 487 * V)/N_   

    #print('File.' + str(i))

    #fix_n = 0
    #fix_p = []
    
    #res = int(ipTable[0])
    #inputdata.write(ipTable[0])
    #inputdata.write('\n')
    ndnum, nodal_dof=result.nodal_solution(0)

    #if(res==1):
    nsnum, nodal_stress=result.nodal_stress(0)
        

    for n in range(0,N_):

        #inputdata.write('Node.' + str(n) + ':\n')
        
        inputdata.write(str(fix_p[n]))     
        inputdata.write(' ')

        #inputdata.write('(position:) ')
        #nodes pos
        for j in range(0,3):
            inputdata.write(str(nodespos[n][j]))
            inputdata.write(' ')
        #print(str(n)+": "+str(nodespos[n-1][0])+" , "+str(nodespos[n-1][1])+" , "+str(nodespos[n-1][2]))
        
        #inputdata.write('(force:) ')
        #nodes force
        if(force_p[n]==1):
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
        #if(res==1):
        for j in range(0,3):
            inputdata.write(str(nodal_stress[n][j]))
            inputdata.write(' ')
        for j in range(0,3):
            inputdata.write(str(nodal_dof[n][j]))
            inputdata.write(' ')
        inputdata.write('\n')
        
        
        #print("Node."+str(n)+": "+str(nodal_stress[n-1][0])+" , "+str(nodal_stress[n-1][1])+
        #    " , "+str(nodal_stress[n-1][2])+" , "+str(nodal_stress[n-1][3])+" , "
        #    +str(nodal_stress[n-1][4])+" , "+str(nodal_stress[n-1][5]))
        
        #print(n)

    #print("fix number = "+str(fix_n))
    #print(fix_p)
    #break
    #print(i)
inputdata.close()
