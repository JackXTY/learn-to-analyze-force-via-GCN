import pyansys
import os
from PIL import ImageGrab

path = 'D:\\summer_research\\rst_data\\rst'
savepath = 'D:\\summer_research\\rst_data\\plot'

#cpos = [(5.583824390900331, 4.98382439090033, 6.333824390900331), (1.0, 0.4, 1.75), (0.0, 0.0, 1.0)]
#cpos = [(5.583824390900331, 4.98382439090033, 6.333824390900331), (1.0, 0.4, 1.75), (0.0, 0.0, 1.0)]
i=0
for file in os.listdir(path):
    i=i+1
    fullname = os.path.join(path,file)
    fname,ext=os.path.splitext(file)

    result = pyansys.read_binary(fullname)


    ipTable=fname.split('_')
    #print(ipTable)
    x=ipTable[1]
    y=ipTable[2]
    z=ipTable[3]
    
    #if (i==1):
    #cpos = result.plot_nodal_solution(0)
    #print(cpos)

    #cpos=cpos
    #screenshot=screen_file
    if(float(x)>0):
        x='00'+x
        screen_file=os.path.join(savepath,x+'.png')
        if(not os.path.exists(screen_file)):
            result.plot_nodal_stress(0, 'X', label='XStress', interactive=True)
    elif(float(x)<0):
        x='01'+x
        screen_file=os.path.join(savepath,x+'.png')
        if(not os.path.exists(screen_file)):
            result.plot_nodal_stress(0, 'X', label='XStress', interactive=True)

    
    if(float(y)>0):
        y='10'+y
        screen_file=os.path.join(savepath,y+'.png')
        if(not os.path.exists(screen_file)):
            result.plot_nodal_stress(0, 'Y', label='YStress', interactive=True)
    elif(float(z)<0):
        if(not os.path.exists(screen_file)):
            result.plot_nodal_stress(0, 'Y', label='YStress', interactive=True)

    if(float(z)>0):
        z='20'+z
        screen_file=os.path.join(savepath,z+'.png')
        if(not os.path.exists(screen_file)):
            result.plot_nodal_stress(0, 'Z', label='ZStress', interactive=True)
    elif(float(z)<0):
        z='21'+z
        screen_file=os.path.join(savepath,z+'.png')
        if(not os.path.exists(screen_file)):
            result.plot_nodal_stress(0, 'Z', label='ZStress', interactive=True)
    
    
    '''
    cpos = result.plot_nodal_solution(0)
    print(cpos)
    result.plot_nodal_solution(0, 'x', label='Displacement',
        #cpos=cpos,
        #screenshot='hexbeam_disp.png',
        window_size=[800, 600], interactive=True)
    '''

    if (i==6):
        break