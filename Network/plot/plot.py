import pyansys
import os
import numpy as np
from PIL import Image

N = 165
test_size = 400
#train_size = 1600


test_displacement_Label = []
test_stress_Label = []
predict_displacement_Label = []
predict_stress_Label = []

file_object = open('test_data.txt')
for i in range(0,test_size):
    #line = file_object.readline().split() # if there is 0 & 1 about break/shuffle
    temp_test_stress_Label = []
    temp_test_displacement_Label = []
    for j in range(0,N):
        line = file_object.readline().split()
        temp_test_stress_Label += [[float(line[7]),float(line[8]),float(line[9])]]
        temp_test_displacement_Label += [[float(line[10]),float(line[11]),float(line[12])]]
    test_stress_Label += [temp_test_stress_Label]
    test_displacement_Label += [temp_test_displacement_Label]
file_object.close()

file_object = open('predict_data.txt')
for i in range(0,test_size):
    #line = file_object.readline().split() # if there is 0 & 1 about break/shuffle
    temp_predict_stress_Label = []
    temp_predict_displacement_Label = []
    for j in range(0,N):
        line = file_object.readline().split()
        temp_predict_stress_Label += [[float(line[0]),float(line[1]),float(line[2])]]
        temp_predict_displacement_Label += [[float(line[3]),float(line[4]),float(line[5])]]
    predict_stress_Label += [temp_predict_stress_Label]
    predict_displacement_Label += [temp_predict_displacement_Label]
file_object.close()

test_stress_Label = np.array(test_stress_Label)
test_displacement_Label = np.array(test_displacement_Label)
predict_stress_Label = np.array(predict_stress_Label)
predict_displacement_Label = np.array(predict_displacement_Label)

print(predict_displacement_Label.shape)
print(test_displacement_Label.shape)

result=pyansys.read_binary('test.rst')
nnum, disp = result.nodal_solution(0)

nsnum, nodal_stress=result.nodal_stress(0)

#result._plot_point_scalars(final_model(nd.array(testCharacter[5])).asnumpy())


for i in range(20):
    screen_file_predict='D:\\summer_research\\png\\predict_displacement\\test'+str(i)+'.png'
    result._plot_point_scalars(predict_displacement_Label[i].transpose(),screenshot=screen_file_predict,interactive=False,stitle='Predict Displacement')
    screen_file_label='D:\\summer_research\\png\\label_displacement\\test'+str(i)+'.png'
    result._plot_point_scalars(test_displacement_Label[i].transpose(),screenshot=screen_file_label,interactive=False,stitle='Label Displacement')
    
    predict_png=Image.open(screen_file_predict)
    label_png=Image.open(screen_file_label)
    combined_png=Image.new('RGB', (1024*2,745))
    combined_png.paste(predict_png,(0,0))
    combined_png.paste(label_png,(1024,0))
    filename='D:\\summer_research\\png\\combine_displacement\\'+str(i)+'.png'
    combined_png.save(filename)


#result._plot_point_scalars(nodal_stress.transpose(),screenshot='D:\\summer_research\\png\\TEST.png',
#    interactive=True,stitle='Predict stress',rng=[-1e4,1e4])



for i in range(30):
    screen_file_predict='D:\\summer_research\\png\\predict_stress\\test'+str(i)+'.png'
    result._plot_point_scalars(predict_stress_Label[i].transpose(),screenshot=screen_file_predict,
        interactive=False,stitle='Predict stress',rng=[-1e4,1e4])
    screen_file_label='D:\\summer_research\\png\\label_stress\\test'+str(i)+'.png'
    result._plot_point_scalars(test_stress_Label[i].transpose(),screenshot=screen_file_label,
        interactive=False,stitle='Label stress',rng=[-1e4,1e4])
    
    predict_png=Image.open(screen_file_predict)
    label_png=Image.open(screen_file_label)
    combined_png=Image.new('RGB', (1024*2,745))
    combined_png.paste(predict_png,(0,0))
    combined_png.paste(label_png,(1024,0))
    filename='D:\\summer_research\\png\\combine_stress\\'+str(i)+'.png'
    combined_png.save(filename)
