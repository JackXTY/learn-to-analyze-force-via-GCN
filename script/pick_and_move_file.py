import shutil
import os
import random

old_path='D:\\summer_research\\rst_data\\rst'
new_path='D:\\summer_research\\rst_data\\test_set'
i = 0
N = 0
for file in os.listdir(old_path):

    probability = random.random()
    if probability > 0.8 :
        shutil.move(old_path+'\\'+file, new_path+'\\'+file)
        print(str(N)+":"+str(i))
        N += 1
    if N >= 400:  
        break
    i += 1
print("Totally "+str(N)+" files.")