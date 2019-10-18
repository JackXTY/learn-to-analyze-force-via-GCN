import pyansys
import numpy as np
import os
import math

'''
Totally 1211 nodes, so there should be at least 2400 edges around(considering iteration).

'''


path='D:\\summer_research\\Real\\assembly\\rst'
inputdata=open('D:\\summer_research\\Real\\assembly\\assembly_adjacency.txt','w')
j=0

nodes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
    44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87,
    88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107,
    108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
    126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
    144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161,
    162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179,
    180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197,
    198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215,
    216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233,
    234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251,
    252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269,
    931, 932, 933, 934, 935, 936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947, 948,
    949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966,
    967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 980, 981, 982, 983, 984,
    985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002,
    1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017,
    1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032,
    1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1043, 1044, 1045, 1046, 1047,
    1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060, 1061, 1062,
    1063, 1064, 1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073, 1074, 1075, 1076, 1077,
    1078, 1079, 1080, 1081, 1082, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1092,
    1093, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107,
    1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122,
    1123, 1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1137,
    1138, 1139, 1140, 1141, 1142, 1143, 1144, 1145, 1146, 1147, 1148, 1149, 1150, 1151, 1152,
    1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167,
    1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182,
    1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197,
    1198, 1199, 1200, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210]
# Totally 550 nodes.


for file in os.listdir(path):
    
    j = j+1
    #compute the file name:node,force
    fname,ext=os.path.splitext(file)
    
    
    ipTable=fname.split('_')
    
    #x=float(ipTable[1])
    #y=float(ipTable[2])
    #z=float(ipTable[3])
    if(int(ipTable[0])==0):
        continue
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
    N = 550
    
    #nodes pos
    geometry=result.geometry
    nodespos=geometry['nodes']
    #print(nodespos)

    print('File.' + str(j))
    
    
    result = [[0 for _i in range(0,N)] for _j in range(0,N)]

    '''
    node_list = []
    for n in range(0,1900):
        if str(nodal_stress[n][0])!="nan" :
            node_list += [n]
    print(len(node_list))
    print(node_list)
    '''

    
    for n in range(0,N):
        
        #print(str(n)+": "+str(nodespos[n][0])+" , "+str(nodespos[n][1])+" , "+str(nodespos[n][2]))

        xd = 0
        yd = 0
        zd = 0
        for i in range(0,N):
            temp_xd = abs(nodespos[nodes[i]][0] - nodespos[nodes[n]][0])
            temp_yd = abs(nodespos[nodes[i]][1] - nodespos[nodes[n]][1])
            temp_zd = abs(nodespos[nodes[i]][2] - nodespos[nodes[n]][2])
            if (temp_xd<xd or xd==0) and temp_xd>xd*0.3:
                xd = temp_xd
            if (temp_yd<yd or yd==0) and temp_yd>yd*0.3:
                yd = temp_yd
            if (temp_zd<zd or zd==0) and temp_zd>zd*0.3:
                zd = temp_zd   
        
        #print()print("xd="+str(xd))print("yd="+str(yd))print("zd="+str(zd))print()
        
        for i in range(0,N):
            _xd = abs(nodespos[nodes[i]][0] - nodespos[nodes[n]][0])
            _yd = abs(nodespos[nodes[i]][1] - nodespos[nodes[n]][1])
            _zd = abs(nodespos[nodes[i]][2] - nodespos[nodes[n]][2])
            
            if  (  ( _xd<xd*1.5 and _xd>xd*0.01 and _yd<yd*0.5 and _zd<zd*0.5)
                or ( _yd<yd*1.5 and _yd>yd*0.01 and _xd<xd*0.5 and _zd<zd*0.5)
                or ( _zd<zd*1.5 and _zd>zd*0.01 and _xd<xd*0.5 and _yd<yd*0.5) ):
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
