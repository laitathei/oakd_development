
import os
import random 
random.seed(0)
jsonfilepath=r'./VOCdevkit/grass/json'
saveBasePath=r"./VOCdevkit/grass/ImageSets/Segmentation"
 
trainval_percent=0.8
train_percent=0.9

temp_json = os.listdir(jsonfilepath)
total_json = []
for json in temp_json:
    #print(json)
    if json.endswith(".json"):
        total_json.append(json)

num=len(total_json)  
list=range(num)  
tv=int(num*trainval_percent)  
tr=int(tv*train_percent)  
trainval= random.sample(list,tv)  
train=random.sample(trainval,tr)  
 
print("train and val size",tv)
print("traub suze",tr)
ftrainval = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
ftest = open(os.path.join(saveBasePath,'test.txt'), 'w')  
ftrain = open(os.path.join(saveBasePath,'train.txt'), 'w')  
fval = open(os.path.join(saveBasePath,'val.txt'), 'w')  
 
for i  in list:  
    #print(total_json[i][:-5]+'\n')
    name=total_json[i][:-5]+'\n'  
    if i in trainval:  
        ftrainval.write(name)  
        if i in train:  
            ftrain.write(name)  
        else:  
            fval.write(name)  
    else:  
        ftest.write(name)  
  
ftrainval.close()  
ftrain.close()  
fval.close()  
ftest .close()