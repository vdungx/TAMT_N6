import json
f = open(r"C:\Users\dungs\Documents\Study\ManageBigData\TAMT-main\filelist\SSv2Full\val_few_shot.txt")
names_list = []
labels_list = []
for line in f:
    print(line)
    # assert 0==1
    iamge_labels, iamge_names = line.split('/')
    iamge_labels = int(iamge_labels[3:])
    iamge_names = iamge_names[:-1].strip()
    iamge_names = 'C:/Users/dungs/Documents/Study/ManageBigData/TAMT-main/datasets/SSV2/20bn-something-something-v2/' + iamge_names + '.webm'
    names_list.append(iamge_names)
    labels_list.append(iamge_labels)

dic = {'image_names': names_list, 'image_labels': labels_list}
with open('val.json', 'w') as F:
    json.dump(dic, F)
