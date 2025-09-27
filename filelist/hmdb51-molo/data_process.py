import json
f = open(r"C:\Users\dungs\Documents\Study\ManageBigData\TAMT-main\filelist\hmdb51-molo\train_few_shot.txt")
names_list = []
labels_list = []
for line in f:
    # print(line)
    # assert 0==1
    iamge_labels, iamge_names = line.split('//')
    iamge_labels = int(iamge_labels[5:])
    iamge_names = iamge_names[:-1].strip()
    iamge_names = iamge_names[7:]
    iamge_names = '/kaggle/input/hmdb51/hmdb51_org/' + iamge_names
    print(iamge_labels)
    names_list.append(iamge_names)
    labels_list.append(iamge_labels)

dic = {'image_names': names_list, 'image_labels': labels_list}
with open(r"C:\Users\dungs\Documents\GitHub\TAMT_N6\filelist\hmdb51-molo\base.json", 'w') as F:
    json.dump(dic, F)
