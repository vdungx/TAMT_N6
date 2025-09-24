import json
f = open(r"C:\Users\dungs\Documents\Study\ManageBigData\TAMT-main\filelist\hmdb51-molo\test_few_shot.txt")
names_list = []
labels_list = []
for line in f:
    # print(line)
    # assert 0==1
    iamge_labels, iamge_names = line.split('//')
    iamge_labels = int(iamge_labels[4:])
    iamge_names = iamge_names[:-1].strip()
    iamge_names = iamge_names[7:]
    iamge_names = 'C:/Users/dungs/Documents/Study/ManageBigData/TAMT-main/datasets/HMDB/hmdb51_org/' + iamge_names
    print(iamge_labels)
    names_list.append(iamge_names)
    labels_list.append(iamge_labels)

dic = {'image_names': names_list, 'image_labels': labels_list}
with open('novel.json', 'w') as F:
    json.dump(dic, F)
