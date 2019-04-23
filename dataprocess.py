import h5py
import json
mat = h5py.File('/Users/zhangqi/Documents/DLHW/SVhn/train/digitStruct.mat','r')
print('keys of mat:',list(mat.keys()))
name = mat['digitStruct']['name'][()]
bbox = mat['digitStruct']['bbox'][()]
print('type of name:',type(name))
print('shape of name:',name.shape)
print('type of bbox:',type(name))
print('shape of bbox:',bbox.shape)

label_group =[]
for i in range(len(name)):

    st = name[i][0]
    obj = mat[st]
    st = ''.join(chr(j) for j in obj[:])
    #print(st)
    stt = bbox[i][0]
    ob = mat[stt]
    height = ob['height']  # <class 'h5py._hl.dataset.Dataset'>
    label = ob['label']
    left = ob['left']
    top = ob['top']
    width = ob['width']

    temp = []
    for j in range(len(height)):
        try:
            temp.append([
            ob[height[j][0]][:][0][0],
            ob[label[j][0]][:][0][0],
            ob[left[j][0]][:][0][0],
            ob[top[j][0]][:][0][0],
            ob[width[j][0]][:][0][0]])
        except AttributeError:
            pass
    #print(temp)
    label_group.append(temp)
with open('datamass.json', 'w') as f:
    json.dump(label_group, f)




