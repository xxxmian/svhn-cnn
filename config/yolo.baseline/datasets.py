import torch
import json
import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms

LOGIT_SIZE=(28, 28, 6)
IMG_SIZE=(448, 448)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_path = '/home/xxxfrank/zqHomework/train'
test_data_path = '/home/xxxfrank/zqHomework/test'
datamass_path = 'datamass.json'
datamass_test_path = 'datamass_test.json'

class MyDataSets(data.Dataset):
    def __init__(self, data_path, datamass_path):
        self.data_path = data_path
        self.train_img = []
        self.train_label = []
        with open(datamass_path, 'r') as f:
            train_labelset = json.load(f)
        for i in range(len(train_labelset)):
            if train_labelset[i] == []:
                continue
            self.train_img.append(str(i+1))
            self.train_label.append(train_labelset[i])
        #self.train_img = self.train_img[:20]
        #self.train_label = self.train_label[:20]
        self.trans = transforms.Compose(
            [
                transforms.Resize(IMG_SIZE),
                transforms.ToTensor()
            ]
        )
        
    def __getitem__(self, item):
        image = Image.open(os.path.join(self.data_path, self.train_img[item]+'.png'))
        label = self.train_label[item]
        tempTensor = torch.zeros(LOGIT_SIZE)
        
        for j in range(len(label)):
            #label[j][2] += (448 - image.size[0]) / 2
            #label[j][3] += (448 - image.size[1]) / 2
            rate_x = float(IMG_SIZE[0])/image.size[0]
            rate_y = float(IMG_SIZE[0])/image.size[1]
            label[j][2] *= rate_x
            label[j][3] *= rate_y
            label[j][4] *= rate_x
            label[j][0] *= rate_y
            row = int(label[j][2] // 16)
            line = int(label[j][3] // 16)
            if row<0:
                row = 0
            if row >27:
                row =27
            if line <0:
                line = 0
            if line > 27:
                line = 27
            tempTensor[row][line][0] = label[j][2]#left
            tempTensor[row][line][1] = label[j][3]#top
            tempTensor[row][line][2] = label[j][4]#width
            tempTensor[row][line][3] = label[j][0]#height
            tempTensor[row][line][4] = 1 #confidence
            tempTensor[row][line][5] = label[j][1]#label
        image = self.trans(image)
        return image, tempTensor

    def __len__(self):
        return len(self.train_img)



train_data = MyDataSets(data_path, datamass_path)
train_loader = data.DataLoader(train_data, batch_size=24, shuffle=True)
test_data = MyDataSets(test_data_path, datamass_test_path)
test_loader = data.DataLoader(test_data, batch_size=1, shuffle=False)
#print(train_img[0].shape)
#print(len(train_data) + 'were loaded')

# every element of train_labelset contains 2x5 elements
# height label left top width
# p[0][1] for classify
# After CenterCrop height and width are not change
# newleft = (img_width - 448)/2 + left
# new_top = (img_height - 448)/2 + top

