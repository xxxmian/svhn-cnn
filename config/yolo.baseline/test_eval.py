import torch
import numpy as np
from datasets import test_loader
from network import resnet50
from torchvision import transforms
to_pil_image = transforms.ToPILImage()
def print_img(image):
    image = image.cpu().squeeze(0)
    img = to_pil_image(image[:])
    img.convert('RGB')
    img.show()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = resnet50().to(device)
net.load_state_dict(torch.load('/home/xxxfrank/zqHomework/best.pth'))
net.eval()
total,right= 0,0
with torch.no_grad():
    for i, (image, target) in enumerate(test_loader):
        
        #print_img(image)
        image = image.to(device)
        target = target.to(device)
        pred = net(image)
        con_mask = torch.ByteTensor(pred.size()).to(device)
        con_mask.zero_()
        con_mask[:, :, :, 4] = 1
        con_mask[:, :, :, 9] = 1
        confidence = pred[con_mask].cpu().numpy() #<type 'tuple'>: (98,)
        con_mask[:, :, :, 10:] = 1
        box = pred[1-con_mask].view(-1,4).cpu().numpy()#<type 'tuple'>: (98,4)
        class_ = pred[:, :, :, 10:]#torch.Size([1, 7, 7, 10])
        # sorted by confidence
        sorted_idx = np.argsort(-confidence)
        sorted_con = np.sort(-confidence)

        t_con_mask = torch.ByteTensor(target.size()).to(device)
        t_con_mask.zero_()
        t_con_mask[:, :, :, 4] = 1

        t_confidence = target[t_con_mask].cpu().numpy()
        t_sorted_idx = np.argsort(-t_confidence)
        t_sorted_con = np.sort(-t_confidence)
        
        t_i = 0
        t_rst=set()
        while True:
            if -t_sorted_con[t_i]<1:
                break
            t_row=t_sorted_idx[t_i] // 28
            t_line = t_sorted_idx[t_i] % 28
            t_r = target[:,t_row,t_line,5]
            t_rst.add(int(t_r.item()))
            t_i+=1
        rst = set()
        for j in range(t_i):
            row=(sorted_idx[j]//2)//28
            line=(sorted_idx[j]//2)%28
            rst.add(torch.argmax(class_[:,row,line]).item())
            posi = -sorted_con[0]
        total+=t_i
        right+=len(rst.intersection(t_rst))
        #print(rst,'hope:',t_rst)
        

print('accuracy: {:.2f} %'.format((1.0*right/total)*100))