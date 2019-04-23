import torch
import numpy as np
from datasets import test_loader
from resnet_yolo import resnet50
from torchvision import transforms
to_pil_image = transforms.ToPILImage()
def print_img(image, path):
    image = image.cpu().squeeze(0)
    img = to_pil_image(image[:])
    img.convert('RGB')
    img.show()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = resnet50().to(device)
net.load_state_dict(torch.load('best.pth'))
net.eval()
with torch.no_grad():
    for i, (image, target) in enumerate(test_loader):
        if i==0:
            continue
        print_img(image,0)
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
        t_rst=[]
        while True:
            if -t_sorted_con[t_i]<1:
                break
            t_row=t_sorted_idx[t_i] // 7
            t_line = t_sorted_idx[t_i] % 7
            t_r = target[:,t_row,t_line,5]
            t_rst.append(t_r.item())
            t_i+=1
        row=(sorted_idx[0]//2)//7
        line=(sorted_idx[0]//2)%7
        rst = torch.argmax(class_[:,row,line])
        posi = -sorted_con[0]
        
        print(rst.item(),'hope:',t_rst)
        
        if i==1:
            break
