import torch
from resnet_yolo import resnet50
from datasets import test_loader
from yololoss import yololoss
import matplotlib.pyplot as plt
import json
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = resnet50().to(device)
net.load_state_dict(torch.load('best.pth'))
criterian = yololoss(7,2,5,0.5)
net.eval()
total_loss = 0
draw_loss = []
rst = []
with torch.no_grad():
    for i, (image, target) in enumerate(test_loader):
        image=image.to(device)
        target=target.to(device)
        out = net(image)
        #rst.append({'p':list(out.cpu().numpy()),'t':list(target.cpu().numpy())})
        loss = criterian(out, target)
        total_loss += loss.detach().item()
        if (i+1) % 50 == 0:
            print('loss:', total_loss/50)
            draw_loss.append(total_loss/50)
            total_loss = 0
#with open('test_rst.json','w') as f:
#    json.dump(rst, f)
plt.plot(draw_loss)
plt.show()