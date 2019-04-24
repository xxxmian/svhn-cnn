import torch
from torch.autograd import Variable
from datasets import train_loader, test_loader
from network_deep import resnet50
from yololoss import yololoss
from torchvision import models
import numpy as np
import json
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epoch = 20
learning_rate = 0.001
S, B, l_coord, l_noobj = 7,2,5,0.5

draw_loss = []
draw_loss_valid = []
draw_acc = []
test_rst = []

net = resnet50().to(device)
resnet = models.resnet50(pretrained=True)
new_state_dict = resnet.state_dict()
dd = net.state_dict()
for k in new_state_dict.keys():
    #print(k)
    if k in dd.keys() and not k.startswith('fc') and not k.startswith('layer4'):
        #print('yes')
        dd[k] = new_state_dict[k]
net.load_state_dict(dd)

criterian = yololoss(S, B, l_coord, l_noobj)
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
total_step = len(train_loader)
cur_lr = learning_rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
net.train()
best_test_loss = np.inf
for epoch in range(num_epoch):
    if epoch == 10:
        cur_lr /= 3
        update_lr(optimizer, cur_lr)
    if epoch == 15:
        cur_lr /= 3
        update_lr(optimizer, cur_lr)
    total_loss = 0
    net.train()
    for i, (img, label) in enumerate(train_loader):
        img = Variable(img).to(device)
        label = label.to(device)
        output = net(img)
        loss = criterian(output, label)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        if (i + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epoch, i + 1, total_step, loss.item()))
            #vis.plot_train_val(loss_train=total_loss / (i + 1))
    draw_loss.append(total_loss / (i + 1))
   
    if epoch % 2 == 0:
        torch.save(net.state_dict(), '/home/xxxfrank/zqHomework/model_{}.ckpt'.format(epoch))

        validation_loss = 0.0
        
    net.eval()
    with torch.no_grad():
        for i, (images, target) in enumerate(test_loader):
            images = Variable(images)
            target = Variable(target)
            
            images, target = images.to(device), target.to(device)
            
            pred = net(images)
            test_rst.append({'p':pred,'t':target})
            loss = criterian(pred, target)
            validation_loss += loss.item()
        validation_loss /= len(test_loader)
        draw_loss_valid.append(validation_loss)
        
        if best_test_loss > validation_loss:
            best_test_loss = validation_loss
            print('get best test loss %.5f' % best_test_loss)
            torch.save(net.state_dict(), '/home/xxxfrank/zqHomework/best.pth')
        
with open('draw_loss.json','w') as f:
    json.dump(draw_loss,f)
with open('draw_loss_valid.json','w') as f:
    json.dump(draw_loss_valid,f)
