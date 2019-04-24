import torch
import numpy as np
from datasets import test_data
from torchvision import transforms
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_pil_image = transforms.ToPILImage()
def print_img(image):
    image = image.cpu().squeeze(0)
    img = to_pil_image(image[:])
    img.convert('RGB')
    img.show()
img,target,label = test_data[0]
print_img(img)

print(label)
t_con_mask = torch.ByteTensor(target.size()).to(device)
t_con_mask.zero_()
t_con_mask[:, :, 4] = 1

t_confidence = target[t_con_mask].cpu().numpy()
t_sorted_idx = np.argsort(-t_confidence)
t_sorted_con = np.sort(-t_confidence)

t_i = 0
t_rst = []
while True:
    if -t_sorted_con[t_i] < 1:
        break
    t_row = t_sorted_idx[t_i] // 28
    t_line = t_sorted_idx[t_i] % 28
    t_r = target[t_row, t_line, 5]
    t_rst.append(t_r.item())
    t_i += 1
print(t_rst)