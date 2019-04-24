import json
import matplotlib.pyplot as plt
NUM = 20
with open('draw_loss.json','r') as f:
    draw_loss = json.load(f)
with open('draw_loss_valid.json','r') as f:
    draw_loss_valid = json.load(f)
assert(len(draw_loss)==NUM)
assert(len(draw_loss_valid)==NUM)
x = [i for i in range(NUM)]
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(x, draw_loss, label='train loss')
plt.plot(x, draw_loss_valid,label = 'validation loss')
plt.legend()
plt.show()
plt.savefig('loss_pic.jpg',bbox_inches='tight')