import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# location loss
# contain loss
# not contain_loss
# no object loss
# class loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class yololoss(nn.Module):
    def __init__(self, S, B, l_coord, l_noobj):
        super(yololoss, self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def computeIOU(self, l1, l2):  # l was like l,t,w,h
        l1l,l1t,l1r,l1b = l1[0],l1[1],l1[2]+l1[0],l1[3]+l1[1]
        l2l,l2t,l2r,l2b = l2[0],l2[1],l2[2]+l2[0],l2[3]+l2[1]
        # l2 cover l1
        if l2l<=l1l and l2t<=l1t and l2r>=l1r and l2b>=l1b:
            iou = l1[2] * l1[3] / (l2[2] * l2[3])
        # l1 cover l2
        elif l1l<=l2l and l1t<=l2t and l1r>=l2r and l1b>=l2b:
            iou = l2[2] * l2[3] / (l1[2] * l1[3])
        # separate
        elif l1r <= l2l or l2r <= l1l or l1b <= l2t or l2b <= l1t:
            iou = 0
        # intersect
        else:
            newleft = max(l1l, l2l)
            newtop = max(l1t, l2t)
            newwidth = min(l1r, l2r) - newleft
            newheight = min(l1b, l2b) - newtop
            newarea = newwidth * newheight
            iou = newarea / (l1[2]*l1[3]+l2[2]*l2[3] - newarea)
        return iou

    def forward(self, pred_tensor, target_tensor):

        # pred_tensor is like (batch_size,SxS,Bx5+10) [l,t,w,h,c]
        # target_tensor batch_size,SxS,4+1+1
        N = pred_tensor.shape[0]
        # compute a matrix as the same size of pred_tensor to denote if there exist obj with every elements is {0,1}
        coo_mask = target_tensor[:, :, :, 4] > 0
        noo_mask = target_tensor[:, :, :, 4] == 0

        """
        coo_mask_target = coo_mask.unsqueeze(-1).expand_as(target_tensor)
        noo_mask_target = noo_mask.unsqueeze(-1).expand_as(target_tensor)
        coo_mask_pred = coo_mask.unsqueeze(-1).expand_as(pred_tensor)
        noo_mask_pred = noo_mask.unsqueeze(-1).expand_as(pred_tensor)
        """
        coo_pred = pred_tensor[coo_mask]#.view(-1, 20)
        assert coo_pred.size()[1] == 20
        box_pred = coo_pred[:, :10].contiguous().view(-1, 5)
        class_pred = coo_pred[:, 10:]

        coo_target = target_tensor[coo_mask]#.view(-1, 6)
        assert coo_target.size()[1] == 6
        box_target = coo_target[:, :5].contiguous().view(-1, 5)
        class_target = coo_target[:, 5]
        class_target = class_target.long()

        # not contain object loss
        noo_pred = pred_tensor[noo_mask].view(-1, 20)
        noo_target = target_tensor[noo_mask].view(-1, 6)
        noo_c1_mask_pred = torch.ByteTensor(noo_pred.size()).to(device)
        noo_c1_mask_pred.zero_()
        noo_c2_mask_pred = torch.ByteTensor(noo_pred.size()).to(device)
        noo_c2_mask_pred.zero_()
        noo_c_mask_target = torch.ByteTensor(noo_target.size()).to(device)
        noo_c_mask_target.zero_()


        noo_c1_mask_pred[:, 4] = 1
        noo_c2_mask_pred[:, 9] = 1
        noo_c_mask_target[:,4] = 1
        noo_pred_c1 = noo_pred[noo_c1_mask_pred]
        noo_pred_c2 = noo_pred[noo_c2_mask_pred]
        noo_target_c = noo_target[noo_c_mask_target]
        noobj_loss = F.mse_loss(noo_pred_c1, noo_target_c,size_average=False) + F.mse_loss(noo_pred_c2, noo_target_c, size_average=False)

        # contain object loss
        # choose the best iou box

        # box_pred is the 2 times of box_target
        coo_response_mask = torch.ByteTensor(box_pred.size()[0], 1).to(device)
        coo_response_mask.zero_()
        coo_response_mask_target = torch.ByteTensor(box_target.size()[0], 1).to(device)
        coo_response_mask_target.zero_()
        box_target_iou = torch.zeros(box_target.size()).to(device)
        # for each target of the pic, compute max iou from 2 pred box of every grid
        for i in range(box_target.size()[0]):
            box1 = box_pred[2*i:2*i+2]
            box2 = box_target[i]
            iou1 = self.computeIOU(box1[0], box2)
            iou2 = self.computeIOU(box1[1], box2)
            if iou1 >= iou2:
                idx, max_iou = 0, iou1
            else:
                idx, max_iou = 1, iou2
            coo_response_mask[2*i+idx] = 1

            coo_response_mask_target[i] = 1

            box_target_iou[i, torch.LongTensor([4])] = max_iou
        box_target_iou = Variable(box_target_iou)
        coo_response_mask_target = coo_response_mask_target.expand_as(box_target_iou)
        box_response_target_iou = box_target_iou[coo_response_mask_target].view(-1, 5)
        coo_response_mask = coo_response_mask.expand_as(box_pred)
        box_response_pred = box_pred[coo_response_mask].view(-1, 5)

        # response loss
        contain_loss = F.mse_loss(box_response_pred[:, 4], box_response_target_iou[:, 4], size_average=False)
        loc_loss = F.mse_loss(torch.clamp(box_response_pred[:, :2], min=0,max=1), torch.clamp(box_target[:, :2], min=0,max=1), size_average=False) \
                   + F.mse_loss(torch.sqrt(torch.clamp(box_response_pred[:, 2:4],min=0,max=1)), torch.sqrt(torch.clamp(box_target[:, 2:4],min=0,max=1)),size_average=False)
        # not response loss
        box_not_response_pred = box_pred[1-coo_response_mask].view(-1, 5)
        box_not_response_target = box_target[coo_response_mask_target].view(-1, 5)
        box_not_response_target[:, 4] = 0

        not_contain_loss = F.mse_loss(box_not_response_pred[:, 4], box_not_response_target[:, 4], size_average=False)
        # class loss
        class_loss = F.cross_entropy(class_pred, class_target, size_average=False)
        total_loss = self.l_coord*loc_loss + 2*contain_loss + not_contain_loss + self.l_noobj*noobj_loss + class_loss
        return (total_loss) / N
    '''
        print('total loss is ',total_loss)
        print('loc_loss',self.l_coord*loc_loss)
        print('contain_loss',2*contain_loss)
        print('not_contain_loss',not_contain_loss)
        print('noobj_loss',self.l_noobj*noobj_loss)
        print('class_loss',class_loss)
    '''
    

if __name__ =='__main__':
    # pred_tensor = torch.randn(7,7,20)
    a,b,c,d=1,2,3,4
    y = yololoss(a,b,c,d)
    pred_tensor = torch.rand(1,7,7,20)
    target_tensor = torch.rand(1,7,7,6)
    rst=y(pred_tensor,target_tensor)
    print(rst)
