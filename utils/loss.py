import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(self, fs=7, nb=2, nc=20, lambda_coord=5.0, lambda_noobj=0.5):
        super(Loss, self).__init__()

        self.FS = fs
        self.NB = nb
        self.NC = nc
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def compute_iou(self, box1, box2):

        def box_area(box):
            # box = 4xn
            return (box[2] - box[0]) * (box[3] - box[1])

        area1 = box_area(box1.T)
        area2 = box_area(box2.T)

        inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
        return inter / (area1[:, None] + area2 - inter)

    def forward(self, prediction, target_tensor):

        S, B, C = self.FS, self.NB, self.NC
        N = 5 * B + C  # 5=len([x, y, w, h, conf]

        batch_size = prediction.size(0)
        coord_mask = target_tensor[:, :, :, 4] > 0
        noobj_mask = target_tensor[:, :, :, 4] == 0
        coord_mask = coord_mask.unsqueeze(-1).expand_as(target_tensor)
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(target_tensor)

        coord_pred = prediction[coord_mask].view(-1, N)
        bbox_pred = coord_pred[:, :5 * B].contiguous().view(-1, 5)
        class_pred = coord_pred[:, 5 * B:]

        coord_target = target_tensor[coord_mask].view(-1, N)
        bbox_target = coord_target[:, :5 * B].contiguous().view(-1, 5)
        class_target = coord_target[:, 5 * B:]

        # Compute loss for the cells with no object bbox.
        noobj_pred = prediction[noobj_mask].view(-1, N)
        noobj_target = target_tensor[noobj_mask].view(-1, N)
        noobj_conf_mask = torch.cuda.BoolTensor(noobj_pred.size()).fill_(0)
        for b in range(B):
            noobj_conf_mask[:, 4 + b * 5] = 1
        noobj_pred_conf = noobj_pred[noobj_conf_mask]
        noobj_target_conf = noobj_target[noobj_conf_mask]
        loss_noobj = F.mse_loss(noobj_pred_conf, noobj_target_conf, reduction='sum')

        # Compute loss for the cells with objects.
        coord_response_mask = torch.cuda.BoolTensor(bbox_target.size()).fill_(0)
        coord_not_response_mask = torch.cuda.BoolTensor(bbox_target.size()).fill_(1)
        bbox_target_iou = torch.zeros(bbox_target.size()).cuda()

        # Choose the predicted bbox having the highest IoU for each target bbox.
        for i in range(0, bbox_target.size(0), B):
            pred = bbox_pred[i:i + B]
            pred_xyxy = torch.FloatTensor(pred.size())

            pred_xyxy[:, :2] = pred[:, :2] / float(S) - 0.5 * pred[:, 2:4]
            pred_xyxy[:, 2:4] = pred[:, :2] / float(S) + 0.5 * pred[:, 2:4]

            target = bbox_target[i]
            target = bbox_target[i].view(-1, 5)
            target_xyxy = torch.FloatTensor(target.size())

            target_xyxy[:, :2] = target[:, :2] / float(S) - 0.5 * target[:, 2:4]
            target_xyxy[:, 2:4] = target[:, :2] / float(S) + 0.5 * target[:, 2:4]

            iou = self.compute_iou(pred_xyxy[:, :4], target_xyxy[:, :4])  # [B, 1]
            max_iou, max_index = iou.max(0)
            max_index = max_index.data.cuda()

            coord_response_mask[i + max_index] = 1
            coord_not_response_mask[i + max_index] = 0

            bbox_target_iou[i + max_index, torch.LongTensor([4]).cuda()] = max_iou.data.cuda()
        bbox_target_iou = bbox_target_iou.cuda()

        bbox_pred_response = bbox_pred[coord_response_mask].view(-1, 5)
        bbox_target_response = bbox_target[coord_response_mask].view(-1, 5)
        target_iou = bbox_target_iou[coord_response_mask].view(-1, 5)

        loss_xy = F.mse_loss(bbox_pred_response[:, :2], bbox_target_response[:, :2], reduction='sum')

        loss_wh = F.mse_loss(torch.sqrt(bbox_pred_response[:, 2:4]),
                             torch.sqrt(bbox_target_response[:, 2:4]), reduction='sum')

        loss_obj = F.mse_loss(bbox_pred_response[:, 4], target_iou[:, 4], reduction='sum')

        loss_class = F.mse_loss(class_pred, class_target, reduction='sum')

        # Total loss
        loss = self.lambda_coord * (loss_xy + loss_wh) + loss_obj + self.lambda_noobj * loss_noobj + loss_class
        loss = loss / float(batch_size)

        return loss
