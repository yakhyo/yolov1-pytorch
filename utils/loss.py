import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(self, feature_size=7, num_bboxes=2, num_classes=20, lambda_coord=5.0, lambda_noobj=0.5):
        super(Loss, self).__init__()

        self.S = feature_size
        self.B = num_bboxes
        self.C = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def compute_iou(self, box1, box2):

        def box_area(box):
            # box = 4xn
            return (box[2] - box[0]) * (box[3] - box[1])

        area1 = box_area(box1.T)
        area2 = box_area(box2.T)

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
        return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

    def forward(self, prediction, target_tensor):
        """ Compute Loss for YOLO training """
        # TODO: Remove redundant dimensions for some Tensors.

        S, B, C = self.S, self.B, self.C
        N = 5 * B + C  # 5=len([x, y, w, h, conf]

        batch_size = prediction.size(0)
        coord_mask = target_tensor[:, :, :, 4] > 0  # mask for the cells which contain objects. [n_batch, S, S]
        noobj_mask = target_tensor[:, :, :, 4] == 0  # mask for the cells which do not contain objects. [n_batch, S, S]
        coord_mask = coord_mask.unsqueeze(-1).expand_as(target_tensor)  # [n_batch, S, S] -> [n_batch, S, S, N]
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(target_tensor)  # [n_batch, S, S] -> [n_batch, S, S, N]

        coord_pred = prediction[coord_mask].view(-1, N)  # pred tensor on the cells which contain objects. [n_coord, N]
        # n_coord: number of the cells which contain objects.
        bbox_pred = coord_pred[:, :5 * B].contiguous().view(-1, 5)  # [n_coord x B, 5=len([x, y, w, h, conf])]
        class_pred = coord_pred[:, 5 * B:]  # [n_coord, C]

        coord_target = target_tensor[coord_mask].view(-1, N)  # target tensor on the cells which contain objects. [n_coord, N]
        # n_coord: number of the cells which contain objects.
        bbox_target = coord_target[:, :5 * B].contiguous().view(-1, 5)  # [n_coord x B, 5=len([x, y, w, h, conf])]
        class_target = coord_target[:, 5 * B:]  # [n_coord, C]

        # Compute loss for the cells with no object bbox.
        noobj_pred = prediction[noobj_mask].view(-1, N)  # pred tensor on the cells which do not contain objects. [n_noobj, N]
        # n_noobj: number of the cells which do not contain objects.
        noobj_target = target_tensor[noobj_mask].view(-1, N)  # target tensor on the cells which do not contain objects. [n_noobj, N]
        # n_noobj: number of the cells which do not contain objects.
        noobj_conf_mask = torch.cuda.BoolTensor(noobj_pred.size()).fill_(0)  # [n_noobj, N]
        for b in range(B):
            noobj_conf_mask[:, 4 + b * 5] = 1  # noobj_conf_mask[:, 4] = 1; noobj_conf_mask[:, 9] = 1
        noobj_pred_conf = noobj_pred[noobj_conf_mask]  # [n_noobj, 2=len([conf1, conf2])]
        noobj_target_conf = noobj_target[noobj_conf_mask]  # [n_noobj, 2=len([conf1, conf2])]
        loss_noobj = F.mse_loss(noobj_pred_conf, noobj_target_conf, reduction='sum')

        # Compute loss for the cells with objects.
        coord_response_mask = torch.cuda.BoolTensor(bbox_target.size()).fill_(0)  # [n_coord x B, 5]
        coord_not_response_mask = torch.cuda.BoolTensor(bbox_target.size()).fill_(1)  # [n_coord x B, 5]
        bbox_target_iou = torch.zeros(bbox_target.size()).cuda()  # [n_coord x B, 5], only the last 1=(conf,) is used

        # Choose the predicted bbox having the highest IoU for each target bbox.
        for i in range(0, bbox_target.size(0), B):
            pred = bbox_pred[i:i + B]  # predicted bboxes at i-th cell, [B, 5=len([x, y, w, h, conf])]
            pred_xyxy = torch.FloatTensor(pred.size())  # [B, 5=len([x1, y1, x2, y2, conf])]
            # Because (center_x,center_y)=pred[:, 2] and (w,h)=pred[:,2:4] are normalized for cell-size and image-size respectively,
            # rescale (center_x,center_y) for the image-size to compute IoU correctly.
            pred_xyxy[:, :2] = pred[:, :2] / float(S) - 0.5 * pred[:, 2:4]
            pred_xyxy[:, 2:4] = pred[:, :2] / float(S) + 0.5 * pred[:, 2:4]

            target = bbox_target[i]  # target bbox at i-th cell. Because target boxes contained by each cell are identical in current implementation, enough to extract the first one.
            target = bbox_target[i].view(-1, 5)  # target bbox at i-th cell, [1, 5=len([x, y, w, h, conf])]
            target_xyxy = torch.FloatTensor(target.size())  # [1, 5=len([x1, y1, x2, y2, conf])]
            # Because (center_x,center_y)=target[:, 2] and (w,h)=target[:,2:4] are normalized for cell-size and image-size respectively,
            # rescale (center_x,center_y) for the image-size to compute IoU correctly.
            target_xyxy[:, :2] = target[:, :2] / float(S) - 0.5 * target[:, 2:4]
            target_xyxy[:, 2:4] = target[:, :2] / float(S) + 0.5 * target[:, 2:4]

            iou = self.compute_iou(pred_xyxy[:, :4], target_xyxy[:, :4])  # [B, 1]
            max_iou, max_index = iou.max(0)
            max_index = max_index.data.cuda()

            coord_response_mask[i + max_index] = 1
            coord_not_response_mask[i + max_index] = 0

            # "we want the confidence score to equal the intersection over union (IOU) between the predicted box and the ground truth"
            # from the original paper of YOLO.
            bbox_target_iou[i + max_index, torch.LongTensor([4]).cuda()] = (max_iou).data.cuda()
        bbox_target_iou = bbox_target_iou.cuda()

        # BBox location/size and objectness loss for the response bboxes.
        bbox_pred_response = bbox_pred[coord_response_mask].view(-1, 5)  # [n_response, 5]
        bbox_target_response = bbox_target[coord_response_mask].view(-1, 5)  # [n_response, 5], only the first 4=(x, y, w, h) are used
        target_iou = bbox_target_iou[coord_response_mask].view(-1, 5)  # [n_response, 5], only the last 1=(conf,) is used
        loss_xy = F.mse_loss(bbox_pred_response[:, :2],
                             bbox_target_response[:, :2],
                             reduction='sum')

        loss_wh = F.mse_loss(torch.sqrt(bbox_pred_response[:, 2:4]),
                             torch.sqrt(bbox_target_response[:, 2:4]),
                             reduction='sum')

        loss_obj = F.mse_loss(bbox_pred_response[:, 4],
                              target_iou[:, 4],
                              reduction='sum')

        # Class probability loss for the cells which contain objects.
        loss_class = F.mse_loss(class_pred,
                                class_target,
                                reduction='sum')

        # Total loss
        loss = self.lambda_coord * (loss_xy + loss_wh) + loss_obj + self.lambda_noobj * loss_noobj + loss_class
        loss = loss / float(batch_size)

        return loss
