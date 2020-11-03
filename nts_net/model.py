from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision.ops import nms
from nts_net import resnet
import numpy as np
from nts_net.anchors import generate_default_anchor_maps
from nts_net.config import CAT_NUM, PROPOSAL_NUM, test_model


class ProposalNet(nn.Module):
    def __init__(self):
        super(ProposalNet, self).__init__()
        self.down1 = nn.Conv2d(2048, 128, 3, 1, 1)
        self.down2 = nn.Conv2d(128, 128, 3, 2, 1)
        self.down3 = nn.Conv2d(128, 128, 3, 2, 1)
        self.ReLU = nn.ReLU()
        self.tidy1 = nn.Conv2d(128, 6, 1, 1, 0)
        self.tidy2 = nn.Conv2d(128, 6, 1, 1, 0)
        self.tidy3 = nn.Conv2d(128, 9, 1, 1, 0)

    def forward(self, x):
        batch_size = x.size(0)
        d1 = self.ReLU(self.down1(x))
        d2 = self.ReLU(self.down2(d1))
        d3 = self.ReLU(self.down3(d2))
        t1 = self.tidy1(d1).view(batch_size, -1)
        t2 = self.tidy2(d2).view(batch_size, -1)
        t3 = self.tidy3(d3).view(batch_size, -1)
        return torch.cat((t1, t2, t3), dim=1)


class attention_net(nn.Module):
    def __init__(self, topN=6, device='cpu', num_classes=200):
        super(attention_net, self).__init__()
        self.device = device
        self.pretrained_model = resnet.resnet50(pretrained=True)
        self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.pretrained_model.fc = nn.Linear(512 * 4, num_classes)
        self.proposal_net = ProposalNet()
        self.topN = topN
        self.concat_net = nn.Linear(2048 * (CAT_NUM + 1), num_classes)
        self.partcls_net = nn.Linear(512 * 4, num_classes)
        _, edge_anchors, _ = generate_default_anchor_maps()
        self.pad_side = 224
        self.edge_anchors = (edge_anchors + 224).astype(np.int)

    def forward(self, x):
        resnet_out, rpn_feature, feature = self.pretrained_model(x)
        x_pad = F.pad(x, (self.pad_side, self.pad_side, self.pad_side, self.pad_side), mode='constant', value=0)
        batch = x.size(0)
        # we will reshape rpn to shape: batch * nb_anchor
        rpn_score = self.proposal_net(rpn_feature.detach())
        top_n_index_list = []
        top_n_prob_list = []
        top_n_coordinates = []
        edge_anchors_copy = torch.tensor(self.edge_anchors, dtype=torch.float32).to(self.device)
        zero_tensor = torch.zeros(rpn_score.size()[1], 1)

        for i in range(batch):
            rpn_score_reshape = rpn_score[i]
            nms_output_idx = nms(edge_anchors_copy, rpn_score_reshape, iou_threshold=0.25)
            nms_output_score = torch.gather(rpn_score_reshape, 0, nms_output_idx)
            nms_output_anchors = torch.index_select(edge_anchors_copy, 0, nms_output_idx)
            top_n_result = torch.topk(nms_output_score, self.topN)
            top_n_anchors = torch.index_select(nms_output_anchors, 0, top_n_result[1])
            y0_1, x0_1, y1_1, x1_1 = torch.split(top_n_anchors, 1, dim=1)
            top_n_anchors_1 = torch.cat([x0_1, y0_1, x1_1, y1_1], dim=1)
            top_n_index_origin = torch.index_select(nms_output_idx, 0, top_n_result[1])
            top_n_index_list.append(top_n_index_origin)
            top_n_prob_list.append(top_n_result[0])
            top_n_coordinates.append(top_n_anchors_1)

        top_n_index = torch.stack(top_n_index_list)
        top_n_prob = torch.stack(top_n_prob_list)
        part_imgs = torch.zeros([batch, self.topN, 3, 224, 224])
        for i in range(batch):
            for j in range(self.topN):
                y0 = top_n_coordinates[i][j][1].long()
                x0 = top_n_coordinates[i][j][0].long()
                y1 = top_n_coordinates[i][j][3].long()
                x1 = top_n_coordinates[i][j][2].long()
                part_imgs[i:i + 1, j] = F.interpolate(x_pad[i:i + 1, :, y0:y1, x0:x1], size=(224, 224), mode='bilinear',
                                                      align_corners=True)
        part_imgs = part_imgs.view(batch * self.topN, 3, 224, 224).to(self.device)
        _, _, part_features = self.pretrained_model(part_imgs.detach())
        part_feature = part_features.view(batch, self.topN, -1)
        part_feature = part_feature[:, :CAT_NUM, ...].contiguous()
        part_feature = part_feature.view(batch, -1)
        # concat_logits have the shape: B*200
        concat_out = torch.cat([part_feature, feature], dim=1)
        concat_logits = self.concat_net(concat_out)
        raw_logits = resnet_out
        # part_logits have the shape: B*N*200
        part_logits = self.partcls_net(part_features).view(batch, self.topN, -1)
        return top_n_coordinates, concat_out, raw_logits, concat_logits, part_logits, top_n_index, top_n_prob


def list_loss(logits, targets):
    temp = F.log_softmax(logits, -1)
    loss = [-temp[i][targets[i].item()] for i in range(logits.size(0))]
    return torch.stack(loss)


def ranking_loss(score, targets, proposal_num=PROPOSAL_NUM):
    loss = Variable(torch.zeros(1).cuda())
    batch_size = score.size(0)
    for i in range(proposal_num):
        targets_p = (targets > targets[:, i].unsqueeze(1)).type(torch.cuda.FloatTensor)
        pivot = score[:, i].unsqueeze(1)
        loss_p = (1 - pivot + score) * targets_p
        loss_p = torch.sum(F.relu(loss_p))
        loss += loss_p
    return loss / batch_size


def ntsnet(model_path, device='cuda'):
    net = attention_net(topN=PROPOSAL_NUM, device=device)
    ckpt = torch.load(model_path)
    net.load_state_dict(ckpt['net_state_dict'])
    print('test_acc:', ckpt['test_acc'])
    return net


if __name__ == '__main__':
    net = ntsnet(test_model)
