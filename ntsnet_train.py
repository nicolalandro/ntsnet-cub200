import argparse
import os
from datetime import datetime

import torch.utils.data
import torchvision
from PIL import Image
from torch.nn import DataParallel
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms

# fix for python 3.6
try:
    import nts_net
except:
    import sys

    sys.path.insert(0, './')

from nts_net.config import PROPOSAL_NUM  # this is also used into another file
from nts_net import model
from nts_net.utils import init_log, progress_bar

parser = argparse.ArgumentParser("NtsNet CUb 200 2011")
parser.add_argument('--gpu', type=str, default='0', help='select the gpu or gpus to use')

parser.add_argument('--path', type=str, default='/media/mint/Barracuda/Datasets/CUB_200_2011/')
parser.add_argument('--start-epoch', type=int, default=1)
parser.add_argument('--max-epoch', type=int, default=50)
parser.add_argument('--batch-size', type=int, default=7)
parser.add_argument('--lr', type=float, default=0.001, help="learning rate for model")
parser.add_argument('--wd', type=float, default=1e-4, help="wd rate for model")
parser.add_argument('--momentum', type=float, default=0.9, help="sgd momentum")
parser.add_argument('--adam-w', type=float, default=0.0, help="adam weight")
parser.add_argument('--sgd-w', type=float, default=1.0, help="sgd weight")

parser.add_argument('--resume-path', type=str, default='', help="resume weight path")
parser.add_argument('--save-freq', type=int, default=500, help="save model at epoch x")
parser.add_argument('--save-dir', type=str, default='/media/mint/Barracuda/Models/Cub200')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# path = '/home/risen/datasets-nas/CUB_200_2011/CUB_200_2011'
path = args.path

SAVE_FREQ = args.save_freq
EPOCHS = args.max_epoch
BATCH_SIZE = args.batch_size
LR = args.lr
WD = args.wd

start_epoch = args.start_epoch
# resume = 'models/Part_2_weighted_20200326_141129/048.ckpt'
resume = False
if args.resume_path != '':
    resume = args.resume_path

adam_w = args.adam_w
sgd_w = args.sgd_w

# save_dir = '/home/risen/nic/multi_optimizer/logs'
save_dir = args.save_dir
experiment_name = f'nts_net_{adam_w}_{sgd_w}'

# logs preparation
save_dir = os.path.join(save_dir, f'{experiment_name}_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
if os.path.exists(save_dir):
    raise NameError('model dir exists!')
os.makedirs(save_dir)
logging = init_log(save_dir)
_print = logging.info


def evaluate():
    global i, data, img, label, batch_size, _, concat_logits, concat_loss
    ##########################  evaluate net on train set  ###############################
    # train_loss = 0
    # train_correct = 0
    # total = 0
    # net.eval()
    # for i, data in enumerate(trainloader):
    #     with torch.no_grad():
    #         img, label = data[0].cuda(), data[1].cuda()
    #         batch_size = img.size(0)
    #         _, _, _, concat_logits, _, _, _ = net(img)
    #         # calculate loss
    #         concat_loss = creterion(concat_logits, label)
    #         # calculate accuracy
    #         _, concat_predict = torch.max(concat_logits, 1)
    #         total += batch_size
    #         train_correct += torch.sum(concat_predict.data == label.data)
    #         train_loss += concat_loss.item() * batch_size
    #         progress_bar(i, len(trainloader), 'eval train set')
    #
    # train_acc = float(train_correct) / total
    # train_loss = train_loss / total
    #
    # _print(
    #     'epoch:{} - train loss: {:.3f} and train acc: {:.3f} total sample: {}'.format(
    #         epoch,
    #         train_loss,
    #         train_acc,
    #         total))
    ##########################  evaluate net on test set  ###############################


if __name__ == '__main__':
    # read dataset
    transform_train = transforms.Compose([
        # transforms.Resize((600, 600), Image.BILINEAR),
        # transforms.CenterCrop((448, 448)),
        transforms.Resize((448, 448), Image.BILINEAR),
        transforms.RandomHorizontalFlip(),  # solo se train
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    transform_test = transforms.Compose([
        # transforms.Resize((600, 600), Image.BILINEAR),
        # transforms.CenterCrop((448, 448)),
        transforms.Resize((448, 448), Image.BILINEAR),
        # transforms.RandomHorizontalFlip(), # solo se train
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_path = f'{path}/train'
    trainset = torchvision.datasets.ImageFolder(root=train_path, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,
                                              num_workers=2)

    test_path = f'{path}/test'
    testset = torchvision.datasets.ImageFolder(root=test_path, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    classes_dict = {i: v for i, v in enumerate(trainset.classes)}
    print(classes_dict)

    # define model
    net = model.attention_net(topN=PROPOSAL_NUM, num_classes=len(classes_dict), device='cuda')

    if resume:
        ckpt = torch.load(resume)
        net.load_state_dict(ckpt['net_state_dict'])
        start_epoch = ckpt['epoch'] + 1
    creterion = torch.nn.CrossEntropyLoss()

    # define optimizers
    raw_parameters = list(net.pretrained_model.parameters())
    part_parameters = list(net.proposal_net.parameters())
    concat_parameters = list(net.concat_net.parameters())
    partcls_parameters = list(net.partcls_net.parameters())

    raw_optimizer = torch.optim.SGD(raw_parameters, lr=LR, momentum=args.momentum, weight_decay=WD)

    concat_optimizer = torch.optim.SGD(concat_parameters, lr=LR, momentum=args.momentum, weight_decay=WD)

    part_optimizer = torch.optim.SGD(part_parameters, lr=LR, momentum=args.momentum, weight_decay=WD)

    partcls_optimizer = torch.optim.SGD(partcls_parameters, lr=LR, momentum=args.momentum, weight_decay=WD)

    schedulers = [MultiStepLR(raw_optimizer, milestones=[60, 100], gamma=0.1),
                  MultiStepLR(concat_optimizer, milestones=[60, 100], gamma=0.1),
                  MultiStepLR(part_optimizer, milestones=[60, 100], gamma=0.1),
                  MultiStepLR(partcls_optimizer, milestones=[60, 100], gamma=0.1)]
    net = net.cuda()
    net = DataParallel(net)

    for epoch in range(start_epoch, EPOCHS):
        for scheduler in schedulers:
            scheduler.step()
        ##########################  train the model  ###############################
        _print('--' * 50)
        net.train()
        for i, data in enumerate(trainloader):
            img, label = data[0].cuda(), data[1].cuda()
            batch_size = img.size(0)
            raw_optimizer.zero_grad()
            part_optimizer.zero_grad()
            concat_optimizer.zero_grad()
            partcls_optimizer.zero_grad()

            _, _, raw_logits, concat_logits, part_logits, _, top_n_prob = net(img)
            part_loss = model.list_loss(part_logits.view(batch_size * PROPOSAL_NUM, -1),
                                        label.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1)).view(batch_size,
                                                                                                  PROPOSAL_NUM)
            raw_loss = creterion(raw_logits, label)
            concat_loss = creterion(concat_logits, label)
            rank_loss = model.ranking_loss(top_n_prob, part_loss)
            partcls_loss = creterion(part_logits.view(batch_size * PROPOSAL_NUM, -1),
                                     label.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1))

            total_loss = raw_loss + rank_loss + concat_loss + partcls_loss
            total_loss.backward()
            raw_optimizer.step()
            part_optimizer.step()
            concat_optimizer.step()
            partcls_optimizer.step()
            progress_bar(i, len(trainloader), 'train')
            _print(f'Batch {i}/{len(trainloader)}   Loss: {total_loss.item()}')

        ##########################  evaluate net and save model  ###############################
        # if epoch % SAVE_FREQ == 0:
        evaluate()
        net.eval()
        test_loss = 0
        test_correct = 0
        total = 0
        for i, data in enumerate(testloader):
            with torch.no_grad():
                img, label = data[0].cuda(), data[1].cuda()
                batch_size = img.size(0)
                _, _, concat_logits, _, _, _, _ = net(img)
                # calculate loss
                concat_loss = creterion(concat_logits, label)
                # calculate accuracy
                _, concat_predict = torch.max(concat_logits, 1)
                total += batch_size
                test_correct += torch.sum(concat_predict.data == label.data)
                test_loss += concat_loss.item() * batch_size
                progress_bar(i, len(testloader), 'eval test set')
        test_acc = float(test_correct) / total
        test_loss = test_loss / total
        _print(
            'epoch:{} - test loss: {:.3f} and test acc: {:.3f} total sample: {}'.format(
                epoch,
                test_loss,
                test_acc,
                total))
        ##########################  save model  ###############################
        net_state_dict = net.module.state_dict()
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        if epoch % SAVE_FREQ == 0:
            torch.save({
                'epoch': epoch,
                # 'train_loss': train_loss,
                # 'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'net_state_dict': net_state_dict},
                os.path.join(save_dir, '%03d.ckpt' % epoch))

    print('finishing training')
