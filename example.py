#
import torch

from nts_net.model import attention_net

device = torch.device('cpu')
net = attention_net(topN=6, num_classes=200, device=device)
checkpoint = 'models/nts_net_cub200.pt'
state_dict = torch.load(checkpoint)
net.load_state_dict(state_dict)
