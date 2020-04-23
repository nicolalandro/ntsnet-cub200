dependencies = ['torch', 'numpy']

import torch

from nts_net.model import attention_net

cub_200_2011_state_dict_url = 'https://github.com/nicolalandro/ntsnet_cub200/releases/download/0.2/nts_net_cub200.pt'


def netsnet(pretrained=False, **kwargs):
    """ # This docstring shows up in hub.help()
    NtsNET model
    pretrained (bool): kwargs, load pretrained weights into the model
    **kwargs
        topN (int): the number of crop to use
        num_classes (int): the number of output classes
        device (str): 'cuda' or 'cpu'
    """
    net = attention_net(**kwargs)
    if pretrained:
        net.load_state_dict(torch.hub.load_state_dict_from_url(cub_200_2011_state_dict_url, progress=True))
        # checkpoint = 'models/nts_net_cub200.pt'
        # state_dict = torch.load(checkpoint)
        # net.load_state_dict(state_dict)
    return net
