---
layout: hub_detail
background-class: hub-background
body-class: hub
category: researchers
title: ntsnet
summary: a fine grane model for image classification.
image: images/nts-net.png
author: Moreno Carraffini and Nicola Landro
tags: [fine grane classification, image classification]
github-link: https://github.com/nicolalandro/ntsnet_cub200
featured_image_1: no-image
featured_image_2: no-image
accelerator: "cuda-optional"
---

```python
# Preprocessing image of cube
# transform_train = transforms.Compose([
#     transforms.Resize((600, 600), Image.BILINEAR),
#     transforms.CenterCrop((448, 448)),
#     transforms.RandomHorizontalFlip(),  # solo se train
#     transforms.ToTensor(),
#     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
# ])
import torch
mdoel = torch.hub.load('nicolalandro/ntsnet_cub200/', 'ntsnet', pretrained=True, **{'topN': 6, 'device':'cpu', 'num_classes': 200})
top_n_coordinates, concat_out, raw_logits, concat_logits, part_logits, top_n_index, top_n_prob = model(img)
# if you need the output props use "concat_out"
```

### Model Description
This is a nts-net pretrained with CUB200 2011 dataset. A fine grane dataset of birds species.

### References
You can read the full paper at this [link](http://artelab.dista.uninsubria.it/res/research/papers/2019/2019-IVCNZ-Nawaz-Birds.pdf).
```bibtex
@INPROCEEDINGS{Gallo:2019:IVCNZ, 
  author={Nawaz, Shah and Calefati, Alessandro and Caraffini, Moreno and Landro, Nicola and Gallo, Ignazio},
  booktitle={2019 International Conference on Image and Vision Computing New Zealand (IVCNZ 2019)}, 
  title={Are These Birds Similar: Learning Branched Networks for Fine-grained Representations},
  year={2019}, 
  month={Dec},
}
```