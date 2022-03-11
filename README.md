[![Open on Torch Hub](https://img.shields.io/badge/Torch-Hub-red?logo=pytorch)](https://pytorch.org/hub/nicolalandro_ntsnet-cub200_ntsnet/) 
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgray)](LICENSE) 
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/are-these-birds-similar-learning-branched/fine-grained-image-classification-on-cub-200-1)](https://paperswithcode.com/sota/fine-grained-image-classification-on-cub-200-1?p=are-these-birds-similar-learning-branched)


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pytorch/pytorch.github.io/blob/master/assets/hub/nicolalandro_ntsnet-cub200_ntsnet.ipynb)

# NtsNet pre trained on cub200 2011
A pretrained ntsnet

## How to use
```python
net = netsnet(pretrained=True, **{'topN': 6, 'device':'cpu', 'num_classes': 200})
```

## Citation
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
![](images/nts-net.png)


