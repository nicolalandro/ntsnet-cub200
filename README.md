# NtsNet pre trained on cub200 2011
A pretrained ntsnet

![][images/nts-net.png]


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

