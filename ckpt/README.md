To sample molecules using pretrained models, please first download the pretrained model weights from [here](https://drive.google.com/drive/folders/1zTrjVehEGTP7sN3DB5jaaUuMJ6Ah0-ps?usp=sharing) and put them in the `./ckpt` folder. There are three model weight files: 
- `MolDiff.pt`: the pretrained complete MolDiff model.
- `MolDiff_simple.pt`: the pretrained simplified MolDiff model that was trained without using the new bond noise schedule.
- `bond_predictor.pt`: the pretrained bond predictor that is used for bond guidance during sampling.

```bash
ckpt
├── MolDiff.pt
├── MolDiff_simple.pt
└── bond_predictor.pt
```
