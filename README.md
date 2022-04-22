# pybolima

### Prerequisites

### Install Torch (needed by Stanza tagger)

```bash
pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

### Tag BoLiMa

```bash
PYTHONPATH=. python scripts/main.py /data/westac/blm/blm.csv ./data/blm_tagged --force
```
