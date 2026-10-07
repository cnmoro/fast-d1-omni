import os
import torch, warnings, logging
from transformers import AutoModel
from safetensors.torch import load_file
def load(dt=torch.float32, dev='cuda'):
    m=AutoModel.from_pretrained(os.environ.get('D1_HF_DIR', 'hf'), trust_remote_code=True, dtype=dt)
    S=load_file(os.path.join(os.environ.get('D1_HF_DIR', 'hf'), 'model.safetensors'))
    sd={k[len('vision.tower.vision_model.'):]:v for k,v in S.items() if k.startswith('vision.tower.vision_model.')}
    missing,unexp=m.vision.tower.load_state_dict(sd, strict=False)
    assert not unexp, unexp
    assert all('head' in k for k in missing), missing
    return m.to(dev, dtype=dt).eval()
