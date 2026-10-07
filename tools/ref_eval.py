import os
# Reference probabilities for a JSONL of requests (images/files hold local paths). Usage: ref_eval.py cases.jsonl [dtype]
import sys, json, torch, numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import ref_model
from PIL import Image
import soundfile as sf
dt = getattr(torch, sys.argv[2]) if len(sys.argv) > 2 else torch.float32
m = ref_model.load(dt)
P = sys.modules[type(m).__module__.rsplit('.',1)[0]+'.prompt']
for line in open(sys.argv[1]):
    if not line.strip(): continue
    r = json.loads(line)
    imgs = [Image.open(p) for p in r.get('images', [])] or None
    if imgs:
        from PIL import ImageOps
        imgs = [ImageOps.exif_transpose(i) for i in imgs]
    aud = None
    if r.get('files'):
        a, sr = sf.read(r['files'][0], dtype='int16'); assert sr == 16000
        if a.ndim > 1: a = a.mean(1).astype(np.int16)
        aud = a
    try:
        (probs, n), = m.probabilities_batch([(r.get('state'), [r['questions'][k] for k in r['questions']], imgs, aud)], _usage=True)
        print(json.dumps({"probs": {k: p for k, p in zip(r['questions'], probs)}, "input_tokens": n}))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
    sys.stdout.flush()
