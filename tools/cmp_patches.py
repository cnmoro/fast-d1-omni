import os
import sys, numpy as np, torch, subprocess
sys.path.insert(0,os.environ.get('D1_HF_DIR', 'hf'))
from PIL import Image, ImageOps
import importlib.util
spec = importlib.util.spec_from_file_location('vision', os.path.join(os.environ.get('D1_HF_DIR', 'hf'), 'vision.py')); V = importlib.util.module_from_spec(spec); spec.loader.exec_module(V)
SP = sys.argv[2]
for p in sys.argv[1].split(','):
    out = subprocess.run([os.environ.get('D1_BIN', 'target/release/d1'),'dump-patches',p,SP+'/pt.bin'],capture_output=True,text=True)
    w,h = map(int,[l for l in out.stderr.split("\n") if l and l[0].isdigit()][0].split()[:2])
    img = ImageOps.exif_transpose(Image.open(p)).convert('RGB')
    ref_rgb = np.asarray(img)
    my_rgb = np.fromfile(SP+'/pt.bin.rgb',dtype=np.uint8).reshape(h,w,3)
    dec = np.abs(ref_rgb.astype(int)-my_rgb.astype(int))
    r = V.preprocess(img)
    ref = torch.cat([r['pixel_values'][i,:hh*ww] for i,(hh,ww) in enumerate(r['spatial_shapes'].tolist())]).numpy()
    mine = np.fromfile(SP+'/pt.bin',dtype=np.float16).astype(np.float32).reshape(-1,768)
    d = np.abs(ref-mine)*127.5
    print(f'{p}: decode max {dec.max()} mean {dec.mean():.4f} | patches {ref.shape} vs {mine.shape} max {d.max():.2f} mean {d.mean():.4f} frac>0.5: {(d>0.5).mean():.5f}', out.stdout.strip(), r['spatial_shapes'].tolist())
