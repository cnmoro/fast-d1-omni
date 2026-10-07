import sys, json, math
def load(p): return [json.loads(l) for l in open(p) if l.strip()]
ref = load(sys.argv[1])
for other in sys.argv[2:]:
    o = load(other); errs = []
    for x, y in zip(ref, o):
        if 'error' in x or 'error' in y: continue
        for k in x['probs']:
            lp = [math.log(max(v,1e-30)) for v in x['probs'][k]]; lq = [math.log(max(v,1e-30)) for v in y['probs'][k]]
            # compare centred log-probs (logits up to a constant)
            cp = sum(lp)/len(lp); cq = sum(lq)/len(lq)
            errs.append(max(abs((a-cp)-(b-cq)) for a, b in zip(lp, lq) if a > -20))
    errs.sort()
    print(f'{other}: n={len(errs)} mean |dlogit| {sum(errs)/len(errs):.4f}  p90 {errs[int(.9*len(errs))]:.4f}  max {errs[-1]:.4f}')
