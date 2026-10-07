import sys, json
a=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]; b=[json.loads(l) for l in open(sys.argv[2]) if l.strip()]
assert len(a)==len(b), (len(a),len(b))
maxd=0; tops=0; n=0; tokd=0; errs=0; worst=[]
for i,(x,y) in enumerate(zip(a,b)):
    if 'error' in x or 'error' in y:
        if ('error' in x) != ('error' in y): print('error mismatch', i, x.get('error'), y.get('error')); errs+=1
        continue
    if x['input_tokens']!=y['input_tokens']: tokd+=1; print('token count mismatch', i, x['input_tokens'], y['input_tokens'])
    for k in x['probs']:
        p,q=x['probs'][k],y['probs'][k]; n+=1
        d=max(abs(u-v) for u,v in zip(p,q)); maxd=max(maxd,d); worst.append((d,i,k))
        if max(range(len(p)),key=p.__getitem__)!=max(range(len(q)),key=q.__getitem__): tops+=1; print('top answer differs', i, k, [round(v,4) for v in p], [round(v,4) for v in q])
worst.sort(reverse=True)
print(f'{n} questions: max |dp| = {maxd:.5f}, mean = {sum(w[0] for w in worst)/max(n,1):.6f}, top-answer mismatches = {tops}, token-count mismatches = {tokd}, error mismatches = {errs}')
print('worst:', [(round(d,5),i,k) for d,i,k in worst[:5]])
