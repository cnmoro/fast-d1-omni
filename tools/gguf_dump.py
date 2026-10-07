import struct, sys
TYPES={0:'F32',1:'F16',2:'Q4_0',3:'Q4_1',6:'Q5_0',7:'Q5_1',8:'Q8_0',30:'BF16'}
def rd(f,fmt): s=struct.calcsize('<'+fmt); return struct.unpack('<'+fmt,f.read(s))
def rstr(f): n,=rd(f,'Q'); return f.read(n).decode('utf-8','replace')
def rval(f,t):
    if t==8: return rstr(f)
    if t==9:
        et,n=rd(f,'IQ'); return [rval(f,et) for _ in range(n)]
    fm={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}[t]
    return rd(f,fm)[0]
f=open(sys.argv[1],'rb')
magic,ver,nt,nkv=rd(f,'4sIQQ'); print(magic,ver,nt,nkv)
for _ in range(nkv):
    k=rstr(f); t,=rd(f,'I'); v=rval(f,t)
    if isinstance(v,list) and len(v)>20: print(k, f'[{len(v)} items]', v[:8])
    else: print(k, v)
for _ in range(nt):
    n=rstr(f); nd,=rd(f,'I'); dims=rd(f,'Q'*nd); t,off=rd(f,'IQ')
    print(n, dims, TYPES.get(t,t), off)
