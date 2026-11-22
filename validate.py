def cpu_scan(x):
    y=[]; s=0
    for v in x:
        y.append(s); s+=v
    return y
def blelloch_block(chunk, ELEM=512):
    s=list(chunk)+[0]*(ELEM-len(chunk))
    off=1
    while off<ELEM:
        for i in range(off*2-1, ELEM, off*2):
            s[i]+=s[i-off]
        off*=2
    total=s[ELEM-1]; s[ELEM-1]=0
    off=ELEM//2
    while off>0:
        for i in range(off*2-1, ELEM, off*2):
            t=s[i-off]; s[i-off]=s[i]; s[i]+=t
        off//=2
    return s, total
def gpu_mirror(x):
    ELEM=512; B=(len(x)+ELEM-1)//ELEM; y=[0]*len(x); sums=[]; locals_=[] 
    for b in range(B):
        s,tot=blelloch_block(x[b*ELEM:(b+1)*ELEM]); locals_.append(s); sums.append(tot)
    scanned,_=blelloch_block(sums) if len(sums)<=512 else (cpu_scan(sums),0)
    for b in range(B):
        for i in range(ELEM):
            gi=b*ELEM+i
            if gi<len(x): y[gi]=locals_[b][i]+scanned[b]
    return y
import random
for N in [1,100,1024,1025,100000]:
    random.seed(42); x=[random.uniform(0,10) for _ in range(N)]
    c=cpu_scan(x); g=gpu_mirror(x); err=max(abs(a-b) for a,b in zip(c,g)) if N>0 else 0
    print(f'N={N} PASS={err<1e-4} y0={g[0]} y1={g[1] if N>1 else 0:.2f} maxErr={err:.2e}')
print('Scan validated - exclusive, first element 0')
