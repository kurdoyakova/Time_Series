import numpy as np
import math
from matplotlib import pyplot as plt
from scipy.linalg import *
from scipy.fftpack import fft,ifft
from  os import path

n=512
dt=1
q=0.01
xf=9.0
a1=1.0
nu1=0.1
fi1=0
gamma=0.5
alfa=0.1
betta=0.05
k=[]
tk=[]
x=[]
x0=[]
x1=[]
d=[]
d2=[]
sig=math.sqrt(a1*a1/(2*gamma))
csi=np.random.normal(loc=0, scale=1, size=n)

for i in range(0,n):
    k.append(i)
    tk.append(dt*k[i])

for i in range(0,n):
    x.append(alfa+betta*tk[i]+a1*math.cos(2*math.pi*nu1*tk[i])+sig*csi[i])
#plt.plot(tk,x)
#plt.show()

#trend
aa=np.vstack([tk,np.ones(n)]).T
b=np.linalg.lstsq(aa,x)[0][0]
c=np.linalg.lstsq(aa,x)[0][1]

for i in range(0,n):
    x1.append(x[i]-b*tk[i]-c)
#plt.plot(tk,x1)
#plt.show()

#center
m=math.fsum(x1)/n
for i in range(0,n):
    x0.append(x1[i]-m)
#plt.plot(tk,x)
#plt.show()
#plt.plot(tk,x0)
#plt.show()

#pereod
n1=256
n2=2*n1
x2=[]
for i in range(n,n2):
    x0.append(0)
    x.append(0)
for i in range(0,n2):
    x1=np.array(x0, dtype=np.complex)
    x2=np.array(x, dtype=np.complex)

xx2=fft(x2)
xx1=fft(x1)
for i in range(0,n1):
    d.append((xx1[i].imag**2+xx1[i].real**2)/(n*n))
    d2.append((xx2[i].imag**2+xx2[i].real**2)/(n*n))
nu=[]
for i in range(0,n1):
    nu.append(i)
    nu[i]=nu[i]/n2

#critic level
f=0
f1=0
for i in range(0,n):
    f=f+x0[i]**2
for i in range(0,n):
    f1=f1+x[i]**2
sig0=f/n
sig01=f1/n
yr=sig0*xf/(n+1)
yr1=sig01*xf/(n+1)
yrr=[]
yrr1=[]
for i in range(0,n1):
    yrr.append(yr)
for i in range(0,n1):
    yrr1.append(yr1)

#first=plt.plot(nu,d)
#first2=plt.plot(nu,d2)
#second=plt.plot(nu,yrr)
#plt.legend(first,second)
#plt.xlabel('Центрированный, alfa=0.1, betta=0.05, A=1, n=512')
#plt.show()
#plt.savefig(path.join("Картиночки", 'Переодограмма3.png'))

first2=plt.plot(nu,d2)
second2=plt.plot(nu,yrr1)
plt.legend(first2,second2)
plt.xlabel('Нецентрированный, alfa=0.1, betta=0.05, A=1, n=512')
#plt.show()
plt.savefig(path.join("Картиночки", 'Переодограмма10.png'))
plt.close()






#corell
nz=n//2
xxc=[]
for i in range(0,n2):
    xxc.append(abs(xx1[i])**2)

xxx=ifft(xxc)
cm=[]
tkm=[]
for i in range(0,n):
    cm.append(xxx[i].real/n)

#plt.plot(range(0,len(cm)),cm)
#plt.show()

#sglag pereod
w=[]
cmw=[]
am=0.25
for i in range(0,nz):
    w.append((1-2*am)+2*am*math.cos(math.pi*i/nz))
    cmw.append(cm[i]*w[i])
for i in range(len(cmw),n2):
    cmw.append(0)

cmm=fft(cmw)
dz=[]
for i in range(0,n1):
    dz.append((2*cmm[i].real-cmw[0])/(nz+1))
#plt.plot(nu,dz)
#plt.show()
