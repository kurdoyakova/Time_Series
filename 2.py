import numpy as np
import matplotlib.pyplot as plt
from scipy import odr
from scipy.fftpack import fft
from scipy.fftpack import ifft
from  os import path

def trend(B,xx):
    return B[0]+B[1]*xx

dt=1
N=2000
q=0.01
X1=9.0
A1=5.0
nu1=0.1
phi1=0
gamma=0.50
alpha=5
beta=0.0
ksi=np.random.normal(loc=0, scale=1, size=N)
x=np.zeros(N)
sigma=np.sqrt(A1*A1/(2*gamma))
y=np.zeros(N)
t=np.zeros(N)
for i in range(N):
    x[i]=(alpha+beta*dt*i+A1*np.cos(2*np.pi*nu1*dt*i-phi1)+sigma*ksi[i])+0j
    y[i]=i
    t[i]=dt*i
plt.plot(y,x.real)
plt.show()
linear_model = odr.Model(trend)
data_to_fit = odr.Data(t,x.real)
job = odr.ODR(data_to_fit, linear_model, beta0=[0.1, 0.05])
results = job.run()
#x0=x-results.beta[0]-results.beta[1]*t
x0=x
plt.plot(y,x0.real)
plt.show()

N1=2
while N1<N:
    N1=N1*2
N2=2*N1
for i in range(N2-N):
    x0=np.append(x0,0+0j)
Xj=fft(x0)
D=np.zeros(N1)
for i in range(N1):
    D[i]=(Xj[i].real*Xj[i].real+Xj[i].imag*Xj[i].imag)/(N*N)
sigma0=np.sum(x0*x0)/(N-1)
nuj=np.zeros(N1)
dnu=1/(N2*dt)
for i in range(N1):
    nuj[i]=dnu*i
plt.plot(nuj,D)
plt.axhline(y=sigma0*X1/N, color='red', linestyle='--')
plt.xlabel('Нецентрированный, alfa=5, betta=0.0, A=5, n=512')
plt.savefig(path.join("Картиночки", 'Переодограмма1212.png'))
plt.show()
plt.close()
