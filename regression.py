from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import itertools
import math as m
from numpy.linalg import inv
Data=pd.read_csv('../credit.txt', sep=',',header=None)
iteration=100
X,Y,R=[],[],[]
f0,f1,f2=[],[],[]
cost=[]
lamba=50	#lambda
alpha=0.001  #learning rate
w0,w1,w2=1,0.05,0.05
Y1=np.ones(shape=(3,100))
Y2=np.ones(shape=(3,100))
H1=np.ones(shape=(100,1))
Z=np.ones(shape=(100,1))
weight=[0,1,1]
L=np.diag(weight)
for i in Data.itertuples():
	tmp=np.ones(4)
	tmp[0]=1
	tmp[1]=i[1]
	tmp[2]=i[2]
	tmp[3]=i[3]
	X.append(tmp)
	Y.append(i[3])


def plot_boundary(W,x1,x2):
	x1,x2=X[0],X[1]
	y=1+(w1*x1)+(w2*x2)
	return y


def gradient_descent(w0,w1,w2,X,n):
	#f(x)=w0 + w1x1 + w2x2
	for i in X:
		y=w0+(i[1]*w1)+(i[2]*w2)
		#print(y)
		h=1/(1+(m.exp(-y)) )
		#print(h)
		f0.append( (h-i[3])*i[0]	+ (lamba*w0) )
		f1.append( (h-i[3])*i[1]+ (lamba*w1) )
		f2.append( (h-i[3])*i[2]+ (lamba*w2) )
		cost.append( ((h-i[3])**2)+(lamba*( (w0**2)+(w1**2)+(w2**2))) )
	
	w0=w0-(alpha*(sum(f0)/n))
	w1=w1-(alpha*(sum(f1)/n))
	w2=w2-(alpha*(sum(f2)/n))
	cost_val=(sum(cost)/(2*n) )
	
	f0.clear()
	f1.clear()
	f2.clear()
	cost.clear()
	
	return w0,w1,w2,cost_val


def classification(w0,w1,w2, X,n):
	t=1
	p=0
	for i in X:
		y=w0+((i[1]*w1)+(i[2]*w2))
		h=1/(1+(m.exp(-y)) )
		if(h>=0.4):
			print('expected: '+str(i[3])+'  give: '+'1')
		else:
			print('expected: '+str(i[3])+'  give: '+'0')
		#print(h)
		plt.plot(t,h,'ro')
		#.label('simoid function')
		t=t+1
	
	
	

n=len(X)
for i in range(iteration):
	w0,w1,w2,cost_val=gradient_descent(w0,w1,w2,X,n)
	plt.plot(i,cost_val,'ro')




#classification(w0,w1,w2, X,n)

#plt.show()


#************************Newton Rapson Method*******************


def create_R(X):
	for i in X:
		y=1+(i[1]*w1)+(i[2]*w2)
		h=1/(1+(m.exp(-y)) )
		a=h*(1-h)
		f0.append(a)
	
	R=np.diag(f0)
	return R

def newton_raphson(H,A,W,Y,n):
	cost=[]
	penalty=[]
	A=np.matrix(A)
	W=np.matrix(W)
	Z=A*W  #100x1
	Z1=Z

	Y=np.matrix(Y)
	Y=Y.transpose()
	
	K1=(inv(H)*(A.transpose())) 
	
	i=0
	for x in np.nditer(Z):
		Z1[i]=1/(1+(m.exp(-x)) )
		i=i+1
	Z1=Z1-Y
	
	
	W=W-(K1*Z1)
	
	for i in range(n):
		cost.append(-(Z[i]*(1-Z[i])) )
	penalty.append( lamba*(((W[0]**2)+(W[1]**2)+(W[2]**2)))/2 )
	
	return W,(((sum(cost))+(sum(penalty)))/n)



R=create_R(X)

A=np.matrix(X)

A=np.delete(A,3,axis=1)

H=(A.transpose())*R*A + (lamba*L)/n


W=np.zeros((3,1))
W[0]=1
W[1]=0.5
W[2]=0.5

#print(w0,w1,w2)
for i in range(iteration):
	W,cost=newton_raphson(H,A,W,Y,n)
	#print(W)
	plt.plot(i,abs(cost),'bo')
#plt.legend(['gradient_descent', 'newton_raphson'],loc='upper right')
plt.show();
'''
print('*****************************************************************')
c=1
x1=np.linspace(1,6,100)
x2=np.linspace(1,6,100)

for i in range(len(x1)):
	for j in range(len(x2)):
		y=plot_boundary(W,x1[i],x2[i])
		plt.plot(c,y,'go')
		c=c+1
plt.show()
'''
#classification(W[0],W[1],W[2], X,n)