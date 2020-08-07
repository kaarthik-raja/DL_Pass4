import numpy as np
import pandas  as pd
import os
import argparse as agp

np.random.seed(1024)

learning_rate =0.001
batch_size = 1
epochs = 3
k=10
h=128
train_path=os.path.join("..","P4Data","train.csv")
test_path=os.path.join("..","P4Data","test.csv")
loadM = 1
loadM = 0

def ParseInp():
	global learning_rate, batch_size, k, h, epochs
	global train_path,test_path,loadM

	parser = agp.ArgumentParser()
	parser.add_argument("--lr", type=float, help="the learning rate", default=0.001)
	parser.add_argument("--batch_size", type=int, help="batch size per step", default= 128)
	parser.add_argument("--k", type=int, help="Num  of Steps / train", default= 30)
	parser.add_argument("--h", type=int, help="Hidden Variables", default= 128)
	parser.add_argument("--train", type=str, help="train file location" , default=os.path.join("..","P4Data","train.csv"))
	parser.add_argument("--test", type=str, help="test file location", default= os.path.join("..","P4Data","test.csv"))
	parser.add_argument("--epochs", type=int, help="# of Epochs", default= 128)
	parser.add_argument("--load", type=int, help="load model,load-1, def-0", default= 0, choices=[0,1])

	args=parser.parse_args()
	k,h = args.k,args.h
	learning_rate=args.lr
	batch_size=args.batch_size
	epochs 	  =args.epochs
	train_path=args.train
	test_path =args.test
	loadM = args.load


def Rand_Data(trainB=1,testB=1):
	global train_X,test_X
	global train_y,test_y
	if trainB:
		perm = np.random.permutation(train_X.shape[0])
		train_X = train_X[perm]
		train_y = train_y[perm]

	if testB:
		perm = np.random.permutation(test_X.shape[0])
		test_X = test_X[perm]
		test_y = test_y[perm]

def Process_Data(trainB=1,testB=1):
	global train,test
	global train_path,test_path
	global train_X,test_X
	global train_y,test_y

	if trainB:
		train=pd.read_csv(train_path, engine='python', encoding='utf-8')
		train=train.drop(columns='id')
		train=train.values 

		train_X=train[:,0:-1]
		train_X=(train_X>127).astype(int)
		train_y=train[:,-1]


	if testB:
		test=pd.read_csv(test_path, engine='python', encoding='utf-8')
		test=test.drop(columns='id')
		test=test.values

		test_X=test[:,0:-1]
		test_X=(test_X>127).astype(int)
		test_y=test[:,-1]
	# Rand_Data()

def init():
	global W,hbias,vbias,h

	W     = np.random.rand(784,h) * np.sqrt(1/(784+h))
	hbias = np.random.rand( 1, h) * np.sqrt(1/(784+h))
	vbias = np.random.rand(1,784) * np.sqrt(1/(784+h))



def sigmoid(x):
    return np.reciprocal(np.add(1,np.exp(np.negative(x) )))

def vgh(h):
	global W, vbias
	prob = sigmoid(np.add(np.dot(h,W.T),vbias))
	sample = np.random.binomial(n=1,p=prob,size=prob.shape)
	return prob,sample

def hgv(v):
	global W, hbias
	prob = sigmoid(np.add(np.dot(v,W),hbias))
	sample = np.random.binomial(n=1,p=prob,size=prob.shape)
	return prob,sample

def CD_helper(V):
	global W, hbias , vbias,learning_rate

	iprob,isamp = hgv(V)
	hsamp = isamp

	for i in range(k):
		vprob,vsamp = vgh(hsamp)
		hprob,hsamp = hgv(vsamp)
	
	dw = learning_rate * (np.dot(V.T, iprob) - np.dot(vsamp.T, hprob)) 
	W = W + dw / batch_size
	vbias+= np.sum(learning_rate * (V - vsamp),axis=0) /batch_size
	hbias+= np.sum(learning_rate * (iprob - hprob),axis=0) /batch_size

def crossentropy(V):
	hprob,hsamp = hgv(V)
	vprob,vsamp = vgh(hprob)
	entrop = -np.mean(np.sum( np.add(V*np.log(vprob),(1 - V)*np.log(1 - vprob) ),axis=1))
	return entrop

def sampleD(V):
	hprob,hsamp = hgv(V)
	vprob,vsamp = vgh(hsamp)
	return vsamp

def saveVar(i=0):
	global W, hbias , vbias
	i = str(i)
	np.save('W'+i+'.npy', W) 
	np.save('vbias'+i+'.npy', vbias) 
	np.save('hbias'+i+'.npy', hbias) 
def loadVar(i=0):
	global W, hbias , vbias
	i = str(i)
	W =	np.load('W'+i+'.npy') 
	vbias = np.load('vbias'+i+'.npy') 
	hbias = np.load('hbias'+i+'.npy') 

def main():
	global train_X,train_y,test_X,test_y
	global h,k
	print("Training Started")

	Ltrain = train_X.shape[0]
	Ltest = test_X.shape[0]
	nof_batch_train = int(Ltrain / batch_size)
	bsz = batch_size
	# loadVar()


	if loadM:
		loadVar()
		print("loaded")
	for qq in [1,20]:
		k=qq
		for q in [64]:
			h = q
			init()

			for i in range(epochs):
				Rand_Data()
				for j in range(nof_batch_train):
					input_Data = train_X[bsz*j:(bsz)*(j+1)]
					CD_helper(input_Data)

					if j%10000==0:
						print("->",i,":",j," k:",k," h:",h," Error:",crossentropy(test_X))

				# if i%3:
						saveVar(q*100+k) 

			# loadVar(0)



if __name__ == '__main__':
	# ParseInp()
	Process_Data()
	main()