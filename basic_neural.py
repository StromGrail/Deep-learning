from numpy import exp,array,random,dot

def sigmoid(x,derivative=False):
	if derivative==True:
		return x*(1-x)
	return 1/(1+exp(-x))

if __name__ == '__main__':
	training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
	training_set_outputs = array([[0, 1, 1, 0]]).T

	random.seed(1)
	synaptic_weights0 = 2*random.random((3,1)) - 1
	print(synaptic_weights0)
	l1=[]
	for i in range(600000):
		l0=training_set_inputs
		l1=sigmoid(dot(l0,synaptic_weights0))
		#print("l1 ",l1)
		error=(training_set_outputs-l1)
		#print("error",error*sigmoid(l1,True))
		adjust=dot(l0.T,error*sigmoid(l1,True))
		#print('adjust',adjust)
		synaptic_weights0+=adjust
		if i%100000==0:	
			print("new synaptic_weights0 ",synaptic_weights0)
	print(l1)