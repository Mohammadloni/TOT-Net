import torch.nn.functional as F
import torch.nn as nn


def swish(x, activationLeft='PRelu', activationRight='PRelu', w=None):
    
    #print("activation_Left: %s , activation_Right: %s" %(activationLeft,activationRight))

    if (activationLeft=='leaky_relu' and activationRight == 'leaky_relu'):
	activationRight= 'relu'
    else:
        activationRight= 'leaky_relu'

    if (activationLeft=='elu'):
        outleft= F.elu(x)
    elif (activationLeft=='elish'):
        outleft= F.elu(x) * F.sigmoid(x)
    elif (activationLeft=='hardtanh'):
        outleft= F.hardtanh(x)
    elif (activationLeft=='relu'):
        outleft= F.relu(x)
    elif (activationLeft=='tanh'):
        outleft= F.tanh(x)
    elif (activationLeft=='sigmoid'):
        outleft= F.sigmoid(x)
    elif (activationLeft == 'swish'):
        outleft = x * F.sigmoid(x)
    elif (activationLeft == 'selu'):
        outleft = F.selu(x)
    elif (activationLeft == 'prelu'):
        outleft = F.prelu(x,w)
    elif (activationLeft == 'leaky_relu'):
        outleft = F.leaky_relu(x, 0.2, False)
    else:
        outleft = F.prelu(x) #nn.PReLU(x, 0.2) 


    if (activationRight == 'elu'):
        outright = F.elu(x)
    elif (activationRight == 'relu'):
        outright = F.relu(x)
    elif (activationRight=='elish'):
        outleft= x * F.sigmoid(x)
    elif (activationRight=='hardtanh'):
        outleft= F.hardtanh(x)
    elif (activationRight == 'tanh'):
        outright = F.tanh(x)
    elif (activationRight == 'sigmoid'):
        outright = F.sigmoid(x)
    elif (activationRight == 'swish'):
        outright = x * F.sigmoid(x)
    elif (activationRight == 'selu'):
        outright = F.selu(x)
    elif (activationRight == 'prelu'):
        outright = F.prelu(x,w) #nn.PReLU(x) 
    elif (activationRight == 'leaky_relu'):
        outright = F.leaky_relu(x, 0.2, False)
    else:
        outright =F.prelu(x)  #nn.PReLU(x, 0.2) 
    
    
  
    x_where_self_x_is_positive = outright  * (x >= 0).float()
    x_where_self_x_is_negative = outleft  * (x < 0).float()
    return (x_where_self_x_is_positive) + (x_where_self_x_is_negative)

    	
    
