import torch.nn as nn
import torch
import torch.nn.functional as F
from activation import swish

class BinActive(torch.autograd.Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension.
    '''

    '''def forward(self, input):
        self.save_for_backward(input)
        size = input.size()
        mean = torch.mean(input.abs(), 1, keepdim=True)
        input = input.sign()
        return input, mean

    def backward(self, grad_output, grad_output_mean):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input'''
    def forward(self, input):
        self.save_for_backward(input)
        size = input.size()
        mean = torch.mean(input.abs(), 1, keepdim=True)
        #input = self.Delta(input)
        input = self.Ternarize(input)
        #print ('************************',input.size())
        return input, mean

    '''def Delta(self,input):
        n = input[0].nelement() 
        s = input.size()
        #delta = 0.7 * input.norm(1,3).sum(2).sum(1).div(n)
        outinput = torch.zeros(input.size())
        outinput = outinput.type(torch.cuda.FloatTensor)
        for i in range(input.size()[0]):
            for w in input[i].view(1,-1):
                #pos_one = (w > delta[i]).type(torch.cuda.FloatTensor)
                #neg_one = torch.mul((w < - delta[i]).type(torch.cuda.FloatTensor),-1)
                pos_one = (w > 0.33 ).type(torch.cuda.FloatTensor)
                neg_one = torch.mul((w < -0.33).type(torch.cuda.FloatTensor),-1)
		#print ('******************', delta[i])
            out = torch.add(pos_one,neg_one).view(input.size()[1:])
	    outinput[i] = torch.add(outinput[i],out)
        return outinput'''
    def Delta(self,input):
        #for index in range(self.num_of_params):#2
	    #print ('222222222222dataaaaaaaaaaaaaaaaaaaaaaaaaaaaa',self.target_modules[index].data)
        n = input[0].nelement() # 1 + train epoch #n= 500 #n = 800 
           # print ('nnnnnnnnnnnnnnnnnnnnnnnnnnnn',n)
	s = input.size()# 1 + train epoch s=(50,20,5,5)(25000) #s =(500,800) (400000)
        if len(s) == 4: # binconv layer
            #delta = 0.7 * self.target_modules[index].data.norm(1, 3, keepdim=True)\
                 #       .sum(2, keepdim=True).sum(1, keepdim=True).div(n)#alpha  average of norm of input 2D
            delta = 0.7 * input.norm(1,3).sum(2).sum(1).div(n)
	       # print('mmmmmmmmmmmmmmmmmmmm',m.max())#(50, 1, 1, 1) # alpha for 50 filter(20*5*5)
        elif len(s) == 2:
                #delta = 0.7 * self.target_modules[index].data.norm(1, 1, keepdim=True).div(n)#alpha   average of norm in FC
	    delta = 0.7 * input.norm(1,1).div(n)
		#print('mmmmmmmmmmmmmmmmmmmm',m.size())#(500,1)
	        #print ('signdataaaaaaaaaaaaaaaaaaaaaaaaaa',self.target_modules[index].data.sign())
                #print ('sizeeeesigndataaaaaaaaaaaaaaaaaaaaaaaaaa',self.target_modules[index].data.sign().size())#(500,800)
        return delta
	    ###########################################torch.mul(self.target_modules[index].data.sign(),m.expand(s), out=self.target_modules[index].data)### mul sign * average of tensor in the s size (weight*alpha)

    def Alpha(self,input,delta):
        Alpha = []
	#for index in range(self.num_of_params):
        for i in range(input.size()[0]):
            count = 0
            abssum = 0
            absvalue = input[i].view(1,-1).abs()
            for w in absvalue:
                truth_value = w > delta[i] #print to see
            count = truth_value.sum()
            count = count.type(torch.cuda.FloatTensor)
            abssum = torch.matmul(absvalue,truth_value.type(torch.cuda.FloatTensor).view(-1,1))
            Alpha.append(abssum/count)
        alpha = Alpha[0]
        for i in range(len(Alpha) - 1):
            alpha = torch.cat((alpha,Alpha[i+1]))
        return alpha##

    def Ternarize(self,input):
	#for index in range(self.num_of_params):
	    #tensor = tensor.cpu()
	    #print('tensorrrrrrrrrrrrrrrrrrrrrrrrrrr',self.target_modules[index].data.size())
        outinput = torch.zeros(input.size())
        outinput = outinput.type(torch.cuda.FloatTensor)
        delta = self.Delta(input)
        alpha = self.Alpha(input,delta)
            #print ('***********delta size********',delta.size())
	    #print ('***********alpha size********',alpha.size())
        for i in range(input.size()[0]):
            for w in input[i].view(1,-1):
                pos_one = (w > delta[i]).type(torch.cuda.FloatTensor)
                neg_one = torch.mul((w < -delta[i]).type(torch.cuda.FloatTensor),-1)
            out = torch.add(pos_one,neg_one).view(input.size()[1:])
            outinput[i] = torch.add(outinput[i],torch.mul(out,alpha[i]))
        #print('**********output*********',output.size())
        return outinput#.cuda()###

    def backward(self, grad_output, grad_output_mean):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0 # -1<input<1
        grad_input[input.le(-1)] = 0
        return grad_input
 

class BinConv2d(nn.Module):
    def __init__(self, input_channels, output_channels,
            kernel_size=-1, stride=-1, padding=-1, dropout=0,Left='prelu',Right='prelu' ):
        super(BinConv2d, self).__init__()
        self.layer_type = 'BinConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout
        self.actl= Left
        self.actr= Right
        self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
        if dropout!=0:
            self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv2d(input_channels, output_channels,
                kernel_size=kernel_size, stride=stride, padding=padding)
        #self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.bn(x)
        x ,mean = BinActive()(x)
        if self.dropout_ratio!=0:
            x = self.dropout(x)
        x = self.conv(x)
        weight = torch.cuda.FloatTensor([0.25])
        x = swish(x, activationLeft=self.actl, activationRight=self.actr, w=weight)
        return x

class Net(nn.Module):
    def __init__(self,Left='prelu',Right='prelu'):
	    
        super(Net, self).__init__()
        self.xnor = nn.Sequential(
                nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False),
                #nn.ReLU(inplace=True),
                BinConv2d(192, 160, kernel_size=1, stride=1, padding=0, Left=Left, Right=Right),
                BinConv2d(160,  96, kernel_size=1, stride=1, padding=0, Left=Left, Right=Right),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

                BinConv2d( 96, 192, kernel_size=5, stride=1, padding=2, dropout=0.5, Left=Left, Right=Right),
                BinConv2d(192, 192, kernel_size=1, stride=1, padding=0, Left=Left, Right=Right),
                BinConv2d(192, 192, kernel_size=1, stride=1, padding=0, Left=Left, Right=Right),
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),

                BinConv2d(192, 192, kernel_size=3, stride=1, padding=1, dropout=0.5, Left=Left, Right=Right),
                BinConv2d(192, 192, kernel_size=1, stride=1, padding=0, Left=Left, Right=Right),
                nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False),
                nn.Conv2d(192,  10, kernel_size=1, stride=1, padding=0),
                #nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=8, stride=1, padding=0),
                )

    def forward(self, x):
        x = self.xnor(x)
        x = x.view(x.size(0), 10)
        return x
