import torch.nn as nn
import numpy
import torch
class BinOp():
    def __init__(self, model):
        # count the number of Conv2d
        count_Conv2d = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                count_Conv2d = count_Conv2d + 1

        start_range = 1
        end_range = count_Conv2d-2
        self.bin_range = numpy.linspace(start_range,
                end_range, end_range-start_range+1)\
                        .astype('int').tolist()
        self.num_of_params = len(self.bin_range)
        self.saved_params = []
        self.target_params = []
        self.target_modules = []
        index = -1
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                index = index + 1
                if index in self.bin_range:
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight)

    def binarization(self):
        self.meancenterConvParams()
        self.clampConvParams()
        self.save_params()
        #self.binarizeConvParams()
	self.TernarizeWeights()


    def meancenterConvParams(self):
        for index in range(self.num_of_params):
            s = self.target_modules[index].data.size()
            negMean = self.target_modules[index].data.mean(1, keepdim=True).\
                    mul(-1).expand_as(self.target_modules[index].data)
            self.target_modules[index].data = self.target_modules[index].data.add(negMean)

    def clampConvParams(self):
        for index in range(self.num_of_params):
            torch.clamp(self.target_modules[index].data, -1.0, 1.0, out=self.target_modules[index].data)

    def save_params(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    '''def binarizeConvParams(self):
        for index in range(self.num_of_params):
            n = self.target_modules[index].data[0].nelement()
            s = self.target_modules[index].data.size()
            m = self.target_modules[index].data.norm(1, 3, keepdim=True)\
                    .sum(2, keepdim=True).sum(1, keepdim=True).div(n)
            self.target_modules[index].data.sign()\
                    .mul(m.expand(s), out=self.target_modules[index].data)'''
    def Delta(self,tensor):
        n = tensor[0].nelement() 
	s = tensor.size()
        delta = 0.7 * tensor.norm(1,3).sum(2).sum(1).div(n)
        return delta
	    
###########################################torch.mul(self.target_modules[index].data.sign(),m.expand(s), out=self.target_modules[index].data)### mul sign * average of tensor in the s size (weight*alpha)

    def Alpha(self,tensor,delta):
        Alpha = []
        for i in range(tensor.size()[0]):
            count = 0
            abssum = 0
            absvalue = tensor[i].view(1,-1).abs()
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

    def Ternarize(self,tensor):
	#tensor = tensor.cpu()
	output = torch.zeros(tensor.size())
        output = output.type(torch.cuda.FloatTensor)
        delta = self.Delta(tensor)
        alpha = self.Alpha(tensor,delta)
        for i in range(tensor.size()[0]):
            for w in tensor[i].view(1,-1):
                pos_one = (w > delta[i]).type(torch.cuda.FloatTensor)
                neg_one = torch.mul((w < -delta[i]).type(torch.cuda.FloatTensor),-1)
            out = torch.add(pos_one,neg_one).view(tensor.size()[1:])
            output[i] = torch.add(output[i],torch.mul(out,alpha[i]))
        return output.cuda()###

    def TernarizeWeights(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data = self.Ternarize(self.target_modules[index].data)

    def restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])


    '''def updateBinaryGradWeight(self):
        for index in range(self.num_of_params):
            weight = self.target_modules[index].data
            n = weight[0].nelement() 
            s = weight.size()
            delgrad = 0.7 * weight.norm(1, 3)\
                        .sum(2).sum(1).div(n)#.expand(s)
            Alpha = []
            for i in range(weight.size()[0]):
                count = 0
                abssum = 0
                absvalue = weight[i].view(1,-1).abs()
                for w in absvalue:
                    truth_value = w > delgrad[i] #print to see
                count = truth_value.sum()
                count = count.type(torch.cuda.FloatTensor)
                abssum = torch.matmul(absvalue,truth_value.type(torch.cuda.FloatTensor).view(-1,1))
                Alpha.append(abssum/count)
            alpha = Alpha[0]
            for i in range(len(Alpha) - 1):
                alpha = torch.cat((alpha,Alpha[i+1]))
            #if len(s) == 4:
            alpha1 = alpha[:,:,None,None]
            alpha1 = alpha1.expand(s)
            elif len(s) == 2:#####
                #alpha1 = alpha[:,:,None,None]####
                alpha1 = alpha.expand(s)###
            
            alpha1[weight.lt(-1.0)] = 0 
            alpha1[weight.gt(1.0)] = 0  #m = alpha
            alpha1 = alpha1.mul(self.target_modules[index].grad.data)#alpha avaz shod

            #weight = weight.cpu()
	    #output = torch.zeros(weight.size())
            outweight = torch.zeros(weight.size())
            outweight = outweight.type(torch.cuda.FloatTensor)
            #delta = self.Delta(tensor)
            
            #print ('***********delta size********',delta.size())
	    #print ('***********alpha size********',alpha.size())
            for i in range(weight.size()[0]):
                for w in weight[i].view(1,-1):
                    pos_one = (w > delgrad[i]).type(torch.cuda.FloatTensor)
                    neg_one = torch.mul((w < - delgrad[i]).type(torch.cuda.FloatTensor),-1)
                out = torch.add(pos_one,neg_one).view(weight.size()[1:])
	        outweight[i] = torch.add(outweight[i],out)
                #output[i] = torch.add(output[i],torch.mul(out,alpha[i]))
            #print('**********output*********',outweight)
            #print('**********outputsize*********',outweight.size())
            #outweight = weight.sign()
	   
            
            m_add = outweight.mul(self.target_modules[index].grad.data)
            m_add = m_add.sum(3, keepdim=True)\
                    .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            m_add = m_add.mul(outweight)
            self.target_modules[index].grad.data = alpha1.add(m_add).mul(1.0-1.0/s[1]).mul(n)'''

    def updateBinaryGradWeight(self):
        for index in range(self.num_of_params):
            weight = self.target_modules[index].data
            n = weight[0].nelement()
            s = weight.size()
            m = weight.norm(1, 3, keepdim=True)\
                    .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            m[weight.lt(-1.0)] = 0 
            m[weight.gt(1.0)] = 0
            # m = m.add(1.0/n).mul(1.0-1.0/s[1]).mul(n)
            # self.target_modules[index].grad.data = \
            #         self.target_modules[index].grad.data.mul(m)
            m = m.mul(self.target_modules[index].grad.data)
            m_add = weight.sign().mul(self.target_modules[index].grad.data)
            m_add = m_add.sum(3, keepdim=True)\
                    .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            m_add = m_add.mul(weight.sign())
            self.target_modules[index].grad.data = m.add(m_add).mul(1.0-1.0/s[1]).mul(n)
