import torch.nn as nn

from torch import mm as mm


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.cnn1 = nn.Conv1d(in_channels=1025, out_channels=4096, kernel_size=3, stride=1, padding=1)
        #self.nl1 = nn.ReLU()
        #self.pool1 = nn.AvgPool1d(kernel_size=5)
        #self.fc1 = nn.Linear(4096*2500,2**5)
        #self.nl3 = nn.ReLU()
        #self.fc2 = nn.Linear(2**10,2**5)
    
    def forward(self, x):
        out = self.cnn1(x)
        #out = self.nl1(out)
        #out = self.pool1(out)
        out = out.view(out.size(0), -1)
        #out = self.fc1(out)
        #out = self.nl3(out)
        #out = self.fc2(out)
        return out


class GramMatrix(nn.Module):
	def forward(self, input):
		a, b, c = input.size()  # a=batch size(=1)
        # b: number of feature maps
        # (c, d): dimensions of a f. map (N=c*d)
		features = input.view(a * b, c)  # resise F_XL into \hat F_XL
		G = mm(features, features.t())  # compute the gram product
        # 'normalize' the values of the gram matrix by dividing by the number of elements in each feature maps
		return G.div(a * b * c)


class StyleLoss(nn.Module):
	def __init__(self, target, weight):
		super(StyleLoss, self).__init__()
		self.target = target.detach() * weight
		self.weight = weight
		self.gram = GramMatrix()
		self.criterion = nn.MSELoss()

	def forward(self, input):
		self.output = input.clone()
		self.G = self.gram(input)
		self.G.mul_(self.weight)
		self.loss = self.criterion(self.G, self.target)
		return self.output

	def backward(self,retain_graph=True):
		self.loss.backward(retain_graph=retain_graph)
		return self.loss
