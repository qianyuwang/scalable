import torch
import torch.nn as nn
import torch.nn.functional as F

class ADLnet(nn.Module):
    def __init__(self, shrinkge_type = 'soft thresh',input_dim1 = [], input_dim2=31):
        super(ADLnet, self).__init__()
        self.viewdim = [64,144,73,128,225,500]
        self.dims = [64,208,281,409,634,1134]
        self.dic = nn.Linear(input_dim1,100)
        self.dictionary =nn.utils.weight_norm(self.dic, name = 'weight', dim = 0) #dim is 0 because we want the 76 to comput norms,therefore 1x50 norm
        self.normalize()
        self.acti = nn.ReLU(True)
        self.W = nn.Linear(100,input_dim2)
        self.acti_relu = nn.ReLU(True)
        self.acti3 = nn.Sigmoid()
        self._shrinkge_type = shrinkge_type
        self.shrinkge_fn = self._shrinkge()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.1)
                m.bias.data.zero_()

    def _shrinkge(self):
        if self._shrinkge_type == 'soft thresh':
            return self._soft_thrsh
        elif self._shrinkge_type == 'smooth soft thresh':
            return self._smooth_soft_thrsh
        else:
            raise NotImplementedError('Double Tanh not implemented')

    def _smooth_soft_thrsh(self, X, beta=5 , b=0.5):
        """
        X  - Input
        theta - tuple(beta, b)
        beta controls the smoothness of the kink of shrinkage operator,
        and b controls the location of the kink
        """
        def smooth_relu(x, beta, b):
            first = beta * b * torch.ones_like(x)
            second = beta * x
            third = torch.zeros_like(x)
            first = first.unsqueeze(0)
            second = second.unsqueeze(0)
            third = third.unsqueeze(0)
            logsumexp = torch.log(torch.sum(torch.exp(torch.cat([first,second, third], 0 )),dim = 0))
            # TODO: logsum exp works well for overflow but seems to cause underflow as well
            return (1 / beta) * logsumexp - b

        return smooth_relu(X, beta, b) - smooth_relu(-X, beta, b)

    def _soft_thrsh(self, X, theta = 0.05):
        out = self.acti_relu(X - theta)-self.acti_relu(-X - theta)
        return out/2.0


    def normalize(self):
        norms = torch.norm(self.dic.weight,dim =1)
        norm_repeat = norms.repeat(self.dic.weight.shape[1],1)
        norm_repeat = torch.transpose(norm_repeat, 0, 1)
        self.dic.weight.data = self.dic.weight.data/norm_repeat

    def forward(self, x ,is_test=False):
        #print('1 norm',torch.norm(self.dictionary.weight,dim=1))

        #print(self.dictionary.weight_v)
        x = self.dictionary(x)
        #self.normalize()
        #x = self.dic(x)
        x = self.shrinkge_fn(x)
        x = self.W(x)
        # x = self.acti2(x)
        # print('x',x)
        # print('F.softmax(x, dim=1)',F.softmax(x, dim=1))
        # x = self.acti3(x)
        x = F.log_softmax(x, dim= 1)
        if(is_test):
            # print('is_test')
            # print('x',x)
            return x   #F.log_softmax(x, dim=1)
        else:
            return x
