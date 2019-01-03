import torch
import torch.nn as nn
import torch.nn.functional as F

class ADLnet(nn.Module):
    def __init__(self, shrinkge_type = 'soft thresh', dataset = 'nus-wide-object'):
        super(ADLnet, self).__init__()
        self.dataset = dataset
        if dataset == 'flickr25k':
            inter_dim1 = 100
            class_num = 38
            view_num = 7

            self.viewdims = [192, 256, 256, 43, 150, 960, 2000]
            self.accu_dims = [192,448,704,747,897,1857,3857]
            self.dic0=nn.utils.weight_norm(nn.Linear(self.viewdims[0] ,inter_dim1), name = 'weight', dim = 0) #dim is 0 because we want the 76 to comput norms,therefore 1x50 norm
            self.dic1=nn.utils.weight_norm(nn.Linear(self.viewdims[1] ,inter_dim1), name = 'weight', dim = 0) #dim is 0 because we want the 76 to comput norms,therefore 1x50 norm
            self.dic2=nn.utils.weight_norm(nn.Linear(self.viewdims[2] ,inter_dim1), name = 'weight', dim = 0) #dim is 0 because we want the 76 to comput norms,therefore 1x50 norm
            self.dic3=nn.utils.weight_norm(nn.Linear(self.viewdims[3] ,inter_dim1), name = 'weight', dim = 0) #dim is 0 because we want the 76 to comput norms,therefore 1x50 norm
            self.dic4=nn.utils.weight_norm(nn.Linear(self.viewdims[4] ,inter_dim1), name = 'weight', dim = 0) #dim is 0 because we want the 76 to comput norms,therefore 1x50 norm
            self.dic5=nn.utils.weight_norm(nn.Linear(self.viewdims[5] ,inter_dim1), name = 'weight', dim = 0) #dim is 0 because we want the 76 to comput norms,therefore 1x50 norm
            self.dic6=nn.utils.weight_norm(nn.Linear(self.viewdims[6] ,inter_dim1), name = 'weight', dim = 0) #dim is 0 because we want the 76 to comput norms,therefore 1x50 norm
            self.W = nn.Linear(view_num * inter_dim1, class_num)

        else:
            #default nus-wide-object
            inter_dim1 = 100
            class_num = 31
            view_num = 6
            self.viewdims = [64,144,73,128,225,500]
            self.accu_dims = [64,208,281,409,634,1134]
            self.dic0=nn.utils.weight_norm(nn.Linear(self.viewdims[0] ,inter_dim1), name = 'weight', dim = 0) #dim is 0 because we want the 76 to comput norms,therefore 1x50 norm
            self.dic1=nn.utils.weight_norm(nn.Linear(self.viewdims[1] ,inter_dim1), name = 'weight', dim = 0) #dim is 0 because we want the 76 to comput norms,therefore 1x50 norm
            self.dic2=nn.utils.weight_norm(nn.Linear(self.viewdims[2] ,inter_dim1), name = 'weight', dim = 0) #dim is 0 because we want the 76 to comput norms,therefore 1x50 norm
            self.dic3=nn.utils.weight_norm(nn.Linear(self.viewdims[3] ,inter_dim1), name = 'weight', dim = 0) #dim is 0 because we want the 76 to comput norms,therefore 1x50 norm
            self.dic4=nn.utils.weight_norm(nn.Linear(self.viewdims[4] ,inter_dim1), name = 'weight', dim = 0) #dim is 0 because we want the 76 to comput norms,therefore 1x50 norm
            self.dic5=nn.utils.weight_norm(nn.Linear(self.viewdims[5] ,inter_dim1), name = 'weight', dim = 0) #dim is 0 because we want the 76 to comput norms,therefore 1x50 norm
            self.W = nn.Linear(view_num*inter_dim1, class_num)

        self.acti = nn.ReLU(False)
        self.acti_relu = nn.ReLU(True)
        self.acti3 = nn.Sigmoid()
        self._shrinkge_type = shrinkge_type
        self.shrinkge_fn = self._shrinkge()

    def forward(self, x ,is_test=False):
        if self.dataset == 'flickr25k':
            x_0 = self.dic0(x[0::,0:self.accu_dims[0]])
            x_1 = self.dic1(x[0::,self.accu_dims[0]:self.accu_dims[1]])
            x_2 = self.dic2(x[0::,self.accu_dims[1]:self.accu_dims[2]])
            x_3 = self.dic3(x[0::,self.accu_dims[2]:self.accu_dims[3]])
            x_4 = self.dic4(x[0::,self.accu_dims[3]:self.accu_dims[4]])
            x_5 = self.dic5(x[0::,self.accu_dims[4]:self.accu_dims[5]])
            x_6 = self.dic6(x[0::,self.accu_dims[5]:self.accu_dims[6]])
            x = torch.cat([x_0, x_1, x_2, x_3, x_4, x_5 ,x_6], dim=1)
        else:
            x_0 = self.dic0(x[0::,0:self.accu_dims[0]])
            x_1 = self.dic1(x[0::,self.accu_dims[0]:self.accu_dims[1]])
            x_2 = self.dic2(x[0::,self.accu_dims[1]:self.accu_dims[2]])
            x_3 = self.dic3(x[0::,self.accu_dims[2]:self.accu_dims[3]])
            x_4 = self.dic4(x[0::,self.accu_dims[3]:self.accu_dims[4]])
            x_5 = self.dic5(x[0::,self.accu_dims[4]:self.accu_dims[5]])
            x = torch.cat([x_0,x_1,x_2,x_3,x_4,x_5],dim = 1)
        x = self.shrinkge_fn(x)
        x = self.W(x)
        x = F.log_softmax(x, dim= 1)
        if(is_test):
            return x   #F.log_softmax(x, dim=1)
        else:
            return x

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