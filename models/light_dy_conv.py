"""
Borrowed from https://github.com/MingjieChen/DYGANVC
"""


import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

class LightConv(nn.Module):    
    def __init__(self, kernel_size, head_size, num_heads):

        super(LightConv, self).__init__()
        self.head_size = head_size

        self.kernel_size = kernel_size
        self.unfold1d = nn.Unfold(kernel_size = [self.kernel_size, 1], padding = [self.kernel_size //2, 0]) 
        self.bias = nn.Parameter(torch.zeros(num_heads * head_size), requires_grad = True)
    def forward(self, x, filters):
        # x: [B,T,C_in]
        # filters: [B,T,num_heads*kernel_size]
        # return: [B,T, num_heads*head_size]
        B,T,_ = x.size()
        conv_kernels = filters.reshape(-1,self.kernel_size,1)
        conv_kernels = torch.softmax(conv_kernels, dim = 1)

        unfold_conv_out = self.unfold1d(x.transpose(1,2).contiguous().unsqueeze(-1))
        unfold_conv_out = unfold_conv_out.transpose(1,2).reshape(B,T,-1,self.kernel_size)

        conv_out = unfold_conv_out.reshape(-1, self.head_size, self.kernel_size)
        conv_out = torch.matmul(conv_out, conv_kernels)
        conv_out = conv_out.reshape(B,T,-1)
        conv_out += self.bias.view(1,1,-1)
        return conv_out


class DynamicConv(nn.Module):   
    def __init__(self, dim_in, dim_out, kernel_size = 3, num_heads = 8, res = True, ln = True):
        super(DynamicConv, self).__init__()
        self.dim_out = dim_out*2
        self.dim_in = dim_in
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.res = res
        self.use_ln = ln
        self.k_layer = nn.Linear(dim_in, self.dim_out)
        # print("self.dim_out: ", self.dim_out)
        self.conv_kernel_layer = nn.Linear(dim_out, kernel_size*num_heads)
        # print("dim_out: ", dim_out)
        
        self.lconv = LightConv(kernel_size, dim_out  // num_heads, num_heads)
        if self.use_ln:
            self.ln = nn.LayerNorm(dim_out)
        self.act = nn.GLU(dim = -1)
    
    def forward(self, inputs ):
        x = inputs
        # x: [B,Cin,T]
        # spk_src: we don't use it here
        # spk_trg: [B, spk_emb_dim]
        # return: [B,Cout,T]
        # x = x.transpose(1,2)
        B,T,C = x.size()
        residual = x
        if self.use_ln:
            x = self.ln(x)
        k = self.act(self.k_layer(x))
        # print("k: ", k.shape)
        # generate light weight conv kernels 
        weights = self.conv_kernel_layer(k) # [B,T, dim_in] -> [B,T,num_heads*kernel_size]
        # print("weights: ", weights.shape)
        weights = weights.view(B, T, self.num_heads, self.kernel_size)
        # conduct conv
        layer_out = self.lconv(k, weights) 
        if self.res:
            layer_out = layer_out + residual    
        # return layer_out.transpose(1,2)
        return layer_out

class Depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size = 3, padding = 1, bias=False):
        super(Depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv1d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias)
        self.pointwise = nn.Conv1d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

if __name__ == '__main__':
    test = DynamicConv(128, 128, kernel_size = 3, num_heads = 8, res = True, ln = True)
    # [b,t,d]
    input_ = torch.randn(4,156,128)
    print(test(input_).shape)