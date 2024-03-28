from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from collections import OrderedDict
import sys

class LatentCapsLayer (nn.Module):
    def __init__(self, prim_caps_size=1024, prim_vec_size = 16,latent_caps_size = 16,latent_vec_size = 40):
        super(LatentCapsLayer, self).__init__()
        self.prim_vec_size = prim_vec_size
        self.prim_caps_size = prim_caps_size
        self.latent_vec_size = latent_vec_size
        self.latent_caps_size = latent_caps_size
        self.W =  nn.Parameter(0.01*torch.randn([prim_caps_size,latent_caps_size* latent_vec_size]))

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = (squared_norm / ((1. + squared_norm)) *(input_tensor/ torch.sqrt(squared_norm)))
        return output_tensor

    def forward (self,x):
        #print(f'x .shape ={x.shape}')
        x = torch.permute(x,(0,2,1))
        #print(f'x .shape ={x.shape}')
        #print(f'W shape ={self.W .shape}')
        u_hat = torch.squeeze(torch.matmul( x, self.W))
        u_hat = u_hat.view(x.shape[0], self.latent_caps_size, self.latent_vec_size,self.prim_vec_size)
        u_hat_detached = u_hat.detach().cuda()
        b_ij = Variable(torch.zeros(x.size(0), self.latent_caps_size, self.latent_vec_size, self.prim_vec_size)).cuda()

        output = torch.zeros_like(b_ij)
        num_iterations = 3
        for iteration in range(num_iterations):
            #print(f'Round of ite ={iteration}')
            c_ij = F.softmax(b_ij, dim=-2).cuda() #32,16,256,16
            #print(f'cij shape ={c_ij.shape}')
            if iteration == num_iterations - 1:
                s_ij = c_ij * u_hat_detached
                #print(f's_ij = {s_ij.shape}')
                s_ij = torch.sum(s_ij, dim=-1)

                #print(f' summation = {s_ij.shape}')
                v_j = self.squash(s_ij)
                #print(f'after squash = {v_j.shape}')
                output = v_j
            else:
                s_ij = c_ij * u_hat_detached
                #print(f'matmul cxu = {s_ij.shape} ')
                s_ij = torch.sum(s_ij, dim=-1, keepdim=True)
                #print(f' summation = {s_ij.shape}')
                v_j = self.squash(s_ij)
                #print(f'after squash = {v_j.shape}')
                b_ij = b_ij + s_ij * u_hat_detached
                #print(f'softmax update ={b_ij.shape}\n')
        return output


  

