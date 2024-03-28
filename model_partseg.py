from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]  #
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


class DGCNN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))



        self.linear = nn.Linear(256, 40, bias=False)
        self.ln8 =  nn.LayerNorm([1024,40], eps=1e-6)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k) # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        # print(f'x shape = {x.shape}')
        x = self.conv1(x) # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]
        #print(f'x1 = {x1.shape} ') #batch 1024 64


        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
        #print(f'x2 = {x2.shape} ') #batch 1024 64

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        #print(f'x3 = {x3.shape} ') #batch 1024 256

        x = torch.cat((x1, x2, x3), dim=1)
        x = torch.permute(x, (0, 2, 1))


        x = F.relu(self.ln8( self.linear(x))) #batch 1024 40

        return x
class LatentCapsLayer(nn.Module):
    def __init__(self, latent_caps_size=40, prim_caps_size=1024, prim_vec_size=40, latent_vec_size=64):
        super(LatentCapsLayer, self).__init__()
        self.prim_vec_size = prim_vec_size
        self.prim_caps_size = prim_caps_size
        self.latent_caps_size = latent_caps_size
        self.W = nn.Parameter(0.01*torch.randn(latent_caps_size, prim_caps_size, latent_vec_size, prim_vec_size))
    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / \
            ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor
    def forward(self, x):
        u_hat = torch.squeeze(torch.matmul(self.W, x[:, None, :, :, None]), dim=-1)
        u_hat_detached = u_hat.detach()
        b_ij = Variable(torch.zeros(x.size(0), self.latent_caps_size, self.prim_caps_size)).cuda()
        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij, 1)
            #print(f'c_ij ={c_ij}')
            if iteration == num_iterations - 1:
                v_j = self.squash(torch.sum(c_ij[:, :, :, None] * u_hat, dim=-2, keepdim=True))
            else:
                v_j = self.squash(torch.sum(c_ij[:, :, :, None] * u_hat_detached, dim=-2, keepdim=True))
                b_ij = b_ij + torch.sum(v_j * u_hat_detached, dim=-1)
        return v_j.squeeze(-2)


class DGCNNCaps_part(nn.Module):
    def __init__(self,args, output_channel =40, seg_part= 64,seg_num_all=50):
        super(DGCNNCaps_part,self).__init__()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(seg_num_all)

        self.DGCNN = DGCNN(args=args,output_channels=output_channel)
        self.Latentcaps = LatentCapsLayer(latent_vec_size=seg_part)

        self.conv1 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv1d(1128, 256, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(128, seg_num_all, kernel_size=1, bias=False),
                                    self.bn4,
                                    nn.LeakyReLU(negative_slope=0.2))

    def forward(self,data,l):
        batch_size = data.size(0)
        num_points = data.size(2)
        #data = data.permute(0, 2, 1)  # batch  3 1024
        num_point = data.size(-1)
        encoder = self.DGCNN(data) #batch 1024 40
        #print(f'encoder ={encoder.shape}')
        Latent  = self.Latentcaps(encoder)
        #print(f'latent = {Latent.shape}') #batch ,40,50
        embsize= torch.bmm(encoder,Latent) #batch , 1024,50
        embsize = embsize.max(dim=-1, keepdim=True)[0] #batch , 1024,1
        print(f' embsize ={embsize.shape}')
        print(f'label size ={l.shape}')
        l = l.view(batch_size, -1, 1)  # (batch_size, num_categoties, 1)
        l = self.conv1(l)  # (batch_size, num_categoties, 1) -> (batch_size, 64, 1)
        print(f' embsize ={embsize.shape}')
        print(f'label size ={l.shape}')
        embsize =  torch.cat((embsize,l), dim=1)
        embsize = embsize.repeat(1, 1, num_points)  # (batch_size, 1088, num_points)
        print(f' embsize after cat ={embsize.shape}')
        embsize = torch.cat((embsize,torch.permute(encoder,(0,2,1))),dim=1)
        print(f' embsize after cat ={embsize.shape}')

        result = self.conv2(embsize)
        result = self.conv3(result)
        result = self.conv4(result)
        result = self.conv5(result)
        print(f' result = {result.shape}')
        return result

