
import torch
import torch.nn as nn
def array2samples_distance(array1, array2):
    """
    CD distance of single point cloud
    arguments:
        array1: the array, size: (num_point, num_feature). (512,3)
        array2: the samples, size: (num_point, num_feature). (512,3)
    returns:
        distances: each entry is the distance from a sample to array1
    """
    num_point1, num_features1 = array1.shape
    num_point2, num_features2 = array2.shape

    expanded_array1 = array1.repeat(num_point2, 1)  # (512*512,1)
    # (512,3) -> (512,1,3) -> (512,512,3) -> (512*512,3)
    expanded_array2 = torch.reshape(torch.unsqueeze(array2, 1).repeat(1, num_point1, 1), (-1, num_features2))
    # expanded_array1: (512,1), expanded_array2: (512*512,3)
    distances = (expanded_array1 - expanded_array2) * (expanded_array1 - expanded_array2)
    #    distances = torch.sqrt(distances)
    distances = torch.sum(distances, dim=1)  # s1中的点和s2中的点两两之间的平方距离
    distances = torch.reshape(distances, (num_point2, num_point1))  #

    distances = torch.min(distances, dim=1)[0]
    distances = torch.mean(distances)
    return distances


def chamfer_distance_numpy(array1, array2):
    """
    batch.
    Parameters
    ----------
    array1: (b,512,3)
    array2: (b,512,3)
    Returns
    -------
    """
    batch_size, num_point, num_features = array1.shape
    dist = 0
    for i in range(batch_size):  # 每次求一个点云的CD值，最后求平均值。
        av_dist1 = array2samples_distance(array1[i], array2[i])
        av_dist2 = array2samples_distance(array2[i], array1[i])
        dist = dist + (0.5 * av_dist1 + 0.5 * av_dist2) / batch_size  # 等于(a1+b1)+(a2+b2)+(a3+b3)+../32
    return dist * 100


class PointLoss(nn.Module):
    def __init__(self):
        super(PointLoss, self).__init__()

    def forward(self, array1, array2):
        """

        Parameters
        ----------
        array1: (b,512,3)
        array2: (b,512,3)
        Returns
        -------
        """
        return chamfer_distance_numpy(array1, array2)


criterion_PointLoss = PointLoss().to('cuda')

# fake: (b,1,512,3) -> (b,512,3), real_center: (b,1,512,3) -> (b,512,3)
fake =torch.randn(1,152,3)
real_center=torch.randn(1,152,3)
CD_LOSS = criterion_PointLoss(fake, real_center)

print(f'{CD_LOSS}')
