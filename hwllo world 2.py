import numpy
import torch
import torch.nn as nn

# Assuming A and B are already defined
B= torch.randn([32,  16,256])


# Reshape A and B to make them compatible for matmul
#A_reshaped = torch.permute(A,(0,2,1))  # Transpose to [32, 16, 1024]
#B_reshaped = torch.permute(B,(0,2,1))  # Transpose to [32, 256, 16]

# Perform matmul
#result = torch.matmul(A, B)
#print(result)
# Final result shape will be [32, 1024, 256]
#print(result.shape)

#B= torch.load('./Latent.pt')
#print(B)
linear = nn.Linear(256,40)#.to('cuda')

result = linear(B)

#print( result)

result = torch.permute(result,(0,2,1))
result = torch.sum(result,dim=-1)
print( result.shape)

import torch

# Assuming label_vector is your 32x1 label vector
label_vector = torch.randint(0, 5, (5, 1))  # Example label vector, replace with your actual label vector

# Convert label vector to one-hot encoded matrix
one_hot_matrix = torch.eye(40)[label_vector.squeeze(1)]

print(label_vector)
print(one_hot_matrix)