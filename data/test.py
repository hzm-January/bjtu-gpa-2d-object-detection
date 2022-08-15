import numpy
import numpy as np
import torch

# image = torch.randn(2,3,4).numpy()
# print(image)
# image1 = image[:, ::-1]
# print('-------------------\n')
# print(image1)
# image2 = image[:, ::-1, ::-1]
# print('-------------------\n')
# print(image2)


# rect = torch.randn(4, 1).numpy()
# centers = torch.randn(4, 2).numpy()
# print(rect)
# print(centers)
# print(rect[0])
# print(centers[:,0])
# print(rect[0] < centers[:, 0])
#
# print(rect[1])
# print(centers[:,1])
# print(rect[1] < centers[:, 1])
# m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
#
# print(m1)
# print(rect[m1])
# print(m1.any())


# arr1 = [1, 2, 3, 4]
# arr2 = numpy.array([2, 3, 4, 5])
# print(arr1)
# print(arr2)
# print(type(arr1))
# print(type(arr2))
# tensor1 = torch.from_numpy(arr1)
# tensor2 = torch.from_numpy(arr2)
# print(tensor1)
# print(tensor2)


rect = [1, 2, 3, 4]
GT = [2, 1, 4, 3]
print(np.maximum(rect[:2], GT[:2]))
print(np.maximum(rect[2:], GT[2:]))
