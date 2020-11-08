import numpy as np


a = np.load('dataset/X_train.npy')
b = np.load('dataset/X_test.npy')

al = np.load('dataset/y_train.npy')
bl = np.load('dataset/y_test.npy')

print(a.shape)
print(b.shape)
print(al.shape)
print(bl.shape)

a1 = a[:5957117, :]
al1 = al[:5957117, :]


print(a1.shape)
print(al1.shape)
print(b.shape)
print(bl.shape)

data = np.concatenate((a1, b), axis=0)
label = np.concatenate((al1, bl), axis=0)

np.save('X_train', data)
np.save('y_train', label)
print(data.shape)
print(label.shape)
print(np.shape(np.load('X_train.npy')))
print(np.shape(np.load('y_train.npy')))
print(np.min(data))
print(np.max(data))

print('Write finished')
