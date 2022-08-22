import h5py
import numpy as np

a = np.zeros((4000,32,32))
b = np.zeros((4000,32,32))

for idx in range(30001,34000):

    piv = h5py.File(f"/storage/aguemes/gan-piv/exptbl/ss01/piv_noise000/SS1_0{idx}.mat", 'r')
    a[idx-30001] = np.array(piv['U']).T[:,:]
    piv = h5py.File(f"/storage/aguemes/gan-piv/exptbl/ss01/piv_noise000LowerDensity/SS1_0{idx}.mat", 'r')
    b[idx-30001] = np.array(piv['Uptv']).T[:,:]

print(np.mean(a))
print(np.mean(b))
print(np.std(a))
print(np.std(b))