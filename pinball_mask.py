import h5py
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

def main():

    flag = np.zeros((72,24)).T
    
    for idx in range(1, 4738):
        d = h5py.File(f"/storage/aguemes/gan-piv/{case}/ss02/ptv_noise010/SS2_{idx:06d}.mat", 'r')
        flag += np.array(d['Flagptv']).T
        # break
    flag = np.where(flag > 0, 1, flag)
    plt.imshow(flag, vmin=0,vmax=1, cmap='Greys', extent=[0,6,-1,1])
    plt.savefig('pinballMask.png')

    return


if __name__ == '__main__':

    case = 'pinball'
    noise = 10

    main()