import h5py
import numpy as np
import scipy.io as sio

def main():

    # grid1 = sio.loadmat(f"/storage/aguemes/gan-piv/{case}/ss01/piv_noise{noise:03d}/SS1_grid.mat")
    # grid2 = sio.loadmat(f"/storage/aguemes/gan-piv/{case}/ss02/ptv_noise{noise:03d}/SS2_grid.mat")
    # X1 = grid1['X'][0,:].astype(int)
    # Y1 = grid1['Y'][:,0].astype(int)
    # X2 = grid2['X'][0,:].astype(int)
    # Y2 = grid2['Y'][:,0].astype(int)
    # X3 = ((X2[::2] + X2[1::2]) / 2).astype(int)
    # Y3 = ((Y2[::2] + Y2[1::2]) / 2).astype(int)

    # errorU = np.zeros((737,36,12))
    # errorV = np.zeros((737,36,12))
    # dnsU = np.zeros((737,72,24))
    # dnsV = np.zeros((737,72,24))
    
    # for idx in range(4001, 4738):
    #     d1 = h5py.File(f"/storage/aguemes/gan-piv/{case}/ss01/piv_noise{noise:03d}/SS1_{idx:06d}.mat", "r")
    #     d4 = sio.loadmat(f"/storage/aguemes/gan-piv/{case}/ss02/dns/SS2_{idx:06d}.mat")
    #     U1 = np.array(d1['Uptv'])
    #     V1 = np.array(d1['Vptv'])
    #     U4 = np.array(d4['UDNS']).T
    #     V4 = np.array(d4['VDNS']).T
    #     U3 = (U4[::2,::2] + U4[1::2,1::2]) / 2
    #     V3 = (V4[::2,::2] + V4[1::2,1::2]) / 2
    #     errorU[idx-4001,:,:] = (U3 - U1)**2
    #     errorV[idx-4001,:,:] = (V3 - V1)**2
    #     dnsU[idx-4001,:,:] = U4
    #     dnsV[idx-4001,:,:] = V4

    # errorU = np.sqrt(np.nanmean(errorU) / np.nanvar(dnsU))
    # errorV = np.sqrt(np.nanmean(errorV) / np.nanvar(dnsV))
    # print(errorU)
    # print(errorV)

    # case = 'channel'
    # # grid1 = sio.loadmat(f"/storage/aguemes/gan-piv/{case}/ss01/piv_noise{noise:03d}/SS1_grid.mat")
    # # grid2 = sio.loadmat(f"/storage/aguemes/gan-piv/{case}/ss02/ptv_noise{noise:03d}/SS2_grid.mat")
    # # X1 = grid1['X'][0,:].astype(int)
    # # Y1 = grid1['Y'][:,0].astype(int)
    # # X2 = grid2['X'][0,:].astype(int)
    # # Y2 = grid2['Y'][:,0].astype(int)
    # # X3 = ((X2[::2] + X2[1::2]) / 2).astype(int)
    # # Y3 = ((Y2[::2] + Y2[1::2]) / 2).astype(int)
    # # print(X1)
    # # print(X3)
    # errorU = np.zeros((1856,64,32))
    # errorV = np.zeros((1856,64,32))
    # dnsU = np.zeros((1856,128,64))
    # dnsV = np.zeros((1856,128,64))
    
    # for idx in range(10001, 11857):
    #     d1 = h5py.File(f"/storage/aguemes/gan-piv/{case}/ss01/piv_noise{noise:03d}/SS1_{idx:06d}.mat", "r")
    #     d4 = h5py.File(f"/storage/aguemes/gan-piv/{case}/ss02/dns/SS2_{idx:06d}.mat", "r")
    #     U1 = np.array(d1['Uptv'])
    #     V1 = np.array(d1['Vptv'])
    #     U4 = np.array(d4['UDNS'])
    #     V4 = np.array(d4['VDNS'])
    #     U3 = (U4[::2,::2] + U4[1::2,1::2]) / 2
    #     V3 = (V4[::2,::2] + V4[1::2,1::2]) / 2
    #     errorU[idx-10001,:,:] = (U3 - U1)**2
    #     errorV[idx-10001,:,:] = (V3 - V1)**2
    #     dnsU[idx-10001,:,:] = U4
    #     dnsV[idx-10001,:,:] = V4

    # errorU = np.sqrt(np.nanmean(errorU) / np.nanvar(dnsU))
    # errorV = np.sqrt(np.nanmean(errorV) / np.nanvar(dnsV))
    # print(errorU)
    # print(errorV)

    # case = 'sst'
    # noise = 0
    # # grid1 = sio.loadmat(f"/storage/aguemes/gan-piv/{case}/ss01/piv_noise{noise:03d}/SS1_grid.mat")
    # # grid2 = sio.loadmat(f"/storage/aguemes/gan-piv/{case}/ss02/ptv_noise{noise:03d}/SS2_grid.mat")
    # # X1 = grid1['X'][0,:].astype(int)
    # # Y1 = grid1['Y'][:,0].astype(int)
    # # X2 = grid2['X'][0,:].astype(int)
    # # Y2 = grid2['Y'][:,0].astype(int)
    # # X3 = ((X2[::2] + X2[1::2]) / 2).astype(int)
    # # Y3 = ((Y2[::2] + Y2[1::2]) / 2).astype(int)
    # # print(X1)
    # # print(X3)
    # errorT = np.zeros((1305,90,45))
    # dnsT = np.zeros((1305,180,90))
    
    # for idx in range(6001, 7306):
    #     d1 = h5py.File(f"/storage/aguemes/gan-piv/{case}/ss01/piv_noise{noise:03d}/SS1_{idx:06d}.mat", "r")
    #     d4 = sio.loadmat(f"/storage/aguemes/gan-piv/{case}/ss02/dns/SS2_{idx:06d}.mat")
        
    #     T1 = np.array(d1['T'])
    #     T4 = np.array(d4['TDNS']).T
    #     T3 = (T4[::2,::2] + T4[1::2,1::2]) / 2
    #     errorT[idx-6001,:,:] = (T3 - T1)**2
    #     dnsT[idx-6001,:,:] = T4

    # errorT = np.sqrt(np.nanmean(errorT) / np.nanvar(dnsT))
    # print(errorT)

    case = 'exptbl'
    noise = 0
    grid1 = h5py.File(f"/storage/aguemes/gan-piv/{case}/ss01/piv_noise{noise:03d}/SS1_000001.mat", "r")
    grid2 = h5py.File(f"/storage/aguemes/gan-piv/{case}/ss02/dns/SS2_000001.mat", "r")
    X1 = grid1['XPIV'][:,0].astype(int)
    Y1 = grid1['YPIV'][0,:].astype(int)
    X2 = grid2['XPIV'][:,0].astype(int)
    Y2 = grid2['YPIV'][0,:].astype(int)
    X3 = ((X2[1:-1:4] + X2[2::4]) / 2).astype(int)
    Y3 = ((Y2[1:-1:4] + Y2[2::4]) / 2).astype(int)
    print(X1)
    print(X3)
    print(Y1)
    print(Y3)
    errorV = np.zeros((4000,32,32))
    errorU = np.zeros((4000,32,32))
    dnsU = np.zeros((4000,128,128))
    dnsV = np.zeros((4000,128,128))
    
    for idx in range(30001, 34001):
        d1 = h5py.File(f"/storage/aguemes/gan-piv/{case}/ss01/piv_noise{noise:03d}/SS1_{idx:06d}.mat", "r")
        d4 = h5py.File(f"/storage/aguemes/gan-piv/{case}/ss02/dns/SS2_{idx:06d}.mat", "r")
        U1 = np.array(d1['U'])
        V1 = np.array(d1['V'])
        U4 = np.array(d4['U'])
        V4 = np.array(d4['V'])
        U3 = (U4[1:-1:4,1:-1:4] + U4[2::4,2::4]) / 2
        V3 = (V4[1:-1:4,1:-1:4] + V4[2::4,2::4]) / 2
        errorU[idx-30001,:,:] = (U3 - U1)**2
        errorV[idx-30001,:,:] = (V3 - V1)**2
        dnsU[idx-30001,:,:] = U4
        dnsV[idx-30001,:,:] = V4

    errorU = np.sqrt(np.nanmean(errorU) / np.nanvar(dnsU))
    errorV = np.sqrt(np.nanmean(errorV) / np.nanvar(dnsV))
    print(errorU)
    print(errorV)

    return


if __name__ == '__main__':

    case = 'pinball'
    noise = 10

    main()