import numpy as np
import netCDF4 as nc
import scipy.io as sio
import matplotlib.pyplot as plt


def main():

    for year in years:

        print(year)

        filename = f"{root_path}raw/sst.day.mean.{year}.nc"
        
        ds = nc.Dataset(filename)

        if 'sst' not in locals():

            sst = ds['sst'][:]
            lat = ds['lat'][:]
            lon = ds['lon'][:]
            time = ds['time'][:]
    
        else:

            sst = np.ma.concatenate(
                (sst, ds['sst'][:]),
                axis=0
            )
    
    sst = sst.filled(np.nan)  
    lat = lat.filled(np.nan) 
    lon = lon.filled(np.nan) 

    for idx in range(sst.shape[0]):
    
        sio.savemat(
            f"{root_path}/matlab/sst_{list(years)[0]}-{list(years)[-1]}_sample-{idx+1:04d}-of-{sst.shape[0]:04d}.mat",
            {
                "sst": sst[idx, :, :]
            }
        )

    sio.savemat(
        f"{root_path}/matlab/sst_{list(years)[0]}-{list(years)[-1]}_grid.mat",
        {
            "lat": lat,
            "lon": lon
        }
    )

    return


if __name__ == '__main__':

    years = range(2000,2020)
    root_path = "/STORAGE01/aguemes/gan-piv/sst/"
    
    main()