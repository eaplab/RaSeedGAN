import ftplib


def main():

    FTP_host = "ftp2.psl.noaa.gov"
    FTP = ftplib.FTP()
    FTP.connect(FTP_host)
    FTP.login()
    FTP.cwd("Datasets/noaa.oisst.v2.highres/")
    FTP.retrlines("LIST")
    
    for year in years:
        
        filename = f"sst.day.mean.{year}.nc"

        with open(f"{root_path}{filename}", 'wb') as contents:
            FTP.retrbinary('RETR %s' % filename, contents.write)
    
    filename = "lsmask.oisst.v2.nc"

    with open(f"{root_path}{filename}", 'wb') as contents:
            FTP.retrbinary('RETR %s' % filename, contents.write)
    
    FTP.quit()

    return


if __name__ == '__main__':

    years = range(2000,2020)
    root_path = "/STORAGE01/aguemes/gan-piv/sst/raw/"

    main()