# **RAndomly-SEEDed super-resolution GAN (RaSeedGAN)**
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
<a href="https://github.com/Andy53/ERC.Xdbg/commits/master">
    <img src="https://img.shields.io/github/last-commit/eaplab/RaSeedGAN?style=flat-square&logo=github&logoColor=white">
</a>

This repository covers the Python implementation of a generative adversarial network (GAN) for estimating high-resolution field quantities from random sparse sensors without needing any full-resolution field for training.

The proposed network has been testes with four different cases. The associated raw data will be completed in the coming weeks. The datasets are:

*   Fluidic Pinball: direct-numerical simulation (DNS) data generated from *Deng et al. (2020)*. **(Available)**
*   Turbulent Channel Flow: DNS data from a turbulent channel flow with friction Reynolds number <img src="https://render.githubusercontent.com/render/math?math=Re_{\tau}=1000"> available at [Johns Hopkins Turbulence Database](http://turbulence.pha.jhu.edu). **(To appear soon)**
*   Turbulent Boundary Layer Flow: experimental data of a turbulent boundary layer with friction Reynolds number <img src="https://render.githubusercontent.com/render/math?math=Re_{\tau}\approx 1000"> acquired in the water-tunnel facility at Universidad Carlos III de Madrid. **(To appear soon)**
*   Sea Surface Temperature: experimental data of the global sea surface temperature from January 2000 to December 2019, downloaded from [NOAA](http://www.esrl.noaa.gov/psd/). **(To appear soon)**

## **Installation**

Use the package manager [pip3](https://pip.pypa.io/en/stable/) to install the required dependencies. Python 3.8 is required.

```bash
pip install -r requirements.txt
```


## **Usage**

To generate the tfrecord files, execute:

```bash
python run_generate_tfrecords.py -c pinball -u 4 -n 010
```

To run the training procedure, execute:

```bash
python run_training.py --case pinball --upsampling 4 --model_name architecture01-noise-010 --noise 10 --learning_rate 1e-4
```

To compute the prediction of the testing dataset, execute:

```bash
python run_compute_predictions.py -c pinball -u 4 -m architecture01-noise-010 -n 10 -l 1e-4
```

On a system with one GPU available, fluidic pinball case with upsampling factor <img src="https://render.githubusercontent.com/render/math?math=f_u=4"> takes approximately 100 seconds to run a sinfle epoch. 

## **Publication**
This repository has been used for the following scientific publication:

- Güemes, A., Sanmiguel Vila, C., & Discetti, S. (2022). Super-resolution GANs of randomly-seeded fields. *arXiv preprint arXiv:2202.11701*.

## **Authorship**
This repository has been developed in the Experimental Aerodynamics and Propulsion group at Universidad Carloss III de Madrid. The following researchers are acknowledged for their contributions:
- Alejandro Güemes
- Stefano Discetti
- Carlos Sanmiguel Vila

## **Contributing**
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## **Funding**
This project has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (grant agreement No 949085).

## **License**
Creative Commons Attribution 4.0 International. See LICENSE.md for further details.

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
