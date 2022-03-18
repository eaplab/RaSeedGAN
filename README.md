# **RAndomly-SEEDed super-resolution GAN (RaSeedGAN)**
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

This repository covers the Python implementation of a generative adversarial network (GAN) for estimating high-resolution field quantities from random sparse sensors without needing any full-resolution field for training.

Four different cases are available:

*   Fluidic Pinball: direct-numerical simulation (DNS) data generated from *Deng et al. (2020)*.
*   Turbulent Channel Flow: DNS data from a turbulent channel flow with friction Reynolds number <img src="https://render.githubusercontent.com/render/math?math=Re_{\tau}=1000"> available at [Johns Hopkins Turbulence Database](http://turbulence.pha.jhu.edu).
*   Turbulent Boundary Layer Flow: experimental data of a turbulent boundary layer with friction Reynolds number <img src="https://render.githubusercontent.com/render/math?math=Re_{\tau}\approx 1000"> acquired in the water-tunnel facility at Universidad Carlos III de Madrid.
*   Sea Surface Temperature: experimental data of the global sea surface temperature from January 2000 to December 2019, downloaded from [NOAA](http://www.esrl.noaa.gov/psd/).

## **Installation**

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required dependencies.

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
python run_training --case pinball --upsampling 4 --model_name architecture01-noise-010 --noise 10 --learning_rate 1e-4
```

To compute the prediction of the testing dataset, execute:

```bash
python run_predictions -c pinball -u 4 -m architecture01-noise-010 -n 10 -l 1e-4
```

## **Publications**
This repository has been used for the following scientific publications:

- Güemes, A., Sanmiguel Vila, C., & Discetti, S. (2022). Super-resolution GANs of randomly-seeded fields. *arXiv preprint arXiv:2202.11701*.
- Güemes, A., Sanmiguel Vila, C., & Discetti, S. (2021, August). GANs-based PIV resolution enhancement without the need of high-resolution input. In *14th International Symposium on Particle Image Velocimetry (Vol. 1, No. 1)*.

## **Authorship**
This repository has been developed in the Experimental Aerodynamics and Propulsion group at Universidad Carloss III de Madrid. The following researchers are acknowledged for their contributions:
- Alejandro Güemes
- Stefano Discetti
- Carlos Sanmiguel

## **Contributing**
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## **License**
See LICENSE.md for further details.
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
