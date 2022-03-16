# **GANs-based PIV resolution enhancement without the need of high-resolution input**

This notebook covers the Python implementation of a generative adversarial network (GAN) for enhancing the resolution of particle-image-velocimetry (PIV) images from incomplete high-resolution pairs. Particle tracking velocimetry is the experimental technique used to acquire those incomplete high-resolution images. The article associate to his work can be found at *[to be published]*.

Four different cases are available:

*   Cylinder wake: direct-numerical simulation (DNS) data generated from *Taira and Colonius (2007)* and *Kutz et al. (2016)*.
*   Turbulent-channel flow: DNS data from a turbulent channel flow with friction Reynolds number $Re_{\tau}=1000$ available at [Johns Hopkins Turbulence Database](http://turbulence.pha.jhu.edu).
*   Turbulent boundary layer flow: experimental data of a turbulent boundary layer with friction Reynolds number $Re_{\tau}\approx 900$ acquired in the water-tunnel facility at Universidad Carlos III de Madrid.
*   Blunt-body wake: experimental data of the flow around a blunt body acquired in the wind-tunnel facility at Universidad Carlos III de Madrid.

## **Installation**

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required dependencies.

```bash
pip install -r requirements.txt
```

## **Usage**

To generate the tfrecord files, execute:

```console
guest@vegeta:~$ python run_generate_tfrecords.py -c channel -u 4
```

To run the training procedure, execute:

```console
guest@vegeta:~$ python run_training --case channel --upsampling 4 --model_name architecture01 --learning_rate 1e-4
```

To compute the prediction of the testing dataset, execute:

```console
guest@vegeta:~$ python run_predictions -c channel -u 4 -m architecture01 -l 1e-4
```

## **Publications**
This repository has been used for the following scientific publications:

To be a nnounced

## **Authorship**
This repository has been developed in the Experimental Aerodynamics and Propulsion group at Universidad Carloss III de Madrid. The following researches and students are acknowledged for their contributions:
- Alejandro Güemes
- Stefano Discetti
- Carlos Sanmiguel

## **Contributing**
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## **License**
[Creative Commons](https://creativecommons.org)