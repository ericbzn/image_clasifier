# Landing Target Detection by Perception

In this work, we present a novel method for the detection of landing targets for the UAV vision aided landing. We propose to model the landing target by taking into account the principles of the human perception. Our model extracts the landing target contours as outliers using the RX anomaly detector and computing proximity and a similarity measure. Finally, we use the error correction Hamming code to reduce the recognition errors. The methodology presented works in an unsupervised mode, i.e., no need to adjust parameters. 

You can find the full paper at: https://link.springer.com/chapter/10.1007/978-3-030-01449-0_20
and look some results of our algorithm at: https://youtu.be/igsQc7VEF2c


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

The presented code was developed in Ubuntu 16.04 using python 2.7 and other external libraries. Therefore, you need to have in your system the following modules:

* python 2.7
* python libraries (numpy, SciPy, Sklearn, Pygame, ...)
* OpenCv

### Installing pip for Python 2

1. Update the package index
```console
sudo apt update
```
2. Install pip for Python
```console
sudo apt install python-pip
```
3. Install needed packages with pip
```console
sudo -H pip install numpy scipy matplotlib pygame opencv-python scikit-learn spectral Pillow 
```

## Running the tests
The code only has a source file (detection_by_perception.py) and is composed of 20 functions. In the main function, the code brings the option to choose between three options depending on the source from the input images come from. 

1. `src_images()` - Take the images for the landing target detection from a repertoire of images 
2. `src_video()` - Take the images for the landing target detection from a live video coming from a camera 
3. `src_video_file()` - Take the images for the landing target detection from a video file

To run a test, try first with the `src_images()`. The images for the test are allocated in *indir\images* and the after the process, the result images sill be save in *outdir\images*. If some of the directories don't exist, the program will send an error.

Also, you may take into account that even if the code could process the landing target detection using the three sources of input images, it may causes errors. When testing, comment/uncomment the respective functions. 

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

* **Eric Bazan** *<eric.bazan@mines-paristech.fr>* - *Initial work*
* **Petr Dokládal** *<petr.dokladaln@mines-paristech.fr>*- *Initial work* 
* **Eva Dokládalová** *<eva.dokladalova@esiee.fr>*- *Initial work* 

## License

This project is licensed under the GNU General Public License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* This research is partially supported by the Mexican National Council for Science and Technology (CONACYT)

