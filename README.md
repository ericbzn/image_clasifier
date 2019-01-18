# Quantitative Analysis of Similarity Measures of Distributions: 
## Image classifier demo

Abstract: The Earth Mover’s Distance (EMD) is a metric based on the theory of optimal transport that has interesting geometrical properties for distributions comparison. However, the use of this measure is limited in comparison with other similarity measures as the Kullback Leibler divergence. The main reason was, until recently, the computation complexity. In this paper, we present a comparative study of the dissimilarity measures most used in the literature for the comparison of distributions through a color-based image classification system and other simple examples with synthetic data. We show that today the EMD is a computationally efficient measure that better reflects the similarity between two distributions.

You can find the full paper at: 

## Getting Started
These instructions will get you a copy of the image classifier used in the article to compare the different similarity measures. The code runs on your local machine for testing and development purposes.

### Prerequisites
The presented code was developed in Ubuntu 18.04 using python 2.7 and other external libraries. Therefore, you need to have in your system the following modules:

* python 2.7
* python common libraries (numpy, SciPy, Sklearn, Pygame, matplotlib,...)
* OpenCv
* Python Optimal Trasport (POT)
* Python joblib.Parallel (joblib)

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
sudo -H pip install numpy scipy matplotlib pygame opencv-python scikit-learn spectral Pillow joblib POT
```
## Running the tests
The image classifier demo contains two folders, one with the query images (query_images/) and other which contains the images with which we are going to compare a query image. The database has 26 different classes (superheroes_database/). The name of the classes corresponds to the name of the superheroes; if a name is repeated, the class is the superhero name plus some characteristic. The classes are:
* batman 
* batman_black
* batman_white
* batman_wings
* boba
* c3po
* capt_america
* flash
* hawkeye
* hulk
* ironman_a
* ironman_b
* loki
* magneto
* robin
* spiderman
* spiderman_black
* superman
* superman_dark
* thor
* troop
* troop_black
* vader
* vader_silver
* widow
* wolverine
* wonderwoman
 
To run the demo, you have to launch the demo_superheroe_classifier.py file and it will guide you through the test. The classifier will ask you for some information:
 
1. Query image: You have to choose an image between the classes listed below.
2. Color space: You can choose between 3 color spaces [lab, rgb, hls].
3. Histogram size: You have to choose a histogram size between 2 and 32 bins per color channel.
4. Metric: You have to choose between the six different similarity measures compared in this work. The possible metrics are:
* corr = Histogram correlation
* inter = Histogram Intersection
* bhatt = Bhattacharyya Distance
* chi2 = Chi Square Statistic
* kl = Kullback-Leibler Divergence
* emd = Earth Movers Distance
* all = Test all the metrics

Once the information is given, the image_classifier.py will create the classifier and will compute the distance between the query image and the images in the database. At the end of the comparison, the classifier shows the more similar image (best match) according to with the set up (color space, histogram size, and metric) and the rest of the database images ordered in increasing dissimilarity.

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

