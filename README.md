# Quantitative Analysis of Similarity Measures of Distributions: 
## Image classifier demo

Abstract: There are many measures of dissimilarity that, depending on the application, do not always have optimal behavior. In this paper, we present a qualitative analysis of the similarity measures most used in the literature and the Earth Mover's Distance (EMD).  The EMD is a metric based on the theory of optimal transport with interesting geometrical properties for the comparison of distributions. However, the use of this measure is limited in comparison with other similarity measures. The main reason was, until recently, the computational complexity. We show the superiority of the EMD through three different experiments. First, analyzing the response of the measures in the simplest of cases; one-dimension synthetic distributions. Second, with two image retrieval systems; using colour and texture features. Finally, using a dimensional reduction technique for a visual representation of the textures. We show that today the EMD is a measure that better reflects the similarity between two distributions.

<!---You can find the full paper at:-->

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
The image classifier demo contains two folders, one with the image classifier based on the color information and another with the classifier based on texture information. Every retrieval system has it own image dataset inside. The datasets are separated in query images (_query_images/_) and model images folder (_superheroes_database/_ in the color-based case and _texture_database/_ in the texture-base case), which contains the images with which we are going to compare a query image.

The superhero toy database has 26 different classes<sup>[1](#myfootnote1)</sup>. The name of the classes corresponds to the name of the superheroes; if a name is repeated, the class is the superhero name plus some characteristic. The second database [[2](http://www.cb.uu.se/~gustaf/texture/)] is composed of images belonging to different surfaces and materials. The database contains 28 different classes; it contains different patches per class. The classes are:

Color labels | Texture label | 
--- | --- |
batman | blanket1 |
batman_black  | blanket2 |
batman_white  | canvas |
batman_wings  | ceiling1 |
boba  | ceiling2 |
c3po  | cushion |
capt_america  | floor1 |
flash  | floor2 |
hawkeye  | grass |
hulk  | lentils |
ironman_a  | linseeds |
ironman_b  | oatmeal |
loki  | pearlsugar |
magneto  | rice1 |
robin  | rice2 |
spiderman  | rug|
spiderman_black | sand |
superman | scarf1 |
superman_dark | scarf2 |
thor | screen |
troop | seat1 |
troop_black | seat2 |
vader | sesameseeds |
vader_silver | stone1 |
widow | stone2|
wolverine | stone3 |
wonderwoman | stoneslab |
 - | wall |

<a name="myfootnote1">1</a>: CC superheros images courtesy of Christopher Chong on Flickr [[1](https://www.flickr.com/photos/126293860@N05/)]

To run the demo, you have to launch the demo_superheroes_classifier.py (for the color case) or the demo_texture_classifier.py (for the texture case) file and it will guide you through the test. The classifier will ask you for some information:

**Color-based demo**
1. Query image: You have to choose an image between the classes listed above.
2. Color space: You can choose between 3 color spaces [lab, rgb, hls].
3. Histogram size: You have to choose a histogram size between 2 and 32 bins per color channel.
4. Metric: You have to choose between the six different similarity measures compared in this work. The possible metrics are:
* corr = Histogram correlation
* inter = Histogram Intersection
* bhatt = Bhattacharyya Distance
* chi2 = Chi Square Statistic
* kl = Kullback-Leibler Divergence
* emd = Earth Mover's Distance
* all = Test all the metrics

**Texture-based demo**
1. Query image: You have to choose an image between the classes listed above.
2. Number of frequencies: You have to choose between 4 and 10 bins.
3. Number of rotations: You have to choose between 4 and 12 bins .
4. Metric: You have to choose between the six different similarity measures compared in this work. The possible metrics are:
* corr = Histogram correlation
* inter = Histogram Intersection
* bhatt = Bhattacharyya Distance
* chi2 = Chi Square Statistic
* kl = Kullback-Leibler Divergence
* emd = Earth Mover's Distance
* all = Test all the metrics

Once the information is given, the color/texture_image_classifier.py will create the classifier and will compute the distance between the query image and the images in the database. At the end of the comparison, the classifier shows the more similar image (best match) according to with the set up (feature space, histogram size, and metric) and the rest of the database images ordered in increasing dissimilarity.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Authors
<!---
Secret until the double-blinded review pass
-->
## License

This project is licensed under the GNU General Public License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments



