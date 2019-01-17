import os
from image_classifier import ImageClassifier
import matplotlib.pyplot as plt

database_path = 'superheroes_database/'
query_path = 'query_images/'
os.system('clear')
print '\nCOLOR-BASED IMAGE CLASSIFIER\n'
query_imgs = list([s.replace('.png', '') for s in sorted(os.listdir(query_path))])


while True:
    print 'Please choose a query image from the list below: \n'
    print '\n'.join(query_imgs)
    q_img = raw_input('\nQuery image: ')

    if q_img not in query_imgs:
        os.system('clear')
        print 'ERROR: The image ' + q_img + ' is not in the list. \n'
    else:
        break

spaces = ['rgb', 'lab', 'hls']
while True:
    color_space = raw_input('\nPlease choose a color space [lab, rgb, hls]: ')
    if color_space not in spaces:
        os.system('clear')
        print 'ERROR: The color space ' + color_space + ' is not supported.'
    else:
        break

while True:
    hist_size = round(float(raw_input('\nPlease choose the histogram size [between 2 and 32 bins per color channel]: ')))
    if hist_size < 2 or hist_size > 32:
        os.system('clear')
        print 'ERROR: The histogram size is too small or too big.'
    else:
        break

metric_description = ['corr = Histogram correlation', 'inter = Histogram Intersection', 'bhatt = Bhattacharyya Distance',
           'chi2 = Chi Square Statistic', 'kl = Kullback-Leibler Divergence', 'emd = Earth Movers Distance', 'all = Test all the metrics' ]

methods = ['corr', 'inter', 'bhatt', 'chi2', 'kl', 'emd', 'all']
while True:
    print '\nPlease choose a metric to do the histogram comparison:\n'
    print '\n'.join(metric_description)
    metric = raw_input('\nMetric: ')
    if metric not in methods:
        os.system('clear')
        print 'ERROR: The metric ' + metric + ' is not supported.'
    else:
        break
        
print '... Initializing classifier'
classifier = ImageClassifier(color_space, hist_size, database_path, query_path)

if metric == 'all':
    for m in methods[:-1]:
        classifier.compare_image(q_img, m)
else:
    classifier.compare_image(q_img, metric)
    
plt.show()
