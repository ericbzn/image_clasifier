from texture_image_classifier import *

database_path = 'texture_database/'
query_path = 'query_images/'
os.system('clear')
print '\nTEXTURE-BASED IMAGE CLASSIFIER\n'
print 'Click on any part of the image to continue'
query_imgs = list([s.split('.')[0] for s in sorted(os.listdir(query_path))])
database_imgs = list([s.split('.')[0] for s in sorted(os.listdir(database_path))])

fig, ax = plt.subplots(4, 7, figsize=(26, 26))
fig.suptitle('QUERY IMAGES (Ordered Alphabetically)')

order_array = np.arange(0, 28, 1).reshape((4, 7))
for ii in range(4):
    for jj in range(7):
        temp_img = read_img(query_path + query_imgs[order_array[ii, jj]] + '.png')
        ax[ii, jj].imshow(temp_img, cmap='gray')
        ax[ii, jj].set_xticks([])
        ax[ii, jj].set_yticks([])
        ax[ii, jj].set_xlabel(query_imgs[order_array[ii, jj]])

plt.draw()
plt.waitforbuttonpress()

while True:
    print 'Please choose a query image from the list below: \n'
    print '\n'.join(query_imgs)
    q_img = raw_input('\nQuery image: ')

    if q_img not in query_imgs:
        os.system('clear')
        print 'ERROR: The image ' + q_img + ' is not in the list. \n'
    else:
        break

# descriptor = ['gabor', 'coomatrix']
# while True:
#     texture_descriptor = raw_input('\nPlease choose a texture descriptor [gabor, coomatrix]: ')
#     if texture_descriptor not in descriptor:
#         os.system('clear')
#         print 'ERROR: The texture descriptor ' + texture_descriptor + ' is not supported.'
#     else:
#         break
texture_descriptor = 'gabor'
while True:
    n_freq = int(raw_input('\nPlease choose the number of frequencies [between 2 and 32 bins]: '))
    if n_freq < 2 or n_freq > 32:
        os.system('clear')
        print 'ERROR: The histogram size is too small or too big.'
    else:
        break

while True:
    n_angles = int(raw_input('\nPlease choose the number of rotations [between 2 and 32 bins]: '))
    if n_angles < 2 or n_angles > 32:
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
classifier = ImageClassifier(texture_descriptor, n_freq, n_angles, database_path, query_path)
classifier.get_database_histogram(database_path)
if metric == 'all':
    for m in methods[:-1]:
        classifier.compare_image(q_img, m)
else:
    classifier.compare_image(q_img, metric)

plt.show()