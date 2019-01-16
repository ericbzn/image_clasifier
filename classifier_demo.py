from image_classifier import ImageClassifier

database_path = 'superheroes_database/'
query_path = 'query_images/'

color_space = 'lab'
hist_size = 16
parallel = 'True'
metric = 'emd'

classifier = ImageClassifier(color_space, hist_size, parallel, database_path, query_path, metric)

classifier.compare_image('troop')

