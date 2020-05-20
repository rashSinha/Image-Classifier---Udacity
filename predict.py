from  import_resources import *
from model_predict import predict



# Create Argument and set it
parser = argparse.ArgumentParser(description='Image Classifier Program')

parser.add_argument('image_path', action="store")
parser.add_argument('saved_model', action="store")
parser.add_argument('--top_k', action="store", dest="top_k", type=int)
parser.add_argument('--category_names', action="store", dest="category_names")

print( parser.parse_args())
result = parser.parse_args()

image_path = result.image_path
saved_model = result.saved_model
top_k = result.top_k
category_names = result.category_names

if top_k == None:
    top_k = 5

# load model
reloaded_keras_model = tf.keras.models.load_model(saved_model,
                                                  custom_objects={'KerasLayer':hub.KerasLayer})

# predict image
image = np.asarray(Image.open(image_path))
probs, classes = predict(image_path, reloaded_keras_model, top_k)

if category_names !=None:
    with open(category_names, 'r') as f:
        class_names = json.load(f)
    names = [str(x+1) for x in classes]
    classes = [class_names.get(name) for name in names]

# print results
print('\n********************************************************************************************')
print('\nthe {} top classes:'.format(top_k))
for i in range(top_k):
    print('\n\u2022 Class: {}'.format(classes[i]), '\n\u2022 Probability: {:.3%}'.format(probs[i]))