from import_resources import *
from process_image import process_image

#Create the predict function
def predict(image_path, model, top_k):
    im = Image.open(image_path)
    image = np.asarray(im)
    image_modify = process_image(image)
    image_modify = np.expand_dims(image_modify, axis=0)

    predicted = model.predict(image_modify)
    probs = - np.partition(-predicted[0], top_k)[:top_k]
    classes = np.argpartition(-predicted[0], top_k)[:top_k]
    return probs, list(classes)