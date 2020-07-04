import numpy as np
from keras.models import load_model
from keras.preprocessing import image

class carbrand:
    def __init__(self,filename):
        self.filename =filename


    def prediction(self):

        model = load_model('carbrandclassifier.h5')
        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)

        if result[0][0] == 1:
            return[{"Prediction" :'Audi'}]
        elif result[0][1]==1:
            return [{"Prediction" :'Lamborghini'}]
        else:
            return [{"Prediction" :'Mercedes-Benz'}]