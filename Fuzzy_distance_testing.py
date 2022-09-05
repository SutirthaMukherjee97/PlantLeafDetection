from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
y_test_1d=encode_y(y)
model_RGB_pred=np.load('')
model_HSV_pred=np.load('')
model_YUV_pred=np.load('')
classifier1=model_RGB_pred
classifier2=model_HSV_pred
classifier3=model_YUV_pred
import numpy as np
import scipy
#Use Verbose if you want to visualise the inner operations
def fuzzy_dist(classifier1, classifier2, classifier3, verbose=False):
    out = np.empty(len(classifier1))
    for i in range(len(classifier1)):
        if np.argmax(classifier1[i]) == np.argmax(classifier2[i]) == np.argmax(classifier3[i]):
            out[i] = np.argmax(classifier2[i])
        else:
            measure = np.zeros(len(classifier1[i]))
            for j in range(len(classifier1[i])):
                scores = np.array(
                    [classifier1[i, j], classifier2[i, j], classifier3[i, j]])
                measure[j] = scipy.spatial.distance.cosine(np.ones(3), scores)*scipy.spatial.distance.euclidean(
                    np.ones(3), scores)*scipy.spatial.distance.cityblock(np.ones(3), scores)
                if verbose:
                    print(measure)
            out[i] = np.argmin(measure)
    return out
  ensemble_pred=fuzzy_dist(classifier1,classifier2,classifier3)
  accuracy=accuracy_score(y_test_1d,ensemble_pred)
  precision_score=precision_score(y_test_1d,ensemble_pred , average= None )
  cm=confusion_matrix(y_test_1d,ensemble_pred)
