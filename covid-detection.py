# Assuming that Covid+ images will introduce/modify some edges/intensities/colors that are not found in Covid-images.
# 1. We convert image from Gray/RGB to YCbCr
# Y<--Luma : Give info about intensity
# CbCr <--Chroma : info about colors
# Most modification clues which can not be detected by the naked eye, are hidden in the chromatic channel since the human visual system is more sensitive to overall intansity Y changes than to colour CbCr changes.

# 2. Apply Maxi and Mini Filters to Y <--Illumination
# Maxi and Mini filters behave like an edge-preserving filter such that clear edges are kept between surfaces under different lighting conditions, and details within a single surface are blurred.(You can aslo try bilateral filter from cv2 and check the results)

# 3. Compute LBP for (illumination, Cb,Cr)
# Assuming that Covid+ introduced the image texture modifications. LBP will help to capture texture information

# 4. LBP histogram is used as feature vector on classifiers(SVM, LDA, KNN, LR, DTree, NBayes)

#%%
import numpy as np
import matplotlib.pyplot as plt
import os
#%%
import cv2
from tqdm import tqdm
from skimage.feature import local_binary_pattern
from PIL import Image, ImageFilter
from datetime import datetime
import skimage.measure
import random

#%%
DATADIR = "/home/muzuri/Documents/My_final_Project/archive/chest_xray/chest_xray/train"

CATEGORIES = ["PNEUMONIA", "NORMAL"]

training_data = []

def create_training_data():
    for category in CATEGORIES:  # do covid+ and covid-

        path = os.path.join(DATADIR,category)  # create path to covid+ and covid-
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=covid+ 1=covid-

        for img in tqdm(os.listdir(path)):  # iterate over each image per covid+ and covid-
            try:
                img_array = cv2.imread(os.path.join(path,img))  # convert to array
                # convert the image to ycbcr
                #y is luma: intensity infos
                #cbcr : chroma : color infos
                image = cv2.cvtColor(img_array, cv2.COLOR_BGR2YCR_CB)
                yi,cr,cb = cv2.split(image)
                a= 1.1 #where a is a constant slightly larger than 1, used to avoid the resulting image being too bright
                t = 0.05 #a small positive number used to avoid being divided by zero 
                im = Image.fromarray(yi)
                # applying the max filter 
                maxv = im.filter(ImageFilter.MaxFilter(size = 9)) 
                # applying the min filter 
                minv = maxv.filter(ImageFilter.MinFilter(size = 9))
                im_max = np.array(minv, dtype='float32')
                L = a*(im_max+t)
                ill = np.array(L, dtype='uint8')#illumination estimation
                #LBP of Ill
                lbp_ill = local_binary_pattern(ill, 32, 3, 'uniform')
                lbp_illl = np.array(lbp_ill, 'uint8')
                #LBP CR and CB
                lbp_cr = local_binary_pattern(cr, 32, 3, 'uniform')
                lbp_cb = local_binary_pattern(cb, 32, 3, 'uniform')
                lbp_cr = np.array(lbp_cr, 'uint8')
                lbp_cb = np.array(lbp_cb, 'uint8')
                #LBPs Histogran
                hist5 = cv2.calcHist([lbp_illl], [0], None, [256], [0, 256])
                hist7  = cv2.calcHist([lbp_cr], [0], None, [256], [0, 256])
                hist8  = cv2.calcHist([lbp_cb], [0], None, [256], [0, 256])
                #Normalize histograms
                cv2.normalize(hist5, hist5)
                hs5 = hist5.flatten()
                cv2.normalize(hist7, hist7)
                hs7 = hist7.flatten()
                cv2.normalize(hist8, hist8)
                hs8 = hist8.flatten()
                #Concatenate histograms
                fall = np.concatenate((hs5,hs7,hs8))
                training_data.append([fall, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            except OSError as e:
                print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))

create_training_data()
 #%%
len(training_data)
#%%
random.shuffle(training_data)

#Our training_data is a list, meaning it's mutable, so it's now nicely shuffled. 
#We can confirm this by iterating over a few of the initial samples and printing out the class.

for sample in training_data[:10]:
    print(sample[1])
#%%
fx = []
fy = []

for features,label in training_data:
    fx.append(features)
    fy.append(label)
print('feature vector size :',len(fx[0]))
#%%
feature_matrix = np.array(fx)
feature_matrix.shape
#%%
#set the test_size. Here is 0.2
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(feature_matrix,fy,test_size=.2,random_state=0)

#%%
#all classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

training_start = datetime.now()
# define support vector classifiers
lr = LogisticRegression()
tree = DecisionTreeClassifier()
knn = KNeighborsClassifier()
lda = LinearDiscriminantAnalysis()
n_bayes = GaussianNB()
svm = SVC(kernel='linear', probability=True,degree=3, gamma='auto')

# fit model
lr.fit(X_train, y_train)
tree.fit(X_train, y_train)
knn.fit(X_train, y_train)
lda.fit(X_train, y_train)
n_bayes.fit(X_train, y_train)
svm.fit(X_train, y_train)

training_end = datetime.now()
print('Training Duration: {}'.format(training_end - training_start))
#%%
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

#10. Score model

#Predictions are calculated on the test data, followed by the accuracy.

pred_start = datetime.now()
# generate predictions
lr_pred = lr.predict(X_test)
tree_pred = tree.predict(X_test)
knn_pred = knn.predict(X_test)
lda_pred = lda.predict(X_test)
n_bayes_pred = n_bayes.predict(X_test)
svm_pred = svm.predict(X_test)

pred_end = datetime.now()
print('Prediction Duration: {}'.format(pred_end- pred_start))

# calculate accuracy
lr_accuracy = accuracy_score(y_test, lr_pred)
tree_accuracy = accuracy_score(y_test, tree_pred)
knn_accuracy = accuracy_score(y_test, knn_pred)
lda_accuracy = accuracy_score(y_test, lda_pred)
n_bayes_accuracy = accuracy_score(y_test, n_bayes_pred)
svm_accuracy = accuracy_score(y_test, svm_pred)


print('LogisticRegression Model accuracy is: ', lr_accuracy)
print('Tree Model accuracy is: ', tree_accuracy)
print('KNN Model accuracy is: ', knn_accuracy)
print('LDA Model accuracy is: ', lda_accuracy)
print('NBayes Model accuracy is: ', n_bayes_accuracy)
print('SVM Model accuracy is: ', svm_accuracy)
#%%
#1. ROC curve + AUC
# generate predictions
lr_pred = lr.predict_proba(X_test)
tree_pred = tree.predict_proba(X_test)
knn_pred = knn.predict_proba(X_test)
lda_pred = lda.predict_proba(X_test)
n_bayes_pred = n_bayes.predict_proba(X_test)
svm_pred = svm.predict_proba(X_test)


# select the probabilities for label 1.0
#y_proba = probabilities[:, 1]
lr_proba = lr_pred[:, 1]
tree_proba = tree_pred[:, 1]
knn_proba = knn_pred[:, 1]
lda_proba = lda_pred[:, 1]
n_bayes_proba = n_bayes_pred[:, 1]
svm_proba = svm_pred[:, 1]

# calculate false positive rate and true positive rate at different thresholds
#false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_proba, pos_label=1)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, tree_proba, pos_label=1)
false_positive_rate1, true_positive_rate1, thresholds1 = roc_curve(y_test, knn_proba, pos_label=1)
false_positive_rate2, true_positive_rate2, thresholds2 = roc_curve(y_test, lr_proba, pos_label=1)
false_positive_rate3, true_positive_rate3, thresholds3 = roc_curve(y_test, lda_proba, pos_label=1)
false_positive_rate4, true_positive_rate4, thresholds4 = roc_curve(y_test, n_bayes_proba, pos_label=1)
false_positive_rate5, true_positive_rate5, thresholds5 = roc_curve(y_test, svm_proba, pos_label=1)

# calculate AUC
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc1 = auc(false_positive_rate1, true_positive_rate1)
roc_auc2 = auc(false_positive_rate2, true_positive_rate2)
roc_auc3 = auc(false_positive_rate3, true_positive_rate3)
roc_auc4 = auc(false_positive_rate4, true_positive_rate4)
roc_auc5 = auc(false_positive_rate5, true_positive_rate5)

fig = plt.figure(figsize=(12, 8))

plt.title('Receiver Operating Characteristic')
# plot the false positive rate on the x axis and the true positive rate on the y axis
roc_plot = plt.plot(false_positive_rate,true_positive_rate,label='Tree = {:0.2f}'.format(roc_auc),linewidth=2)
roc_plot1 = plt.plot(false_positive_rate1,true_positive_rate1,label='KNN = {:0.2f}'.format(roc_auc1),linewidth=2)
roc_plot2 = plt.plot(false_positive_rate2,true_positive_rate2,label='LR = {:0.2f}'.format(roc_auc2),linewidth=2)
roc_plot3 = plt.plot(false_positive_rate3,true_positive_rate3,label='LDA = {:0.2f}'.format(roc_auc3),linewidth=2)
roc_plot4 = plt.plot(false_positive_rate4,true_positive_rate4,label='NB = {:0.2f}'.format(roc_auc4),linewidth=2)
roc_plot5 = plt.plot(false_positive_rate5,true_positive_rate5,label='SVM = {:0.2f}'.format(roc_auc5),linewidth=2)

plt.legend(loc=0, fontsize='20')
plt.plot([0,1], [0,1], ls='--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#fig.savefig('covid.png')
