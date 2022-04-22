# General Steps for Building Classification Models with a given data set:
# Step 1: Clean your data set (Impute missing values, remove samples with missing values etc.). Cleaning actions
# are problem specific, not performed in this file. Use an already cleaned data set for experimentation
# Step 2: Balance your data set. This code is written to work with unbalanced data sets. Use one of the balancing methods
# listed below. These techniques have their own parameters which can be tuned. By default one is selected
# Step 3: Transform your features (normalize, standardize etc.) to different range of values (e.g. [0,1])
# Step 4: Perform feature selection (reduce the number of features, keep important features etc.)
# Use one of the feature selection methods listed below. By default one is selected
# Step 5: Optimize hyper-parameters (Tune parameters) with K-fold stratified cross-validation
# Step 6: Build the actual models and perfrom evaluations with K-fold stratified cross-validation

######Important#############
# For handling missing Values: use 'sklearn.preprocessing.Imputer'
# Use 'sklearn.preprocessing.OneHotEncoder', (OneHotEncoder encoding) to feed categorical predictors to linear models
# and SVMs with the standard kernels.

from collections import Counter
from time import time
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

from sklearn import manifold, datasets
from sklearn.datasets import make_classification
from imblearn.under_sampling import ClusterCentroids
import datetime

import sklearn as sk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import array
import statistics
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import svm
from sklearn import preprocessing
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import average_precision_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import OneSidedSelection
import os
from sklearn.metrics import confusion_matrix
from collections import OrderedDict

from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# print(len(data[0,:]))
print("Process Started...")

NUM_RUNS = 1  # The number of trials
NUM_JOBS = 8  # The number of threads for parallelization. Each CPU core can handle 2 threads
SEED = 42  # SEED used by the random number generator. Used for re-producability
TEST_TRAIN_RATIO = 0.2
FIGUREWIDTH = 18
FIGUREHEIGHT = 15
INCHWIDTH = 36
INCHHEIGHT = 24
FONTSIZE = 20
DPI = 500

path = os.getcwd()
path = path[0:-5]

result = []
TARGET_NAMES = ["Non_Violated", "Violated"]
FEATURES_LIST = ['num_of_tasks', 'average_num_of_instances_per_job',
                 'average_num_cpus_req_per_job', 'average_mem_req_per_job',
                 'average_total_max_real_cpu_num_per_job',
                 'average_total_average_real_cpu_num_per_job',
                 'average_total_max_mem_usage_per_job',
                 'average_total_average_mem_usage_per_job', 'average_CPU_per_job',
                 'average_nor_mem_per_job']


def prepareData(filename):
    ########################### Load Data Set From File################################################################
    # Load data set with python panda read_csv method
    dataSetFileName = path + "/input/jobs/" + filename + "small.xlsx"
    dataset1 = pd.read_excel(dataSetFileName)
    numOfrows, numOfColumns = dataset1.shape

    dataset_data = dataset1.iloc[:, 0:numOfColumns - 1]  # all predictor variable
    dataset_target = dataset1.iloc[:,
                     numOfColumns - 1]  # dependent variable. Assumption is that the last column contains
    # the dependent variable

    # LabelEncoder will convert string class names into numeric class names (e.g. 0,1,2 etc.)
    labelEncoder = preprocessing.LabelEncoder()
    convertedIntoClasses = labelEncoder.fit_transform(list(dataset_target))
    encodedClasses = np.unique(np.array(convertedIntoClasses))  # unique labels in the converted labels
    print("New names for classes:", encodedClasses)
    print("Actual names for classes:", labelEncoder.inverse_transform(encodedClasses))
    print()

    # Use the newly encoded class names
    dataset_target = convertedIntoClasses

    print("Count of Samples Per Class in unbalanced state:")

    data_all = dataset_data
    label_all = dataset_target
    return data_all, label_all


def plot2D(imbalanceTechnique, X, Y, filename, scaler,init, n_iter, lw, alpha,learning_rate):

    fig = plt.figure(figsize=(FIGUREWIDTH + 3, FIGUREHEIGHT + 2))
    n_points = X.shape[0]
    tsne = TSNE(n_components=2, random_state=SEED, init=init, n_iter=n_iter)
    Y_tsne = tsne.fit_transform(X)
    print(tsne.embedding_)
    print("888888888_type of Y:", Y.shape)
    print("888888888_Y:", Y)

    colors = ['red', 'black', 'blue']

    # Y_tsne = tsne.fit_transform(X)  # 转换后的输出
    plt.scatter(Y_tsne[Y == 0, 0], Y_tsne[Y == 0, 1], label="Non_violated", color=colors[0], alpha=alpha, lw=lw)
    plt.scatter(Y_tsne[Y == 1, 0], Y_tsne[Y == 1, 1], label="Violated", color=colors[1], alpha=alpha, lw=lw)
    # plt.title("t-SNE (%.2g sec)" % (t1 - t0))
    # ax.xaxis.set_major_formatter(NullFormatter())  # 设置标签显示格式为空
    # ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')
    plt.legend(loc="upper right", fontsize=FONTSIZE)
    plt.xlim((-150, 150))
    plt.ylim(-150,150)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)

    myfig = plt.gcf()
    # myfig.set_size_inches(INCHWIDTH,INCHHEIGHT)
    myfig.set_figwidth(FIGUREWIDTH)
    myfig.set_figheight(FIGUREHEIGHT)

    if(scaler == None):
        myfig.savefig(
            path + "/figures/raw_2D_" + imbalanceTechnique + "_" + filename + "_" + init + "_" + str(n_iter) + "_ " + str(lw) + "_" + str(alpha) + "_" + str(learning_rate) +  "_.jpg",
        dpi = DPI, format = "jpg", bbox_inches = 'tight')
    elif(scaler != None):
        myfig.savefig(
            path + "/figures/scaled_2D_" + imbalanceTechnique + "_" + filename + "_" + init + "_" + str(n_iter) + "_ " + str(lw) + "_" + str(alpha) + "_" + str(learning_rate) +  "_.jpg",
        dpi = DPI, format = "jpg", bbox_inches = 'tight')



def plot3D(imbalanceTechnique, X, Y, filename, scaler, init, n_iter, lw, alpha):
    # plt.clf()

    ax = plt.subplot(1, 2, 2, projection='3d')

    fig = plt.figure(figsize=(FIGUREWIDTH + 3, FIGUREHEIGHT + 2))
    n_points = X.shape[0]


    tsne = TSNE(n_components=3, random_state=SEED, init=init, n_iter=n_iter)
    Y_tsne = tsne.fit_transform(X)
    print(tsne.embedding_)
    print("888888888_type of Y:", Y.shape)
    print("888888888_Y:", Y)

    colors = ['red', 'black', 'blue']

    # Y_tsne = tsne.fit_transform(X)  # 转换后的输出
    plt.scatter(Y_tsne[Y == 0, 0], Y_tsne[Y == 0, 1], Y_tsne[Y == 0, 2], label="NO_Violation", color=colors[0],
                alpha=alpha, lw=lw)
    plt.scatter(Y_tsne[Y == 1, 0], Y_tsne[Y == 1, 1], Y_tsne[Y == 1, 2], label="Violation_Type_1", color=colors[1],
                alpha=alpha, lw=lw)
    plt.scatter(Y_tsne[Y == 2, 0], Y_tsne[Y == 2, 1], Y_tsne[Y == 2, 2], label="Violation_Type_2", color=colors[2],
                alpha=alpha, lw=lw)

    # plt.title("t-SNE (%.2g sec)" % (t1 - t0))
    # ax.xaxis.set_major_formatter(NullFormatter())  # 设置标签显示格式为空
    # ax.yaxis.set_major_formatter(NullFormatter())
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    ax.set_title("First three PCA directions")
    ax.set_xlabel("1st eigenvector")
    ax.set_ylabel("2nd eigenvector")
    ax.set_zlabel("3rd eigenvector")
    ax.view_init(30, 10)

    myfig = plt.gcf()
    # myfig.set_size_inches(INCHWIDTH,INCHHEIGHT)
    myfig.set_figwidth(FIGUREWIDTH)
    myfig.set_figheight(FIGUREHEIGHT)

    if(scaler == None):
        myfig.savefig(
            path + "/figures/raw_3D" + imbalanceTechnique + "_" + filename + "_" + init + "_" + n_iter + "_ " + lw + "_" + alpha + "_.jpg",
        dpi = DPI, format = "jpg", bbox_inches = 'tight')
    elif(scaler != None):
        myfig.savefig(
            path + "/figures/scaled_3D" + imbalanceTechnique + "_" + filename + "_" + init + "_" + n_iter + "_ " + lw + "_" + alpha + "_.jpg",
        dpi = DPI, format = "jpg", bbox_inches = 'tight')




def run(imbalanceTechnique, featureTrans, featureSelection, clfName, data_all, label_all, filename, init,
        n_iter, lw, alpha,learning_rate):
    print("imbalanceTechnique:", imbalanceTechnique)
    print("featureTrans:", featureTrans)
    print("featureSelection:", featureSelection)
    print("clfName:", clfName)

    if (imbalanceTechnique == 'RandomUnderSampler'):
        imbalanceHandler = RandomUnderSampler(random_state=SEED)
    elif (imbalanceTechnique == 'ClusterCentroids'):
        imbalanceHandler = ClusterCentroids(random_state=SEED)
    elif (imbalanceTechnique == 'NearMiss1'):
        imbalanceHandler = NearMiss(version=1, random_state=SEED, n_jobs=-1)
    elif (imbalanceTechnique == 'NearMiss2'):
        imbalanceHandler = NearMiss(version=2, random_state=SEED, n_jobs=-1)
    elif (imbalanceTechnique == 'NearMiss3'):
        imbalanceHandler = NearMiss(version=3, random_state=SEED, n_jobs=-1)
    elif (imbalanceTechnique == 'OnesidedSelcection'):
        imbalanceHandler = OneSidedSelection(sampling_strategy='auto', random_state=SEED, n_jobs=-1)
    elif (imbalanceTechnique == 'RandomOverSampler'):
        imbalanceHandler = RandomOverSampler(random_state=SEED)
    elif (imbalanceTechnique == 'SMOTE_regular'):
        imbalanceHandler = SMOTE(random_state=SEED, ratio='auto', kind='regular', k_neighbors=5, n_jobs=NUM_JOBS)
    elif (imbalanceTechnique == 'SMOTE_svm'):
        imbalanceHandler = SMOTE(ratio='auto', random_state=SEED, kind="svm")
    elif (imbalanceTechnique == 'SMOTE_borderline1'):
        imbalanceHandler = SMOTE(ratio='auto', random_state=SEED, kind="borderline1")
    elif (imbalanceTechnique == 'SMOTE_borderline2'):
        imbalanceHandler = SMOTE(ratio='auto', random_state=SEED, kind="borderline2")
    elif (imbalanceTechnique == 'SMOTE_ENN'):
        imbalanceHandler = SMOTEENN(ratio='auto', random_state=SEED)
    elif (imbalanceTechnique == 'SMOTE_Tomek'):
        smoteObject = SMOTE(random_state=SEED, ratio='auto', kind='regular', k_neighbors=5, n_jobs=NUM_JOBS)
        tomekObject = TomekLinks(random_state=SEED, ratio='auto', n_jobs=NUM_JOBS)
        imbalanceHandler = SMOTETomek(random_state=SEED, ratio='auto', smote=smoteObject, tomek=tomekObject)
    elif (imbalanceTechnique == 'No_Resampling'):
        imbalanceHandler = None

    if (imbalanceHandler != None):
        X_resampled, y_resampled = imbalanceHandler.fit_sample(data_all, label_all)
        print("Count of Samples Per Class in balanced state:")
        dataset_data = X_resampled
        dataset_target = y_resampled
    else:
        print("this is 555555555")
        dataset_data = np.array(data_all)
        dataset_target = np.array(label_all)



    # Please do not comment out this lines.
    # This part is common for the above balancing methods. Executed after the balancing process.
    X_dataset, Y_dataset = dataset_data, dataset_target
    print("X_dataset:", X_dataset.shape)
    X_data, Y_data = X_dataset[:, 0:len(X_dataset[0])], Y_dataset
    print("Before feature selection. Note the number of predictors (second value)")
    print(X_data.shape)  # reduced data set number of rows and columns

    if (featureTrans == "StandardScaler"):
        #####################Feature transformation (only one of the following methods need to be used)############################################
        # print("Feature values after transformation")
        # Feature Transformation 1: Standardize features to 0 mean and unit variance
        scaler = preprocessing.StandardScaler().fit(X_data)
    elif (featureTrans == "MinMaxScaler"):
        # Feature Transformation 2: transforms features by scaling each feature to a [0,1] range.
        scaler = preprocessing.MinMaxScaler().fit(X_data)
    elif (featureTrans == "Normalizer"):
        scaler = preprocessing.Normalizer().fit(X_data)
    elif (featureTrans == "No_FeatureTrans"):
        scaler = None

    if (scaler != None):
        X_data_transformed = scaler.transform(X_data)
        X_data = X_data_transformed
    else:
        X_data = X_data

    plot2D(imbalanceTechnique, X_data, Y_data, filename, scaler,init, n_iter, lw, alpha,learning_rate)
    #plot3D(imbalanceTechnique, X_data, Y_data, filename, scaler,init, n_iter, lw, alpha,scaler)



resampingMethod = ["No_Resampling", "SMOTE_Tomek", "SMOTE_ENN", "RandomOverSampler", "SMOTE_regular", "SMOTE_svm",
                   "SMOTE_borderline1", "SMOTE_borderline2", "RandomUnderSampler", "ClusterCentroids", "NearMiss1",
                   "NearMiss2", "NearMiss3", "OnesidedSelcection"]
scalerMethod = ["No_FeatureTrans","StandardScaler","MinMaxScaler"]
featureSelMethod = ["No_FeatureSelection"]
clfMethod = ["OneVsRestRF"]

if __name__ == '__main__':
    labelBinarizer = preprocessing.LabelBinarizer()
    starttime = datetime.datetime.now()

    filename = "sla99"
    data_all, label_all = prepareData(filename)

    for resampling in resampingMethod[0:1]:
        for scaler in scalerMethod[2:]:
            for featureSel in featureSelMethod:
                for clf in clfMethod:
                    #run(resampling, scaler, featureSel, clf, data_all, label_all, filename, 'random', 5000, 1, 0.9, 10)
                    #run(resampling, scaler, featureSel, clf, data_all, label_all, filename, 'random', 5000, 1, 0.9, 200)
                    #run(resampling, scaler, featureSel, clf, data_all, label_all, filename, 'random', 5000, 1, 0.9,1000)
                    #run(resampling, scaler, featureSel, clf, data_all, label_all, filename, 'pca', 5000, 1, 0.9,10)
                    run(resampling, scaler, featureSel, clf, data_all, label_all, filename, 'pca', 5000, 1, 0.9,200)
                    #run(resampling, scaler, featureSel, clf, data_all, label_all, filename, 'pca', 5000, 1, 0.9,1000)
                    #run(resampling, scaler, featureSel, clf, data_all, label_all, filename, 'pca', 3000, 1, 0.9,10)
                    #run(resampling, scaler, featureSel, clf, data_all, label_all, filename, 'pca', 3000, 1, 0.9,200)
                    #run(resampling, scaler, featureSel, clf, data_all, label_all, filename, 'pca', 3000, 1, 0.9,1000)
                    #run(resampling, scaler, featureSel, clf, data_all, label_all, filename, 'random', 3000, 1, 0.9,10)
                    #run(resampling, scaler, featureSel, clf, data_all, label_all, filename, 'random', 3000, 1, 0.9,200)
                    #run(resampling, scaler, featureSel, clf, data_all, label_all, filename, 'random',3000, 1, 0.9,1000)



    endtime = datetime.datetime.now()

    print("consuming total time by minutes:", (endtime - starttime) / 3600)
