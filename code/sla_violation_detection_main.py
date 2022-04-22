'''
this code is for my journal paper: 
Zeng, Xuezhi, et al. "Detection of SLA Violation for Big Data Analytics Applications in Cloud." IEEE
Transactions on Computers 70.5 (2020): 746-758.

'''
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

import matplotlib.pyplot as plt
import numpy as np
import itertools
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from collections import Counter
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature
from imblearn.under_sampling import NeighbourhoodCleaningRule
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from keras.models import Sequential
from keras.layers import Dense
# from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as imbPipeline
from sklearn.datasets import make_classification
from imblearn.under_sampling import ClusterCentroids
import datetime
from sklearn.svm import SVC

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score
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

from sklearn.metrics import confusion_matrix
from collections import OrderedDict

from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from imblearn.over_sampling import ADASYN
from sklearn.metrics import roc_curve, auc


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

N_CLASSES = 1
SPLITS = 10
TARGET_NAMES = ["No_Violation", "Violation_Type_1"]
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


def match(arr, para):
    result = []
    for each in arr:
        # print(each)
        temp = (each == para)
        if (set(temp) == {True}):
            result.append(True)
        else:
            result.append(False)
    return result


def plot_confusion_matrix(seqOfRun, imbalanceTechnique, featureTrans, featureSelection, clfName, cm, tag,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):

    plt.clf()

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(FIGUREWIDTH, FIGUREHEIGHT))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45, fontsize=FONTSIZE)
        plt.yticks(tick_marks, target_names, fontsize=FONTSIZE)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(i, j, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[j, i] > thresh else "black", size=FONTSIZE)
        else:
            plt.text(i, j, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[j, i] > thresh else "black", size=FONTSIZE)

    plt.tight_layout()
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.xlabel('True label\n\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass), fontsize=FONTSIZE)
    plt.ylabel('Predicted label', fontsize=FONTSIZE)

    myfig = plt.gcf()
    # fig.set_size_inches(cm2inch(40, 20))
    # fig.set_size_inches(cm2inch(40*4, 20*4))
    # myfig.set_size_inches(INCHWIDTH,INCHHEIGHT)
    myfig.set_figwidth(FIGUREWIDTH)
    myfig.set_figheight(FIGUREHEIGHT)

    if (tag == 1):
        myfig.savefig(
            path + "/figures/matrix_[(Y_data.argmax(axis=1), Y_pred.argmax(axis=1)]_" + str(
                seqOfRun) + "_" + imbalanceTechnique + "_" + featureTrans + "_" + featureSelection + "_" + clfName + ".jpg",
            dpi=DPI, format="jpg", bbox_inches='tight')
    elif (tag == 2):
        myfig.savefig(
            path + "/figures/matrix_[Y_data.argmax(axis=1), Y_pred2)]_" + str(
                seqOfRun) + "_" + imbalanceTechnique + "_" + featureTrans + "_" + featureSelection + "_" + clfName + ".jpg",
            dpi=DPI, format="jpg", bbox_inches='tight')
    elif (tag == 3):
        myfig.savefig(
            path + "/figures/matrix_[Y_data.argmax(axis=1), Y_pred_proba_transform)]_" + str(
                seqOfRun) + "_" + imbalanceTechnique + "_" + featureTrans + "_" + featureSelection + "_" + clfName + ".jpg",
            dpi=DPI, format="jpg", bbox_inches='tight')


def Fbeta_score(Y_true, Y_hat, beta):
    return metrics.fbeta_score(Y_true, Y_hat, beta=beta, average='macro')
    # return 2 * precision * recall / (precision + recall)

def calFbetaScore(precision, recall, beta):
    if(((np.square(beta)*precision) + recall) !=0):
        return (1+np.square(beta))*precision * recall / ((np.square(beta)*precision) + recall)
    else:
        return 0


def plot_FeatureImportance(seqOfRun, imbalanceTechnique, featureTrans, featureSelection, clfName,
                           features_contributions):
    plt.clf()
    plt.barh(range(len(features_contributions)), features_contributions, color='rgb', tick_label=FEATURES_LIST)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    myfig = plt.gcf()
    # myfig.set_size_inches(INCHWIDTH, INCHHEIGHT)
    myfig.set_figwidth(FIGUREWIDTH)
    myfig.set_figheight(FIGUREHEIGHT)

    myfig.savefig(
        path + "/figures/featureImportance_" + str(
            seqOfRun) + "_" + imbalanceTechnique + "_" + featureTrans + "_" + featureSelection + "_" + clfName + ".jpg",
        dpi=DPI, format="jpg", bbox_inches='tight')

    # myfig.show()


def plot_ROC(seqOfRun, imbalanceTechnique, featureTrans, featureSelection, clfName, tag, Y_true, Y_pred_result,
             target_names=TARGET_NAMES):
    plt.clf()
    NUMBER_OF_CATAGORIES = 0
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, _ = roc_curve(Y_true, Y_pred_result)
    roc_auc = auc(fpr, tpr)

    # Compute micro-average ROC curve and ROC area

    import numpy as np
    from scipy import interp
    from itertools import cycle
    # Compute macro-average ROC curve and ROC area


    # Plot all ROC curves
    plt.figure(figsize=(FIGUREWIDTH, FIGUREHEIGHT))
    lw = 2


    colors = cycle(['black', 'blue', 'red'])
    plt.plot(fpr, tpr, color='black', lw=lw,
             label='AUC = %0.2f' % roc_auc)
        # label = 'ROC curve of {0}'.format(target_names[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.xlabel('False Positive Rate', fontsize=FONTSIZE)
    plt.ylabel('True Positive Rate', fontsize=FONTSIZE)
    plt.title('ROC Curve')
    plt.legend(loc="lower right", fontsize=FONTSIZE)

    myfig = plt.gcf()
    # myfig.set_size_inches(INCHWIDTH, INCHHEIGHT)
    myfig.set_figwidth(FIGUREWIDTH)
    myfig.set_figheight(FIGUREHEIGHT)

    if (tag == 1):
        myfig.savefig(path + "/figures/auc_[Y_data,Y_pred]_" + str(
            seqOfRun) + "_" + imbalanceTechnique + "_" + featureTrans + "_" + featureSelection + "_" + clfName + ".jpg",
                      dpi=DPI, format="jpg", bbox_inches='tight')
    elif (tag == 2):
        myfig.savefig(path + "/figures/auc_[Y_data, Y_pred2_encoding]_" + str(
            seqOfRun) + "_" + imbalanceTechnique + "_" + featureTrans + "_" + featureSelection + "_" + clfName + ".jpg",
                      dpi=DPI, format="jpg", bbox_inches='tight')
    elif (tag == 3):
        myfig.savefig(
            path + "/figures/auc_[Y_data,Y_pred_proba]_" + str(
                seqOfRun) + "_" + imbalanceTechnique + "_" + featureTrans + "_" + featureSelection + "_" + clfName + ".jpg",
            dpi=DPI, format="jpg", bbox_inches='tight')
    return roc_auc
    # myfig.show()


def show_values(pc, fmt="%.2f", **kw):
    '''
    Heatmap with text in each cell with matplotlib's pyplot
    Source: https://stackoverflow.com/a/25074150/395857
    By HYRY
    '''
    # print("type of pc:", type(pc))

    pc.update_scalarmappable()
    # ax = pc.get_axes()
    ax = pc.axes

    for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, fontsize=FONTSIZE, **kw)


def cm2inch(*tupl):
    '''
    Specify figure size in centimeter in matplotlib
    Source: https://stackoverflow.com/a/22787457/395857
    By gns-ank
    '''
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


def heatmap(seqOfRun, imbalanceTechnique, featureTrans, featureSelection, clfName, tag, AUC, title, xlabel, ylabel,
            xticklabels, yticklabels, figure_width=40, figure_height=20,
            correct_orientation=False, cmap='Blues'):
    '''
    Inspired by:
    - https://stackoverflow.com/a/16124677/395857
    - https://stackoverflow.com/a/25074150/395857
    '''
    plt.clf()
    fig, ax = plt.subplots()
    # c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap='RdBu', vmin=0.0, vmax=1.0)
    c = ax.pcolor(AUC, edgecolors='k', linestyle='dashed', linewidths=0.2, cmap=cmap)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]), minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]), minor=False)

    # set tick labels
    # ax.set_xticklabels(np.arange(1,AUC.shape[1]+1), minor=False)
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    # set title and x/y labels
    # plt.title(title)
    plt.xlabel(xlabel, fontsize=FONTSIZE)
    plt.ylabel(ylabel, fontsize=FONTSIZE)

    # Remove last blank column
    plt.xlim((0, AUC.shape[1]))

    # Turn off all the ticks
    ax = plt.gca()
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # Add color bar
    plt.colorbar(c)

    # Add text in each cell
    show_values(c)

    # Proper orientation (origin at the top left instead of bottom left)
    if correct_orientation:
        ax.invert_yaxis()
        ax.xaxis.tick_top()

        # resize

    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    myfig = plt.gcf()
    # myfig.set_size_inches(INCHWIDTH, INCHHEIGHT)
    myfig.set_figwidth(FIGUREWIDTH)
    myfig.set_figheight(FIGUREHEIGHT)

    if (tag == 1):
        myfig.savefig(path + "/figures/Report_[Y_data, Y_pred]_" + str(
            seqOfRun) + "_" + imbalanceTechnique + "_" + featureTrans + "_" + featureSelection + "_" + clfName + ".jpg",
                      dpi=DPI, format="jpg", bbox_inches='tight')
    elif (tag == 2):
        myfig.savefig(path + "/figures/Report_[Y_data.argmax(axis=1), Y_pred.argmax(axis=1)]_" + str(
            seqOfRun) + "_" + imbalanceTechnique + "_" + featureTrans + "_" + featureSelection + "_" + clfName + ".jpg",
                      dpi=DPI, format="jpg", bbox_inches='tight')
    elif (tag == 3):
        myfig.savefig(path + "/figures/Report_[Y_data.argmax(axis=1), Y_pred2)]_" + str(
            seqOfRun) + "_" + imbalanceTechnique + "_" + featureTrans + "_" + featureSelection + "_" + clfName + ".jpg",
                      dpi=DPI, format="jpg", bbox_inches='tight')
    elif (tag == 4):
        myfig.savefig(path + "/figures/Report_[Y_data.argmax(axis=1), Y_pred_proba_transform)]_" + str(
            seqOfRun) + "_" + imbalanceTechnique + "_" + featureTrans + "_" + featureSelection + "_" + clfName + ".jpg",
                      dpi=DPI, format="jpg", bbox_inches='tight')
    # myfig.show()


def plot_classification_report(seqOfRun, imbalanceTechnique, featureTrans, featureSelection, clfName, tag,
                               classification_report, title='Classification report ', cmap='Blues'):
    '''
    Plot scikit-learn classification report.
    Extension based on https://stackoverflow.com/a/31689645/395857
    '''
    lines = classification_report.split('\n')

    classes = []
    plotMat = []
    support = []
    class_names = []
    for line in lines[2: (len(lines) - 1)]:
        t = line.strip().split()
        if len(t) < 2: continue
        if (len(t) == 5):
            classes.append(t[0])
        if (len(t) == 6):
            t_new = []
            t_new.append(t[0] + "_" + t[1])
            t_new.extend(t[2:])
            t = t_new
            classes.append(t[0])

        v1 = [float(x) for x in t[1: len(t) - 1]]
        v2 = int(t[len(t) - 1])
        v1.append(v2)
        v = v1
        support.append(int(t[-1]))
        class_names.append(t[0])
        plotMat.append(v)

    print('plotMat: {0}'.format(plotMat))
    print("type of plotMat:", type(plotMat))
    print('support: {0}'.format(support))

    xlabel = 'Metrics'
    ylabel = 'Classes'
    xticklabels = ['Precision', 'Recall', 'F1-score', "Support"]
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup in enumerate(support)]
    figure_width = 25
    figure_height = len(class_names) + 12
    correct_orientation = True
    heatmap(seqOfRun, imbalanceTechnique, featureTrans, featureSelection, clfName, tag, np.array(plotMat), title,
            xlabel,
            ylabel, xticklabels, yticklabels, FIGUREWIDTH, FIGUREHEIGHT,
            correct_orientation, cmap=cmap)


def calAllMetrics(seqOfRun, imbalanceTechnique, featureTrans, featureSelection, clfName, Y_true, Y_pred,auc_area_Y_pred,pr_area_Y_pred,x,z):
    r00, r10, r01, r11 = confusion_matrix(Y_true, Y_pred).ravel()

    print("r00:",r00)
    print("r01:",r01)
    print("r10:",r10)
    print("r11:",r11)


    if((r00 + r01 ) != 0):
        precision_no_violation = r00 / (r00 + r01 )
    else:
        precision_no_violation = 0

    if((r10 + r11 ) !=0 ):
        precision_violation_type_1 = r11 / (r10 + r11)
    else:
        precision_violation_type_1 = 0


    if((r00 + r10) !=0):
        recall_no_violation = r00 / (r00 + r10)
    else:
        recall_no_violation = 0

    if((r01 + r11 ) !=0):
        recall_violation_type_1 = r11 / (r01 + r11 )
    else:
        recall_violation_type_1 = 0


    print("precision_no_violation:", precision_no_violation)
    print("precision_violation_type_1:", precision_violation_type_1)
    print("recall_no_violation:", recall_no_violation)
    print("recall_violation_type_1:", recall_violation_type_1)

    precision_macro = (precision_score(Y_true, Y_pred, average='macro') )
    precision_micro = (precision_score(Y_true, Y_pred, average='micro') )
    precision_weighted = (precision_score(Y_true, Y_pred,average='weighted') )

    recall_macro = (recall_score(Y_true, Y_pred, average='macro') )
    recall_micro = (recall_score(Y_true, Y_pred, average='micro'))
    recall_weighted = (recall_score(Y_true, Y_pred, average='weighted'))


    best_profit_0 = z[0]*(r00+r10)
    best_profit_1 = z[1]*(r00+r10)
    best_profit_2 = z[2]*(r00+r10)
    best_profit_3 = z[3]*(r00+r10)
    best_profit_4 = z[4]*(r00+r10)
    best_profit_5 = z[5]*(r00+r10)
    best_profit_6 = z[6]*(r00+r10)
    best_profit_7 = z[7]*(r00+r10)
    best_profit_8 = z[8]*(r00+r10)


    penalty_0 = z[0]*(r10) +  (1+x[0])*(r01)
    profit_0 = best_profit_0-penalty_0

    penalty_1 = z[1]*(r10) +  (1+x[1])*(r01)
    profit_1 = best_profit_1 - penalty_1

    penalty_2 = z[2]*(r10) +  (1+x[2])*(r01)
    profit_2 = best_profit_2 - penalty_2

    penalty_3 = z[3]*(r10) +  (1+x[3])*(r01)
    profit_3 = best_profit_3 - penalty_3

    penalty_4 = z[4]*(r10) +  (1+x[4])*(r01)
    profit_4 = best_profit_4 - penalty_4

    penalty_5 = z[5]*(r10) +  (1+x[5])*(r01)
    profit_5 = best_profit_5 - penalty_5

    penalty_6 = z[6]*(r10) +  (1+x[6])*(r01)
    profit_6 = best_profit_6 - penalty_6

    penalty_7 = z[7]*(r10) +  (1+x[7])*(r01)
    profit_7 = best_profit_7 - penalty_7

    penalty_8 = z[8]*(r10) +  (1+x[8])*(r01)
    profit_8 = best_profit_8 - penalty_8


    accuracy = (r00 + r11 ) / (r00 + r10  + r01 + r11)

    f_half_score_no_violation = calFbetaScore(precision_no_violation, recall_no_violation, beta=0.5)
    f_half_score_violation_type_1 = calFbetaScore(precision_violation_type_1, recall_violation_type_1, beta=0.5)

    f1_score_no_violation = calFbetaScore(precision_no_violation, recall_no_violation, beta=1)
    f1_score_violation_type_1 = calFbetaScore(precision_violation_type_1, recall_violation_type_1, beta=1)


    f2_score_no_violation = calFbetaScore(precision_no_violation, recall_no_violation, beta=2)
    f2_score_violation_type_1 = calFbetaScore(precision_violation_type_1, recall_violation_type_1, beta=2)

    print("f1_score_no_violation:",f1_score_no_violation)
    print("f1_score_violation_type_1:",f1_score_violation_type_1)


    print("f2_score_no_violation:",f2_score_no_violation)
    print("f2_score_violation_type_1:",f2_score_violation_type_1)


    fscore_half_micro = metrics.fbeta_score(Y_true, Y_pred, beta=0.5, average='micro')
    fscore_1_micro = metrics.fbeta_score(Y_true, Y_pred, beta=1, average='micro')
    fscore_2_micro = metrics.fbeta_score(Y_true, Y_pred, beta=2, average='micro')
    fscore_half_macro = metrics.fbeta_score(Y_true, Y_pred, beta=0.5, average='macro')
    fscore_1_macro = metrics.fbeta_score(Y_true, Y_pred, beta=1, average='macro')
    fscore_2_macro = metrics.fbeta_score(Y_true, Y_pred, beta=2, average='macro')
    fscore_half_weighted = metrics.fbeta_score(Y_true, Y_pred, beta=0.5, average='weighted')
    fscore_1_weighted = metrics.fbeta_score(Y_true, Y_pred, beta=1, average='weighted')
    fscore_2_weighted = metrics.fbeta_score(Y_true, Y_pred, beta=2, average='weighted')

    mydict = OrderedDict()
    mydict["seqOfRun"] = seqOfRun
    mydict["clf"] = clfName
    mydict["method"] = imbalanceTechnique + "_" + featureTrans + "_" + featureSelection
    mydict["accuracy|F0.5micro|F1micro|F2micro|Pmicro|Rmicro|Rweighted"] = float(accuracy)
    mydict["precision_no_violation"] = float(precision_no_violation)
    mydict["precision_violation_type_1"] = float(precision_violation_type_1)

    mydict["precision_macro"] = float(precision_macro)

    mydict["precision_weighted"] = float(precision_weighted)

    mydict["recall_no_violation"] = float(recall_no_violation)
    mydict["recall_violation_type_1"] = float(recall_violation_type_1)
    mydict["recall_macro"] = float(recall_macro)
 


    mydict["f0.5_score_no_violation"] = float(f_half_score_no_violation)
    mydict["f0.5_score_violation_type_1"] = float(f_half_score_violation_type_1)

    mydict["f1_score_no_violation"] = float(f1_score_no_violation)
    mydict["f1_score_violation_type_1"] = float(f1_score_violation_type_1)

    mydict["f2_score_no_violation"] = float(f2_score_no_violation)
    mydict["f2_score_violation_type_1"] = float(f2_score_violation_type_1)

    mydict["fscore_half_macro"] = float(fscore_half_macro)
    mydict["fscore_half_weighted"] = float(fscore_half_weighted)

    mydict["fscore_1_macro"] = float(fscore_1_macro)
    mydict["fscore_1_weighted"] = float(fscore_1_weighted)

    mydict["fscore_2_macro"] = float(fscore_2_macro)
    mydict["fscore_2_weighted"] = float(fscore_2_weighted)

    mydict["auc_area_pred_No_Violation"] = float(auc_area_Y_pred)
    mydict["auc_area_pred_violation_type_1"] = float(auc_area_Y_pred)

    mydict["pr_area_pred_No_Violation"] = float(pr_area_Y_pred)
    mydict["pr_area_pred_violation_type_1"] = float(pr_area_Y_pred)


    mydict["penalty_0"] = float(penalty_0)
    mydict["profit_0"] = float(profit_0)
    mydict["penalty_1"] = float(penalty_1)
    mydict["profit_1"] = float(profit_1)
    mydict["penalty_2"] = float(penalty_2)
    mydict["profit_2"] = float(profit_2)
    mydict["penalty_3"] = float(penalty_3)
    mydict["profit_3"] = float(profit_3)
    mydict["penalty_4"] = float(penalty_4)
    mydict["profit_4"] = float(profit_4)
    mydict["penalty_5"] = float(penalty_5)
    mydict["profit_5"] = float(profit_5)
    mydict["penalty_6"] = float(penalty_6)
    mydict["profit_6"] = float(profit_6)
    mydict["penalty_7"] = float(penalty_7)
    mydict["profit_7"] = float(profit_7)
    mydict["penalty_8"] = float(penalty_8)
    mydict["profit_8"] = float(profit_8)

    return mydict


def create_model():
    # create model
    model = Sequential()
    model.add(Dense(3, input_dim=10, activation='relu'))
    #model.add(Dense(3, activation='relu'))
    model.add(Dense(3, activation='sigmoid'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


def calAllMetrics2(imbalanceTechnique, featureTrans, featureSelection, clfName, accuracy_statics,
                   precision_macro_statics, precision_micro_statics, precision_weighted_statics, recall_macro_statics,
                   recall_micro_statics,
                   recall_weighted_statics, f1_macro_statics, f1_micro_statics, f1_weighted_statics, f2_macro_statics,
                   f2_micro_statics, f2_weighted_statics):
    mydict = OrderedDict()
    mydict["clf"] = clfName
    mydict["method"] = imbalanceTechnique + "_" + featureTrans + "_" + featureSelection
    mydict["accuracy"] = (accuracy_statics)
    '''
    mydict["precision_no_violation"] = (precision_no_violation_statics)
    mydict["precision_violation_type_1"] = (precision_violation_type_1_statics)
    mydict["precision_violation_type_2"] = (precision_violation_type_2_statics)

    mydict["recall_no_violation"] = (recall_no_violation_statics)
    mydict["recall_violation_type_1"] = (recall_violation_type_1_statics)
    mydict["recall_violation_type_2"] = (recall_violation_type_2_statics)

    '''
    mydict["precision_macro"] = (precision_macro_statics)
    mydict["precision_micro"] = (precision_micro_statics)
    mydict["precision_weighted"] = (precision_weighted_statics)

    mydict["recall_macro"] = (recall_macro_statics)
    mydict["recall_micro"] = (recall_micro_statics)
    mydict["recall_weighted"] = (recall_weighted_statics)

    '''
    mydict["f1_no_violation"] = (f1_no_violation_statics)
    mydict["f1_violation_type_1"] = (f1_violation_type_1_statics)
    mydict["f1_violation_type_2"] = (f1_violation_type_2_statics)

    mydict["f2_no_violation"] = (f2_no_violation_statics)
    mydict["f2_violation_type_1"] = (f2_violation_type_1_statics)
    mydict["f2_violation_type_2"] = (f2_violation_type_2_statics)
    '''

    mydict["f1_macro"] = (f1_macro_statics)
    mydict["f1_micro"] = (f1_micro_statics)
    mydict["f1_weighted"] = (f1_weighted_statics)

    mydict["f2_macro"] = (f2_macro_statics)
    mydict["f2_micro"] = (f2_micro_statics)
    mydict["f2_weighted"] = (f2_weighted_statics)

    return mydict


def plotPRcurve(seqOfRun, imbalanceTechnique, featureTrans, featureSelection, clfName, y_one_hot, y_score, tag):

    # For each class
    plt.clf()
    precision = dict()
    recall = dict()
    average_precision = dict()
    precision, recall, _ = precision_recall_curve(y_one_hot,y_score)
    average_precision = average_precision_score(y_one_hot, y_score)

    precision, recall, _ = precision_recall_curve(y_one_hot, y_score)

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    #fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=FONTSIZE)
    plt.ylabel('Precision', fontsize=FONTSIZE)
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
        average_precision),fontsize=FONTSIZE)


    # plt.show()
    myfig = plt.gcf()
    # myfig.set_size_inches(INCHWIDTH,INCHHEIGHT)
    myfig.set_figwidth(FIGUREWIDTH)
    myfig.set_figheight(FIGUREHEIGHT)

    if (tag == 1):
        myfig.savefig(
            path + "/figures/PR_[Y_data, Y_pred]_" + str(
                seqOfRun) + "_" + imbalanceTechnique + "_" + featureTrans + "_" + featureSelection + "_" + clfName + ".jpg",
            dpi=DPI, format="jpg", bbox_inches='tight')
    elif (tag == 2):
        myfig.savefig(
            path + "/figures/PR_[Y_data, Y_pred2_encoding]_" + str(
                seqOfRun) + "_" + imbalanceTechnique + "_" + featureTrans + "_" + featureSelection + "_" + clfName + ".jpg",
            dpi=DPI, format="jpg", bbox_inches='tight')
    elif (tag == 3):
        myfig.savefig(
            path + "/figures/PR_[Y_data, Y_pred_proba]_" + str(
                seqOfRun) + "_" + imbalanceTechnique + "_" + featureTrans + "_" + featureSelection + "_" + clfName + ".jpg",
            dpi=DPI, format="jpg", bbox_inches='tight')
    return average_precision


def run(imbalanceTechnique, featureTrans, featureSelection, clfName, data_all, label_all,x,z):
    print("imbalanceTechnique:", imbalanceTechnique)
    print("featureTrans:", featureTrans)
    print("featureSelection:", featureSelection)
    print("clfName:", clfName)

    if (imbalanceTechnique == 'RandomUnderSampler'):
        imbalanceHandler = RandomUnderSampler(sampling_strategy='auto',random_state=SEED)
    elif (imbalanceTechnique == 'ClusterCentroids'):
        imbalanceHandler = ClusterCentroids(sampling_strategy='auto',random_state=SEED)
    elif (imbalanceTechnique == 'NearMiss1'):
        imbalanceHandler = NearMiss(sampling_strategy='auto',version=1, random_state=SEED, n_jobs=-1)
    elif (imbalanceTechnique == 'NearMiss2'):
        imbalanceHandler = NearMiss(sampling_strategy='auto',version=2, random_state=SEED, n_jobs=-1)
    elif (imbalanceTechnique == 'NearMiss3'):
        imbalanceHandler = NearMiss(sampling_strategy='auto',version=3, random_state=SEED, n_jobs=-1)
    elif (imbalanceTechnique == 'OnesidedSelcection'):
        imbalanceHandler = OneSidedSelection(sampling_strategy='auto', random_state=SEED, n_jobs=-1)
    elif (imbalanceTechnique == 'RandomOverSampler'):
        imbalanceHandler = RandomOverSampler(random_state=SEED)
    elif (imbalanceTechnique == 'SMOTE_regular'):
        imbalanceHandler = SMOTE(random_state=SEED, ratio='minority', kind='regular', k_neighbors=5, n_jobs=NUM_JOBS)
    elif (imbalanceTechnique == 'SMOTE_svm'):
        imbalanceHandler = SMOTE(ratio='minority', random_state=SEED, kind="svm")
    elif (imbalanceTechnique == 'SMOTE_borderline1'):
        imbalanceHandler = SMOTE(ratio='minority', random_state=SEED, kind="borderline1")
    elif (imbalanceTechnique == 'SMOTE_borderline2'):
        imbalanceHandler = SMOTE(ratio='minority', random_state=SEED, kind="borderline2")
    elif (imbalanceTechnique == 'SMOTE_ENN'):
        imbalanceHandler = SMOTEENN(ratio='minority', random_state=SEED)
    elif (imbalanceTechnique == 'SMOTE_Tomek'):
        smoteObject = SMOTE(random_state=SEED, ratio='minority', kind='regular', k_neighbors=5, n_jobs=NUM_JOBS)
        tomekObject = TomekLinks(random_state=SEED, ratio='auto', n_jobs=NUM_JOBS)
        imbalanceHandler = SMOTETomek(random_state=SEED, ratio='minority', smote=smoteObject, tomek=tomekObject)
    elif(imbalanceTechnique == "Neighborhood_Cleaning"):
        imbalanceHandler = NeighbourhoodCleaningRule(sampling_strategy= "auto",n_jobs=-1, random_state= SEED)
    elif(imbalanceTechnique == "ADASYN"):
        imbalanceHandler = ADASYN(ratio='minority',random_state=SEED)
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
        scaler = preprocessing.StandardScaler()
    elif (featureTrans == "MinMaxScaler"):
        # Feature Transformation 2: transforms features by scaling each feature to a [0,1] range.
        scaler = preprocessing.MinMaxScaler()
    elif (featureTrans == "Normalizer"):
        scaler = preprocessing.Normalizer().fit(X_data)
    elif (featureTrans == "No_FeatureTrans"):
        scaler = None

    if (scaler != None):
        pass
    else:
        X_data = X_data

    if (featureSelection == "RFSelect"):
        clf = RandomForestClassifier(random_state=SEED)
        clf = clf.fit(X_data, Y_data)
        print(clf.feature_importances_)
        model = SelectFromModel(clf, prefit=True)
        X_new = model.transform(X_data)
        print(X_new.shape)  # reduced data set number of rows and columns
        X_data = X_new
    elif (featureSelection == "SelectKBest"):
        model = SelectKBest(chi2, k=2)
        X_new = model.fit_transform(X_data, Y_data)
        print(X_new.shape)
        X_data = X_new
    elif (featureSelection == "SVR"):
        estimator = SVR(kernel="linear")
        selector = RFECV(estimator, step=1, cv=5)
        selector = selector.fit(X_data, Y_data)
        X_new = selector.transform(X_data)
        print(X_new.shape)
        X_data = X_new
    elif (featureSelection == "No_FeatureSelection"):
        X_data = X_data

    parameters_grid_OneVsRest = {
        "clf__estimator__n_estimators": [100, 120, 150, 200],  # The number of trees in the forest.
        # "estimator__n_estimators": [10],  # The number of trees in the forest.
        "clf__estimator__max_depth": [18],  # maximum depth of each tree
        # "estimator__max_depth": [2],  # maximum depth of each tree
        "clf__estimator__max_features": [4],  # max features per random selection
        # "estimator__max_features": [2],  # max features per random selection
        # "estimator__max_leaf_nodes":[5,8,10],  # max leaf nodes per tree. minimum 1
        # "estimator__max_leaf_nodes": [2],  # max leaf nodes per tree. minimum 1
        # "estimator__min_samples_leaf":[2,5,7,10], # min # samples per leaf
        # "estimator__min_samples_leaf": [2],  # min # samples per leaf
    }

    parameters_grid_pipe = {
        "estimator__n_estimators": [80, 100],  # The number of trees in the forest.
        # "estimator__n_estimators": [10],  # The number of trees in the forest.
        "estimator__max_depth": [14, 16],  # maximum depth of each tree
        # "estimator__max_depth": [2],  # maximum depth of each tree
        "estimator__max_features": [3, 4],  # max features per random selection
        # "estimator__max_features": [2],  # max features per random selection
        # "estimator__max_leaf_nodes":[5,8,10],  # max leaf nodes per tree. minimum 1
        # "estimator__max_leaf_nodes": [2],  # max leaf nodes per tree. minimum 1
        # "estimator__min_samples_leaf":[2,5,7,10], # min # samples per leaf
        # "estimator__min_samples_leaf": [2],  # min # samples per leaf
    }

    parameters_grid_PureCLF = {
        "n_estimators": [50, 60, 80, 100],  # The number of trees in the forest.
        "max_depth": [8, 12, 14, 16],  # maximum depth of each tree
        # "min_samples_leaf": [2,5,7,10],  # min # samples per leaf
        "max_leaf_nodes": [5, 8, 10],
        "max_features": [2, 3, 4]
    }

    if(clfName == "best"):
        pass

    if (clfName == "guess"):
        pass

    elif (clfName == "OneVsRestRF"):

        clf = OneVsRestClassifier(RandomForestClassifier(random_state=SEED, n_jobs=-1))
        parameters_grid = parameters_grid_OneVsRest
        if (imbalanceTechnique == 'No_Resampling'):
            pipeline = imbPipeline([
                ('scaler', scaler),
                ('clf', clf)
            ])
        else:
            pipeline = imbPipeline([
                ('scaler', scaler),
                ('samplingMethod', imbalanceHandler),
                ('clf', clf)
            ])

    elif (clfName == "OneVsRestLR"):
        clf = OneVsRestClassifier(LogisticRegression())
        pipeline = imbPipeline([
            ('samplingMethod', imbalanceHandler),
            ('clf', clf)

        ])

    elif(clfName == "GradientBoostingClassifier"):
        clf = GradientBoostingClassifier(random_state=SEED)
        pipeline = imbPipeline([
            ('samplingMethod', imbalanceHandler),
            ('clf', clf)

        ])
    elif(clfName == "SVC"):
        clf = SVC(gamma='auto')
        pipeline = imbPipeline([
            ('scaler',scaler),
            ('samplingMethod', imbalanceHandler),
            ('clf', clf)

        ])
    elif (clfName == "OneVsRestXGB"):
        clf = OneVsRestClassifier(xgb.XGBClassifier())
        pipeline = imbPipeline([
            ('clf', clf)

        ])
    elif (clfName == "PureRF"):
        clf = RandomForestClassifier(random_state=SEED)
        parameters_grid = parameters_grid_PureCLF
        pipeline = imbPipeline([
            ('samplingMethod', imbalanceHandler),
            ('clf', clf)
        ])
    elif (clfName == "KerasNN"):
        clf = KerasClassifier(build_fn=create_model, verbose=0, epochs=150, batch_size=100)
        pipeline = imbPipeline([
            ('samplingMethod', imbalanceHandler),
            # ('scaler',scaler),
            ('clf', clf)
        ])

    else:
        pass

    data_all_np = np.array(data_all)
    label_all_np = np.array(label_all)

    print("label_all_np:", label_all_np)
    print("shape of label_all_np:", label_all_np.shape)
    grid_search_best_scores = np.zeros(NUM_RUNS)  # numpy arrays
    final_evaluation_scores = np.zeros(NUM_RUNS)  # numpy arrays

    # folds_for_grid_search = StratifiedKFold(n_splits=SPLITS, shuffle=True, random_state=SEED)


    for each in range(NUM_RUNS):
        folds_for_grid_search = StratifiedKFold(n_splits=SPLITS, shuffle=True, random_state=each)

        accuracy = []

        precision_no_violation_list = []
        precision_violation_type_1_list = []
        precision_violation_type_2_list = []

        recall_no_violation_list = []
        recall_violation_type_1_list = []
        recall_violation_type_2_list = []

        precision_macro = []
        precision_micro = []
        precision_weighted = []

        recall_macro = []
        recall_micro = []
        recall_weighted = []

        f_half_no_violation_list = []
        f_half_violation_type_1_list = []
        f_half_violation_type_2_list = []


        f1_no_violation_list = []
        f1_violation_type_1_list = []
        f1_violation_type_2_list = []

        f2_no_violation_list = []
        f2_violation_type_1_list = []
        f2_violation_type_2_list = []

        f_half_macro = []
        f_half_micro = []
        f_half_weighted = []

        f1_macro = []
        f1_micro = []
        f1_weighted = []

        f2_macro = []
        f2_micro = []
        f2_weighted = []

        auc_no_violation_list = []
        auc_violation_type_1_list = []
        auc_violation_type_2_list = []

        pr_no_violation_list = []
        pr_violation_type_1_list = []
        pr_violation_type_2_list = []


        penalty_0_list = []
        penalty_1_list = []
        penalty_2_list = []
        penalty_3_list = []
        penalty_4_list = []
        penalty_5_list = []
        penalty_6_list = []
        penalty_7_list = []
        penalty_8_list = []

        profit_0_list = []
        profit_1_list = []
        profit_2_list = []
        profit_3_list = []
        profit_4_list = []
        profit_5_list = []
        profit_6_list = []
        profit_7_list = []
        profit_8_list = []
        F1_list = []
        F2_list = []


        i = 0
        for trainindex, testindex in folds_for_grid_search.split(data_all_np, label_all_np):
            print("type of trainindex:", type(trainindex))
            print("length of trainindex:", trainindex.shape)
            print("length of testindex:", testindex.shape)

            X_train = data_all_np[trainindex]
            Y_train = label_all_np[trainindex]
            X_test = data_all_np[testindex]
            Y_test = label_all_np[testindex]




            if(clfName == "best"):
                Y_test_pred = Y_test
            elif(clfName == "guess"):
                Y_test_pred = np.zeros(Y_test.shape)

            elif (clfName == "guess2"):
                Y_test_pred = np.zeros(Y_test.shape)+1
            else:
                clf_fit = pipeline.fit(X_train, Y_train)
                tuned_model = clf_fit

                # clf_fit = tuned_model.fit(X_train, Y_train)
                print("type of clf_fit:", type(clf_fit))
                # print("type of tuned_model:", type(tuned_model))
                Y_test_pred = clf_fit.predict(X_test)
                scores = clf_fit.score(X_test, Y_test)  #
                print("score:", scores)

            print(confusion_matrix(Y_test, Y_test_pred))
            print("----------------------------------")

            # generate classification report
            print()
            print("classification results _1_Y_test_pred: ")
            print(classification_report(Y_test, Y_test_pred, digits=6))
            print("----------------------------------")

            print('              NoVol      Type1        ')
            print('precision:', precision_score(Y_test, Y_test_pred, average=None))

            print('recall:   ', recall_score(Y_test, Y_test_pred, average=None))

            print('f1 score: ', f1_score(Y_test, Y_test_pred, average=None))
            print('f2 score: ', (metrics.fbeta_score(Y_test, Y_test_pred, beta=2, average=None)))
            print('-' * 50)

            Y_test_Encoding = onehot_encoder.fit_transform(Y_test.reshape((len(Y_test),1)))
            print("Y_test_Encoding:", Y_test_Encoding)
            Y_test_pred_Encoding = onehot_encoder.fit_transform(Y_test_pred.reshape((len(Y_test_pred),1)))
            print("Y_test_pred_Encoding:",Y_test_pred_Encoding)

            auc_area_Y_pred = plot_ROC(i, imbalanceTechnique, featureTrans, featureSelection, clfName, 1,
                                       Y_test, Y_test_pred)

            pr_area_Y_pred = plotPRcurve(i, imbalanceTechnique, featureTrans, featureSelection, clfName,Y_test, Y_test_pred, 1)

            temp = calAllMetrics(i, imbalanceTechnique, featureTrans, featureSelection, clfName, Y_test,
                                 Y_test_pred,auc_area_Y_pred,pr_area_Y_pred,x,z)

            acc = metrics.accuracy_score(Y_test, Y_test_pred)

            print("acc999:",acc)

            accuracy.append(acc * 100)

            F1_list.append(f1_score(Y_test, Y_test_pred))
            F2_list.append(metrics.fbeta_score(Y_test, Y_test_pred,beta = 2))

            precision_no_violation_list.append(temp["precision_no_violation"] * 100)
            precision_violation_type_1_list.append(temp["precision_violation_type_1"] * 100)

            recall_no_violation_list.append((temp["recall_no_violation"] * 100))
            recall_violation_type_1_list.append((temp["recall_violation_type_1"] * 100))

            precision_macro.append(precision_score(Y_test, Y_test_pred, average='macro') * 100)
            precision_micro.append(precision_score(Y_test, Y_test_pred, average='micro') * 100)
            precision_weighted.append(precision_score(Y_test, Y_test_pred, average='weighted') * 100)

            recall_macro.append(recall_score(Y_test, Y_test_pred, average='macro') * 100)
            recall_micro.append(recall_score(Y_test, Y_test_pred, average='micro') * 100)
            recall_weighted.append(recall_score(Y_test, Y_test_pred, average='weighted') * 100)

            f_half_no_violation_list.append(temp["f0.5_score_no_violation"] * 100)
            f_half_violation_type_1_list.append(temp["f0.5_score_violation_type_1"] * 100)


            f1_no_violation_list.append(temp["f1_score_no_violation"] * 100)
            f1_violation_type_1_list.append(temp["f1_score_violation_type_1"] * 100)

            f2_no_violation_list.append((temp["f2_score_no_violation"] * 100))
            f2_violation_type_1_list.append((temp["f2_score_violation_type_1"] * 100))


            auc_no_violation_list.append(temp["auc_area_pred_No_Violation"] *100)
            auc_violation_type_1_list.append(temp["auc_area_pred_violation_type_1"] *100)


            pr_no_violation_list.append(temp["pr_area_pred_No_Violation"] * 100)
            pr_violation_type_1_list.append(temp["pr_area_pred_violation_type_1"] * 100)



            f_half_macro.append(metrics.fbeta_score(Y_test, Y_test_pred, beta=0.5, average='macro') * 100)
            f_half_micro.append(metrics.fbeta_score(Y_test, Y_test_pred, beta=0.5, average='micro') * 100)
            f_half_weighted.append(metrics.fbeta_score(Y_test, Y_test_pred, beta=0.5, average='weighted') * 100)

            f1_macro.append(f1_score(Y_test, Y_test_pred, average='macro') * 100)
            f1_micro.append(f1_score(Y_test, Y_test_pred, average='micro') * 100)
            f1_weighted.append(f1_score(Y_test, Y_test_pred, average='weighted') * 100)

            f2_macro.append(metrics.fbeta_score(Y_test, Y_test_pred, beta=2, average='macro') * 100)
            f2_micro.append(metrics.fbeta_score(Y_test, Y_test_pred, beta=2, average='micro') * 100)
            f2_weighted.append(metrics.fbeta_score(Y_test, Y_test_pred, beta=2, average='weighted') * 100)

            penalty_0_list.append(temp["penalty_0"])
            profit_0_list.append(temp["profit_0"])

            penalty_1_list.append(temp["penalty_1"])
            profit_1_list.append(temp["profit_1"])

            penalty_2_list.append(temp["penalty_2"])
            profit_2_list.append(temp["profit_2"])

            penalty_3_list.append(temp["penalty_3"])
            profit_3_list.append(temp["profit_3"])

            penalty_4_list.append(temp["penalty_4"])
            profit_4_list.append(temp["profit_4"])


            penalty_5_list.append(temp["penalty_5"])
            profit_5_list.append(temp["profit_5"])


            penalty_6_list.append(temp["penalty_6"])
            profit_6_list.append(temp["profit_6"])


            penalty_7_list.append(temp["penalty_7"])
            profit_7_list.append(temp["profit_7"])

            penalty_8_list.append(temp["penalty_8"])
            profit_8_list.append(temp["profit_8"])



            print("accuracy:", accuracy)
            print("precision_macro:", precision_macro)
            print("precision_micro:", precision_micro)
            print("precision_weighted:", precision_weighted)
            print("recall_macro:", recall_macro)
            print("recall_micro:", recall_micro)
            print("recall_weighted:", recall_weighted)
            print("f1_macro:", f1_macro)
            print("f1_micro:", f1_micro)
            print("f1_weighted:", f1_weighted)
            print("f2_macro:", f2_macro)
            print("f2_micro:", f2_micro)
            print("f2_weighted:", f2_weighted)

            cm1 = confusion_matrix(Y_test, Y_test_pred)

            plot_confusion_matrix(i, imbalanceTechnique, featureTrans, featureSelection, clfName, cm1, 1,
                                  target_names=TARGET_NAMES, normalize=False)

            plot_classification_report(i, imbalanceTechnique, featureTrans, featureSelection, clfName, 1,
                                       classification_report(Y_test, Y_test_pred, digits=5,
                                                             target_names=TARGET_NAMES))
            result = []
            result.append(temp)
            temp_result_df = pd.DataFrame(result)
            temp_result_df.to_csv(OUTPUT_PATH, mode='a', index=False, header=False)
            i = i + 1

        print("accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(accuracy,dtype=np.float16), np.std(accuracy,dtype=np.float16)))
        print("precision_macro: %.2f%% (+/- %.2f%%)" % (np.mean(precision_macro,dtype=np.float16), np.std(precision_macro,dtype=np.float16)))
        print("precision_micro: %.2f%% (+/- %.2f%%)" % (np.mean(precision_micro,dtype=np.float16), np.std(precision_micro,dtype=np.float16)))
        print("precision_weighted: %.2f%% (+/- %.2f%%)" % (np.mean(precision_weighted,dtype=np.float16), np.std(precision_weighted,dtype=np.float16)))
        print("recall_macro: %.2f%% (+/- %.2f%%)" % (np.mean(recall_macro,dtype=np.float16), np.std(recall_macro,dtype=np.float16)))
        print("recall_micro: %.2f%% (+/- %.2f%%)" % (np.mean(recall_micro,dtype=np.float16), np.std(recall_micro,dtype=np.float16)))
        print("recall_weighted: %.2f%% (+/- %.2f%%)" % (np.mean(recall_weighted,dtype=np.float16), np.std(recall_weighted,dtype=np.float16)))

        print("f1_macro score: %.2f%% (+/- %.2f%%)" % (np.mean(f1_macro,dtype=np.float16), np.std(f1_macro,dtype=np.float16)))
        print("f1_micro score: %.2f%% (+/- %.2f%%)" % (np.mean(f1_micro,dtype=np.float16), np.std(f1_micro,dtype=np.float16)))
        print("f1_weighted score: %.2f%% (+/- %.2f%%)" % (np.mean(f1_weighted,dtype=np.float16), np.std(f1_weighted,dtype=np.float16)))
        print("f2_macro score: %.2f%% (+/- %.2f%%)" % (np.mean(f2_macro,dtype=np.float16), np.std(f2_macro,dtype=np.float16)))
        print("f2_micro score: %.2f%% (+/- %.2f%%)" % (np.mean(f2_micro,dtype=np.float16), np.std(f2_micro,dtype=np.float16)))
        print("f2_weighted score: %.2f%% (+/- %.2f%%)" % (np.mean(f2_weighted,dtype=np.float16), np.std(f2_weighted,dtype=np.float16)))


        accuracy_statics = [np.mean(accuracy), np.std(accuracy)]

        precision_no_violation_statics = [np.mean(precision_no_violation_list), np.std(precision_no_violation_list)]
        precision_violation_type_1_statics = [np.mean(precision_violation_type_1_list),
                                              np.std(precision_violation_type_1_list)]

        recall_no_violation_statics = [np.mean(recall_no_violation_list), np.std(recall_no_violation_list)]
        recall_violation_type_1_statics = [np.mean(recall_violation_type_1_list), np.std(recall_violation_type_1_list)]

        precision_macro_statics = [np.mean(precision_macro), np.std(precision_macro)]
        precision_micro_statics = [np.mean(precision_micro), np.std(precision_micro)]
        precision_weighted_statics = [np.mean(precision_weighted), np.std(precision_weighted)]

        recall_macro_statics = [np.mean(recall_macro), np.std(recall_macro)]
        recall_micro_statics = [np.mean(recall_micro), np.std(recall_micro)]
        recall_weighted_statics = [np.mean(recall_weighted), np.std(recall_weighted)]

        f_half_no_violation_statics = [np.mean(f_half_no_violation_list), np.std(f_half_no_violation_list)]
        f_half_violation_type_1_statics = [np.mean(f_half_violation_type_1_list), np.std(f_half_violation_type_1_list)]

        f1_no_violation_statics = [np.mean(f1_no_violation_list), np.std(f1_no_violation_list)]
        f1_violation_type_1_statics = [np.mean(f1_violation_type_1_list), np.std(f1_violation_type_1_list)]

        f2_no_violation_statics = [np.mean(f2_no_violation_list), np.std(f2_no_violation_list)]
        f2_violation_type_1_statics = [np.mean(f2_violation_type_1_list), np.std(f2_violation_type_1_list)]

        f_half_macro_statics = [np.mean(f_half_macro), np.std(f_half_macro)]
        f_half_micro_statics = [np.mean(f_half_micro), np.std(f_half_micro)]
        f_half_weighted_statics = [np.mean(f_half_weighted), np.std(f_half_weighted)]


        f1_macro_statics = [np.mean(f1_macro), np.std(f1_macro)]
        f1_micro_statics = [np.mean(f1_micro), np.std(f1_micro)]
        f1_weighted_statics = [np.mean(f1_weighted), np.std(f1_weighted)]

        f2_macro_statics = [np.mean(f2_macro), np.std(f2_macro)]
        f2_micro_statics = [np.mean(f2_micro), np.std(f2_micro)]
        f2_weighted_statics = [np.mean(f2_weighted), np.std(f2_weighted)]


        auc_no_violation_statics = [np.mean(auc_no_violation_list), np.std(auc_no_violation_list)]
        auc_violation_type_1_statics = [np.mean(auc_violation_type_1_list), np.std(auc_violation_type_1_list)]


        pr_no_violation_statics = [np.mean(pr_no_violation_list), np.std(pr_no_violation_list)]
        pr_violation_type_1_statics = [np.mean(pr_violation_type_1_list), np.std(pr_violation_type_1_list)]


        penalty_0_statics = [np.mean(penalty_0_list), np.std(penalty_0_list)]
        profit_0_statics = [np.mean(profit_0_list), np.std(profit_0_list)]

        penalty_1_statics = [np.mean(penalty_1_list), np.std(penalty_1_list)]
        profit_1_statics = [np.mean(profit_1_list), np.std(profit_1_list)]



        penalty_2_statics = [np.mean(penalty_2_list), np.std(penalty_2_list)]
        profit_2_statics = [np.mean(profit_2_list), np.std(profit_2_list)]



        penalty_3_statics = [np.mean(penalty_3_list), np.std(penalty_3_list)]
        profit_3_statics = [np.mean(profit_3_list), np.std(profit_3_list)]


        penalty_4_statics = [np.mean(penalty_4_list), np.std(penalty_4_list)]
        profit_4_statics = [np.mean(profit_4_list), np.std(profit_4_list)]


        penalty_5_statics = [np.mean(penalty_5_list), np.std(penalty_5_list)]
        profit_5_statics = [np.mean(profit_5_list), np.std(profit_5_list)]


        penalty_6_statics = [np.mean(penalty_6_list), np.std(penalty_6_list)]
        profit_6_statics = [np.mean(profit_6_list), np.std(profit_6_list)]


        penalty_7_statics = [np.mean(penalty_7_list), np.std(penalty_7_list)]
        profit_7_statics = [np.mean(profit_7_list), np.std(profit_7_list)]


        penalty_8_statics = [np.mean(penalty_8_list), np.std(penalty_8_list)]
        profit_8_statics = [np.mean(profit_8_list), np.std(profit_8_list)]

        F1_statics = [np.mean(F1_list), np.std(F1_list)]
        F2_statics = [np.mean(F2_list), np.std(F2_list)]

        mydict = OrderedDict()
        mydict["clf"] = clfName
        mydict["method"] = imbalanceTechnique + "_" + featureTrans + "_" + featureSelection
        mydict["accuracy|F0.5micro|F1micro|F2micro|Pmicro|Rmicro|Rweighted"] = (accuracy_statics)

        mydict["precision_no_violation"] = (precision_no_violation_statics)
        mydict["precision_violation_type_1"] = (precision_violation_type_1_statics)

        mydict["precision_macro"] = (precision_macro_statics)
        # mydict["precision_micro"] = (precision_micro_statics)
        mydict["precision_weighted"] = (precision_weighted_statics)


        mydict["recall_no_violation"] = (recall_no_violation_statics)
        mydict["recall_violation_type_1"] = (recall_violation_type_1_statics)



        mydict["recall_macro"] = (recall_macro_statics)

        mydict["f0.5_no_violation"] = (f_half_no_violation_statics)
        mydict["f0.5_violation_type_1"] = (f_half_violation_type_1_statics)


        mydict["f1_no_violation"] = (f1_no_violation_statics)
        mydict["f1_violation_type_1"] = (f1_violation_type_1_statics)

        mydict["f2_no_violation"] = (f2_no_violation_statics)
        mydict["f2_violation_type_1"] = (f2_violation_type_1_statics)

        mydict["f0.5_macro"] = (f_half_macro_statics)
        mydict["f0.5_weighted"] = (f_half_weighted_statics)

        mydict["f1_macro"] = (f1_macro_statics)
        mydict["f1_weighted"] = (f1_weighted_statics)

        mydict["f2_macro"] = (f2_macro_statics)
        mydict["f2_weighted"] = (f2_weighted_statics)

        mydict["auc_No"] = (auc_no_violation_statics)
        mydict["auc_T1"] = (auc_violation_type_1_statics)

        mydict["pr_No"] = (pr_no_violation_statics)
        mydict["pr_T1"] = (pr_violation_type_1_statics)



        mydict["penalty0"] =  (penalty_0_statics)
        mydict["profit0"] =  (profit_0_statics)

        mydict["penalty1"] = (penalty_1_statics)
        mydict["profit1"] = (profit_1_statics)

        mydict["penalty2"] = (penalty_2_statics)
        mydict["profit2"] = (profit_2_statics)

        mydict["penalty3"] = (penalty_3_statics)
        mydict["profit3"] = (profit_3_statics)

        mydict["penalty4"] = (penalty_4_statics)
        mydict["profit4"] = (profit_4_statics)

        mydict["penalty5"] = (penalty_5_statics)
        mydict["profit5"] = (profit_5_statics)

        mydict["penalty6"] = (penalty_6_statics)
        mydict["profit6"] = (profit_6_statics)

        mydict["penalty7"] = (penalty_7_statics)
        mydict["profit7"] = (profit_7_statics)

        mydict["penalty8"] = (penalty_8_statics)
        mydict["profit8"] = (profit_8_statics)

        mydict["F1_Score"] = (F1_statics)
        mydict["F2_Score"] = (F2_statics)


        print("===========BEGIN ===========")
        print("OUTPUT_PATH_STATICS:",OUTPUT_PATH_STATICS)
        result2 = []
        result2.append(mydict)
        temp_result_df2 = pd.DataFrame(result2)
        temp_result_df2.to_csv(OUTPUT_PATH_STATICS, mode='a', index=True, header=False)
        print("===========END===========")


resampingMethod = ["No_Resampling", "SMOTE_Tomek","SMOTE_ENN", "RandomOverSampler", "SMOTE_regular", "SMOTE_svm",
                   "SMOTE_borderline1", "SMOTE_borderline2", "RandomUnderSampler", "ClusterCentroids", "NearMiss1",
                   "NearMiss2", "NearMiss3", "OnesidedSelcection","Neighborhood_Cleaning","ADASYN"]
scalerMethod = ["StandardScaler","MinMaxScaler"]
featureSelMethod = ["No_FeatureSelection"]
clfMethod = ['guess2',"SVC","GradientBoostingClassifier","best","guess","OneVsRestRF","OneVsRestLR","OneVsRestXGB","KerasNN", "PureRF", "OneVsRestRFClassWeight5", "OneVsRestRFClassWeight10",
             "OneVsRestRFClassWeight50"]


scores_list = []


def cutoff_predict(clf, X_test, cutoff):
    return ((clf.predict_proba(X_test)[:,1]>cutoff)+0) 


def custom_f1(cutoff):
    def f1_cutoff(clf, X_test, y_test):
        y_test_pred = cutoff_predict(clf, X_test, cutoff)
        return metrics.f1_score(y_test, y_test_pred)
    return f1_cutoff


def writeHead(headlist, output):
    df = pd.DataFrame(headlist)
    df.to_csv(output,index=True, header=True)

if __name__ == '__main__':
    onehot_encoder = OneHotEncoder(sparse=False)
    x = [0.1,0.25,1,0.1,0.25,1,0.1,0.25,1]
    z = [0.1,0.1,0.1,0.2,0.2,0.2,0.3,0.3,0.3]
    starttime = datetime.datetime.now()

    filename = "sla99"
    data_all, label_all = prepareData(filename)
    column = OrderedDict(
        {"clf": "", "method": "", "acc|F0.5micro|F1micro|F2micro|Pmicro|Rmicro|Rweighted": "", "P_No": "", "P_T1": "",
         "P_macro": "", "P_weighted": "", "R_No": "", "R_T1": "",
         "R_macro": "", "F0.5_No": "", "F0.5_T1": "", "F1_No": "", "F1_T1": "", "F2_No": "", "F2_T1": "",
         "F0.5_macro": "", "F0.5_weighted": "", "F1_macro": "", "F1_weighted": "", "F2_macro": "", "F2_weighted": "",
         "auc_No": "", "auc_T1": "", "pr_No": "", "pr_T1": "", "penalty0": "", "profit0": "",
         "penalty1": "", "profit1": "", "penalty2": "",
         "profit2": "", "penalty3": "", "profit3": "", "penalty4": "", "profit4": "", "penalty5": "", "profit5": "",
         "penalty6": "", "profit6": "", "penalty7": "", "profit7": "", "penalty8": "", "profit8": "","F1_Score":"","F2_Score":""})

    column_list = []
    column_list.append(column)



    for clf in clfMethod[7:8]:
        OUTPUT_PATH = path + "/output/" + filename + "_" + clf + ".csv"
        OUTPUT_PATH_STATICS = path + "/output/" + filename + "_" + clf + "_statics.csv"
        if (os.path.exists(OUTPUT_PATH) == False):
            writeHead(column_list, OUTPUT_PATH)
        if(os.path.exists(OUTPUT_PATH_STATICS) == False):
            writeHead(column_list, OUTPUT_PATH_STATICS)

    for resampling in resampingMethod[0:2]:
        for scaler in scalerMethod[1:2]:
            for featureSel in featureSelMethod:
                for clf in clfMethod[7:8]:
                    run(resampling, scaler, featureSel, clf, data_all, label_all,x,z)

    endtime = datetime.datetime.now()

    print("consuming total time by minutes:", (endtime - starttime) / 3600)