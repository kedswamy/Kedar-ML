######################################        Data preparation starts     ###################################################
"""1 -Import statements for all the packages"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import *
import matplotlib.pyplot as plt

"""2 - Import the Horse colic dataset.(URL1)"""

url1 = "https://archive.ics.uci.edu/ml/machine-learning-databases/horse-colic/horse-colic.data"
horseColic = pd.read_csv(url1, header=None,sep = '\s+')
columnNames = ['surgery','age','hospitalNo','rectalTemp','pulse','respRate','tempOfExtremity','peripheralPulse',
               'mucousMembranes','capRefillTime','painLevel','peristalsis','abdominalDistension','nasogastricTube',
               'nasogastricReflux','nasogastricRefluxPH','feces','abdomen','packedCellVol','totalProtein','abdomenCentesisApp',
               'abdomenCentesisProtein','outcome','surgicalLesion','typeofLesion1','typeofLesion2','typeofLesion3','cp_data']
horseColic.columns = columnNames

""" 3- Drop irrelevant colmns"""

horseColic.drop(['hospitalNo','typeofLesion1','typeofLesion2','typeofLesion3','cp_data'], axis=1,inplace =True)

""" 4 - Create a dictionary of column names & their types for further use"""

columnTypes = ['cate','cate','cont','cont','cont','cate','cate','cate',
               'cate','cate','cate','cate','cate','cate','cont','cate',
               'cate','cont','cont','cate','cont','cate','cate']
data_dictionary = {horseColic.columns[i]: columnTypes[i] for i in range(len(columnTypes))} 

"""5 - Force the  entire df to numeric , so that missing values are converted to Nans"""

for column in horseColic.columns:
    horseColic[column] = pd.to_numeric(horseColic[column], errors = 'coerce')

"""6 - Impute the missing categorical variables with the value which occurs most frequently in that column."""

mode1 = horseColic.mode(axis='rows', numeric_only=False, dropna=True)
for key in data_dictionary:
    if data_dictionary[key] == 'cate':
        horseColic[key] = horseColic[key].fillna(mode1[key].iloc[0])
###############################################################################################

""" 7 - Replace Nans with median of the column."""
for key in data_dictionary:
    if data_dictionary[key] == 'cont':
        horseColic[key] = horseColic[key].fillna(np.nanmedian(horseColic[key]))

""" 8 -Replace outliers , based on iqr, recursively with medians of the column, till no outliers remain."""
while(True):
    mastersum = 0;
    for key in data_dictionary:
        if key in ['rectalTemp','abdomenCentesisProtein','nasogastricRefluxPH']:# skip columns which are too close
            continue
        if data_dictionary[key] == 'cont':
            q1 = np.quantile(horseColic[key], 0.25)
            q3 = np.quantile(horseColic[key], 0.75) 
            iqr = q3-q1
            limitHi = q3 + 1.5*iqr 
            limitLo = q1 - 1.5*iqr
    #creates a boolean for values that are outside limits
            FlagBad =(horseColic.loc[:,key] < limitLo)|(horseColic.loc[:,key] > limitHi)
            mastersum = mastersum + sum(FlagBad)
    # FlagGood is the complement of FlagBad
            FlagGood = ~FlagBad
    # Replace outliers with the median of non-outliers
            horseColic.loc[FlagBad,key] = np.median(horseColic.loc[FlagGood,key])
    if mastersum == 0:
        break

"""9 - Normalize all the numeric columns, using standard score."""
k1 = np.mean(horseColic['respRate']) # Calculate mean & std dev as will be required later to denormalize data.
k2 = np.std(horseColic['respRate'])
cont_list = []
for key in data_dictionary:
    if data_dictionary[key] == 'cont':
        cont_list.append(key)
standardization_scale = StandardScaler().fit(horseColic.loc[:,cont_list])
horseColic.loc[:,cont_list] = standardization_scale.transform(horseColic.loc[:,cont_list])

"""10-consolidate the 14 categorical variables and reduce the number of types in each of them."""

varList =['outcome','outcome','abdomenCentesisApp','abdomenCentesisApp','abdomen',
          'abdomen','abdomen','feces','feces',	'nasogastricReflux','nasogastricReflux',
          'nasogastricTube','nasogastricTube','abdominalDistension','abdominalDistension',
          'peristalsis','peristalsis','capRefillTime','capRefillTime','painLevel','painLevel',
          'painLevel','mucousMembranes','mucousMembranes','mucousMembranes','mucousMembranes','peripheralPulse',
          'peripheralPulse','tempOfExtremity','tempOfExtremity','age','age','surgery','surgery','surgicalLesion','surgicalLesion']
valueList = [[1],	[2,3],	[1],	[2,3],	[1,2],	[3],	[4,5],	[1,2,3],	[4],	
             [1,3],	[2],	[1,2],	[3],	[1,2,3],	[4],	[1,2],	[3,4],	[1],	[2,3],
             [1,2],	[3],	[4,5],	[1,2],	[3],	[4,6],	[5],	[1,2],	[3,4],	[1,2],	[3,4],[1],[9],[1],[2],[1],[2]]

categoryList = ['lived','died','normal','abnormal','normal','mech_impact','surgLesion','normal','abnormal','normal','abnormal','normal',
'abnormal','normal','abnormal','normal','abnormal','normal','abnormal',	'low','moderate','severe','normal','early_shock',
'severe','septic','normal','abnormal','normal','abnormal','adult','young','Yes','No','Yes','No']

tuple1 = zip(varList,valueList, categoryList)

for var,value,category in tuple1:
    for item in value:
        Replace = horseColic.loc[:, var] == item
        horseColic.loc[Replace, var] = category

"""11 - Perform the one hot encoding for all the categorical columns."""

horseColic = pd.get_dummies(horseColic, columns=['surgery','age','tempOfExtremity','peripheralPulse',
               'mucousMembranes','capRefillTime','painLevel','peristalsis','abdominalDistension','nasogastricTube',
               'nasogastricReflux','feces','abdomen','abdomenCentesisApp','surgicalLesion','outcome'],drop_first=True)

######################################        Data preparation ends      ###################################################

"""
12 - We will use the standard function in sklearn to split the features( all columns except outcome_lived) 
and the target(outcome_lived).
Here we use a ratio of 0.2 for test-train data.
Also the random_state variable is set to 10 for reproducible results.
"""
X_train, X_test, y_train, y_test = train_test_split(horseColic.loc[:, horseColic.columns != 'outcome_lived'],horseColic['outcome_lived'],test_size=0.2, random_state=10)
threshold = 0.5 #Set Probability threshold to 0.5
"""13 - Next we train & apply the SVM classifier on the data
We also calculate & print Confusion matrix ,Accuracy , Precision,Sensitivity , recall and F1 score """

clf_svc = svm.SVC(gamma=0.1,probability=True)
clf_svc.fit(X_train, y_train)
predicted_SVC = clf_svc.predict_proba(X_test)
expertOutcomes_SVC = (predicted_SVC[:,0] < threshold).astype('int')
expertProb_SVC = np.where(y_test > 0,predicted_SVC[:,1],predicted_SVC[:,0])
Accuracy_SVC = np.sum(y_test == expertOutcomes_SVC).astype('int') /len(y_test)
print ("\n\nAccuracy forSVC:\n", Accuracy_SVC)
CM_SVC = confusion_matrix(expertOutcomes_SVC, y_test)
print ("\n\nConfusion matrix for SVC:\n", CM_SVC)
tn1, fp1, fn1, tp1 = CM_SVC.ravel()
print ("\nTP, TN, FP, FN for SVC:", tp1, ",", tn1, ",", fp1, ",", fn1)
P_SVC = tp1/(tp1 + fp1)
print ("\nPrecision for SVC:", np.round(P_SVC, 2))
R_SVC = tp1/(tp1 + fn1)
print ("\nRecall for SVC:", np.round(R_SVC, 2))
F1_SVC = tp1/(tp1 + 1/2*(fp1 + fn1))
print ("\nF1 score for SVC:", np.round(F1_SVC, 2))
fprSVC, tprSVC, thSVC = roc_curve(y_test, expertProb_SVC,drop_intermediate = False) # False Positive Rate, True Posisive Rate, probability thresholds
AUCSVC = auc(fprSVC, tprSVC)
print ("\nTP rates for SVC:", np.round(tprSVC, 2))
print ("\nFP rates for SVC:", np.round(fprSVC, 2))
print ("\nProbability thresholds for SVC:", np.round(thSVC, 2))

"""14 - Next we train & apply the Neural network based classifier on the data
We also calculate & print Confusion matrix ,Accuracy , Precision,Sensitivity , recall and F1 score """
clf_NN = MLPClassifier(random_state=1, max_iter=100)
clf_NN.fit(X_train, y_train)
predicted_NN = clf_NN.predict_proba(X_test)
expertOutcomes_NN = (predicted_NN[:,0] < threshold).astype('int')
expertProb_NN = np.where(y_test > 0,predicted_NN[:,1],predicted_NN[:,0])
Accuracy_NN = np.sum(y_test == expertOutcomes_NN).astype('int') /len(y_test)
print ("\n\nAccuracy for NN:\n", Accuracy_NN)
CM_NN = confusion_matrix(expertOutcomes_NN, y_test)
print ("\n\nConfusion matrix for NN:\n", CM_NN)
tn2, fp2, fn2, tp2 = CM_NN.ravel()
print ("\nTP, TN, FP, FN for NN:", tp2, ",", tn2, ",", fp2, ",", fn2)
P_NN = tp2/(tp2 + fp2)
print ("\nPrecision for NN:", np.round(P_NN, 2))
R_NN = tp2/(tp2 + fn2)
print ("\nRecall for NN:", np.round(R_NN, 2))
F1_NN = tp2/(tp2 + 1/2*(fp2 + fn2))
print ("\nF1 score for NN:", np.round(F1_NN, 2))
fprNN, tprNN, thNN = roc_curve(y_test, expertProb_NN,drop_intermediate = False) # False Positive Rate, True Posisive Rate, probability thresholds
AUCNN = auc(fprNN, tprNN)
print ("\nTP rates for SVC:", np.round(tprNN, 2))
print ("\nFP rates for SVC:", np.round(fprNN, 2))
print ("\nProbability thresholds for SVC:", np.round(thNN, 2))

""" 15 - In below section we plot the ROC curves for each of the methods & compare the results."""

LW = 1.5 # line width for plots
LL = "lower right" # legend location
LC = 'darkgreen' # Line Color

#####################
fig = plt.figure(figsize=(40,90))
(ax1, ax2) = plt.subplots(2)[1]
plt.tight_layout()
plt.subplots_adjust(top = 4,bottom = 2)
ax1.set_title('ROC curve for SVC')
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.0])
ax1.set_xlabel('FPR-SVC')
ax1.set_ylabel('TPR-SVC')
ax1.plot(fprSVC, tprSVC, color='black',lw=LW, label='ROC curve(area = %0.2f)' % AUCSVC)
ax1.plot([0, 1], [0, 1], color='Red', lw=LW, linestyle='--') # reference line for random classifier
ax1.legend(loc=LL)

ax2.set_title('ROC curve for NN')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.0])
ax2.set_xlabel('FPR-NN')
ax2.set_ylabel('TPR-NN')
ax2.plot(fprNN, tprNN, color=LC,lw=LW, label='ROC curve(area = %0.2f)' % AUCNN)
ax2.plot([0, 1], [0, 1], color='navy', lw=LW, linestyle='--') # reference line for random classifier
ax2.legend(loc=LL)

plt.show()
####################

""" 16 - In below section we bin the column resp_rate into 5 values & check the effect on the accuracy of SVM classifier.
"""
horsecolic_mod = horseColic.copy(deep = True)
horsecolic_mod['respRate'] = horsecolic_mod['respRate']*k2 + k1 # denormalize the column.
bins = np.array([16.0,25.0,32.0,40.0]) # 5 bins for the respRate variable
horsecolic_mod['respRate'] = np.digitize(horsecolic_mod['respRate'] , bins,right=True)# Bin using NP digitize
valList = sorted(horsecolic_mod['respRate'].unique()) # Binned values for resp rate
catList = ['verylow','low','medium','high','veryhigh']# decode numeric values into text values
t1 = zip(valList,catList)
for item,category in t1: # Replace the column with the categorical values.
    Replace = horsecolic_mod.loc[:, 'respRate'] == item
    horsecolic_mod.loc[Replace, 'respRate'] = category
horsecolic_mod = pd.get_dummies(horsecolic_mod, columns=['respRate'],drop_first=True) # perform one hot encoding

X_retrain, X_retest, y_retrain, y_retest = train_test_split(horsecolic_mod.loc[:, horsecolic_mod.columns != 'outcome_lived'],horsecolic_mod['outcome_lived'],test_size=0.2, random_state=10)
threshold_retest = 0.5 #Set Probability threshold to 0.5
"""17 - Next we retrain & apply the SVM classifier on the data
 """

clf2_svc = svm.SVC(gamma=0.1,probability=True)
clf2_svc.fit(X_retrain, y_retrain)
repredicted_SVC = clf2_svc.predict_proba(X_retest)
reexpertOutcomes_SVC = (repredicted_SVC[:,0] < threshold_retest).astype('int')
reexpertProb_SVC = np.where(y_retest > 0,repredicted_SVC[:,1],repredicted_SVC[:,0])
reAccuracy_SVC = np.sum(y_retest == reexpertOutcomes_SVC).astype('int') /len(y_retest)
print("Accuracy after binning resprate column is " ,reAccuracy_SVC)

"""
Conclusion 

1) We applied both the SVC & Neural network based classifier to the dataset.
2) The performance of both the classifiers seem to be identical with the SVC performing marginally better.
3) When applying the SVC it was found that a very low gamma value tends to overfitting & hence the gamma was changed from 0.0001 to 0.1 for optimum accuracy.
4) In the case of the Neural network classifier, the optimum value for max_iterations = 100 , and accuracy drops if the same is increased to 1000 or even 10000.
5) Even though we binned an additional column (respRate) and re-applied the SVM , the accuracy seemed to have reduced.)
"""