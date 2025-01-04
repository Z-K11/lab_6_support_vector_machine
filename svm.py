import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# Imported libraries 
cancer_data_df = pd.read_csv("cell_samples.csv")
print(cancer_data_df.head())
# read the data from the csv file and save it into a pandas data frame
ax =cancer_data_df[cancer_data_df['Class']==4][0:50].plot(kind="scatter",x='Clump',y="UnifSize",color='Red',label='malignant')
cancer_data_df[cancer_data_df['Class']==2][0:50].plot(kind="scatter",x='Clump',y='UnifSize',color="Green",label='benign',ax=ax)
#Don't need to print the plot everytime commenting it out 
#plt.savefig("scatter.png")
print(cancer_data_df.dtypes)
#prints the name of all the columns and their data types 
cancer_data_df=cancer_data_df[pd.to_numeric(cancer_data_df['BareNuc'],errors='coerce').notnull()]
cancer_data_df['BareNuc']=cancer_data_df['BareNuc'].astype('int')
print(cancer_data_df.dtypes)
feature_cancer_data=cancer_data_df[['Clump','UnifSize','UnifShape','MargAdh','SingEpiSize','BareNuc','BlandChrom','NormNucl','Mit']]
#Create a feature data set on witch the model will train, only includes the stated columns in the new data set
X=np.asarray(feature_cancer_data)
print(X[0:5])
y=np.asarray(cancer_data_df['Class'])
print(y[0:5])
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)
print('Train Set',x_train.shape,y_train.shape)
print('Test Set',x_test.shape,y_test.shape)
from sklearn import svm
model = svm.SVC(kernel='rbf')
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print(predictions[0:5])
from sklearn.metrics import classification_report,confusion_matrix
import itertools
def plot_confusion_matrix(cm,classes,normalize=False,title='Confusion Matrix',cmap=plt.cm.Blues):
    if normalize:
        cm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
        print("Normalized Confusion Matrix")
    else:
        print("Confusion Matrix")
    print(cm)
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=45)
    plt.yticks(tick_marks,classes)
    fmt='.2f' if normalize else 'd'
    thresh = cm.max()/2
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,format(cm[i,j],fmt),horizontalalignment="center",color="white" if cm[i,j]>thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('Confusion_Matrix.png')

cnf_matrix =confusion_matrix(y_test,predictions ,labels=[2,4])
np.set_printoptions(precision=2)
print(classification_report(y_test,predictions))
plt.figure()
plot_confusion_matrix(cnf_matrix,classes=['Benign(2)','Malignant(4)'],normalize=False,title='Confusion Matrix')
from sklearn.metrics import f1_score
print(f1_score(y_test,predictions,average='weighted'))
from sklearn.metrics import jaccard_score
print('jaccard score ',jaccard_score(y_test,predictions,pos_label=2))

