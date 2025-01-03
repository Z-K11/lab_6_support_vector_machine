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