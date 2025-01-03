import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# Imported libraries 
cancer_data_df = pd.read_csv("cell_samples.csv")
print(cancer_data_df.head())
# read the data from the csv file and save it into a pandas data frame