import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#importing the raw data
rawData = pd.read_csv('data.csv')

#seperating label from data
infoData = pd.DataFrame()
infoData = rawData[["FLAG", "CONS_NO"]].copy()
data = rawData.drop(columns=["FLAG", "CONS_NO"])

#removing duplicate data
dropIndex = data[data.duplicated()].index
data = data.drop(dropIndex, axis=0)
infoData = infoData.drop(dropIndex, axis=0)

#removing instances with all zeros/NaN value
zeroIndex = data[(data.sum(axis=1) == 0)].index  
data = data.drop(zeroIndex, axis=0)
infoData = infoData.drop(zeroIndex, axis=0)

#reindexing columns according to datetime
data.columns = pd.to_datetime(data.columns)  
data = data.sort_index(axis=1)

#resetting the index
data.reset_index(inplace=True, drop=True)  
infoData.reset_index(inplace=True, drop=True)

#filling of missing values(using linear model)
data = data.interpolate(method='linear', limit=2,  
                        limit_direction='both', axis=0).fillna(0)

#detection and masking of outlier(using 3-sigma rule) 
z_scores = (data-data.mean())/ data.std()
data = data.where(z_scores <= 3, data.mean() + 3 * data.std(), axis=1)

#raw csv for visualizaton purpose
data.to_csv(r'visualization.csv', index=False, header=True)

#data scaled using min-max scaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
scaled_data = pd.DataFrame(scaled_data, columns=data.columns)

#finalising the data for further training and testing purpose
preprData = pd.concat([infoData, scaled_data], axis=1)
preprData.to_csv('preprocessedR.csv', index=False)
print("Preprocessing complete. Files saved as 'visualization.csv' and 'preprocessedR.csv'.")