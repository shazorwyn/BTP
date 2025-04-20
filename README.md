# Electricity-Fraud-Detection-BTP

The repository contains scripts for detection of electricity fraud. 
### File Description
1.) data.7z
data for the project
2.) preprocessing.py
clean and prepare the data for visualization and further training
3.) visualization.py
script for obtaining all the plots and insight for our data
4.) models.py
contains all the models applied to data and outputs results for each of them

## Follow the given steps to run this code:

1.) Import the project on [Google Colab](https://colab.research.google.com/)

```
!git clone https://github.com/shazorwyn/BTP
!mv /content/Electricity-Fraud-Detection/* ./
```

2.) Unzip the dataset

```
!7za e data.7z
```

3.) Run [preprocessing.py](preprocessing.py) to obtain the datasets for visualizations, training and testing.

```
!python preprocessing.py
```

4.) Uncomment the required the model from [models.py](models.py) and run it to get the required results

```
!python models.py
```
