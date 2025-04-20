# Electricity-Fraud-Detection-BTP

The repository contains scripts for detection of electricity fraud. 
### File Description
1.) data.7z
Data for the project.
2.) preprocessing.py
Clean and prepare the data for visualization and further training.
3.) visualization.py
Script for obtaining all the plots and insight for our data.
4.) models.py
Contains all the models applied to data and outputs results for each of them.

## Follow the given steps to run this code:


1.) Import the project on [Google Colab](https://colab.research.google.com/)
```
!git clone https://github.com/shazorwyn/BTP
!mv /content/BTP/* ./
```
2.) Download the dataset
```
# Install gdown if not already installed
!pip install -q gdown

# Download the 7z file from your shared Google Drive link
import gdown

file_id = "1VHNAizO_Jbym6tNe7D2g8x-OWhryZihk"
url = f"https://drive.google.com/uc?id={file_id}"
output = "data.7z"

gdown.download(url, output, quiet=False)

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
