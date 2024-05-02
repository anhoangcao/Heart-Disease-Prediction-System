# Heart Disease Prediction System

## Diagnosis is by signs of heart disease (module 1)

The app created with Python to predict person's heart health condition based on well-trained machine learning model (Voting classifier).

![App overview](https://imgur.com/Ay46Amh.png)

## Diagnosis of heart disease using ECG images (module 2)

The app created with Python to predict person's heart health condition based on well-trained machine learning model (2D-CNN).

![App overview](https://imgur.com/eG3p9xu.png)

## Diagnose heart disease using heartbeat sounds (module 3)

The app created with Python to predict person's heart health condition based on well-trained machine learning model (CNN-LSTM).

![App overview](https://imgur.com/JTQhEx8.png)

## Table of Contents
1. [General info](#general-info)
2. [Technologies](#technologies)
3. [Installation](#installation)


## General info
In this project, Voting classifier, 2D-CNN and CNN-LSTM was used to predict person's heart health condition expressed as a dichotomous variable (heart disease: yes/no). The model was trained on approximately 70,000 data from an annual telephone survey of the health of U.S. residents from the year 2020. The dataset is publicly available at the following link: https://www.cdc.gov/brfss/annual_data/annual_2020.html. The data is originally stored in SAS format. The original dataset contains approx. 400,000 rows and over 200 variables. The data conversion and cleaning process is described in another repository: https://github.com/anhoangcao/. This project contains:
* The App - the application structure is in the file `app_main.py` which includes `app_heart_key.py`, `app_heart_ecg.py` and `app_heart_sound.py`. This file uses data from the `data` directory and saved (previously trained) ML models from the `model` directory.

The Voting classifier, 2D-CNN and CNN-LSTM model was found to be satisfactorily accurate (accuracy approx. 80%).

## Technologies
The app is fully written in Python 3.9.9. `streamlit 1.5.1` was used to create the user interface, and the machine learning itself was designed using the module `scikit-learn 1.0.2`. `pandas 1.41.`, `numpy 1.22.2` and `polars 0.13.0` were used to perform data converting operations.

## Installation
The project was uploaded to the web using heroku. You can use it online at the following link: https://github.com/anhoangcao/Heart-Disease-Prediction-System/blob/main/app_main.py. If you want to use this app on your local machine, make sure that you have installed the necessary modules in a version no smaller than the one specified in the `requirements.txt` file. You can either install them globally on your machine or create a virtual environment (`pipenv`), which is highly recommended.
1. Setup mongodb: https://github.com/mongodb/mongo

2.  Install the packages according to the configuration file `requirements.txt`.
```
pip install -r requirements.txt
```

3.  Ensure that the `streamlit` package was installed successfully. To test it, run the following command:
```
streamlit hello
```
If the example application was launched in the browser tab, everything went well. You can also specify a port if the default doesn't respond:
```
streamlit hello --server.port port_number
```
Where `port_number` is a port number (8501, for example).

4.  To start the app, type:
```
streamlit run app_main.py
```

And that's it! Now you can predict your heart health condition expressed as a binary variable based on a dozen factors that best describe you.
