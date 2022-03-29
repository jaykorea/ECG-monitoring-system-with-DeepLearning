# DeepECG
ECG classification programs based on ML/DL methods. There are two datasets:
 - **training2017.zip** file contains one electrode voltage measurements taken as the difference between RA and LA electrodes with no ground. It is taken from The 2017 PhysioNet/CinC Challenge.
 - **MIT-BH.zip file** contains two electrode voltage measurements: MLII and V5.

## Prerequisites:
- Python 3.5 and higher
- Keras framework with TensorFlow backend
- Numpy, Scipy, Pandas libs
- Scikit-learn framework 

## Data
1) Extract the **training2017.zip** and **MIT-BH.zip** files into folders **training2017/** and **MIT-BH/** respectively

```
    make unzip
```

## Training 

1) add a folder.

```
    mkdir trained_models
```

2) run the training script.

```
    python train_conv1D.py
```

# Additional info
### Citation
If you use my repo - then, please, cite my paper. This is a BibTex citation:


    @article{pyakillya_kazachenko_mikhailovsky_2017,
        author = {Boris Pyakillya, Natasha Kazachenko, Nick Mikhailovsky},
        title = {Deep Learning for ECG Classification},
        journal = {Journal of Physics: Conference Series},
        year = {2017},
        volume = {913},
        pages = {1-5},
        DOI={10.1088/1742-6596/913/1/012004},
        url = {http://iopscience.iop.org/article/10.1088/1742-6596/913/1/012004/pdf}
    }


### For feature extraction and hearbeat rate calculation:
- https://github.com/PIA-Group/BioSPPy (Biosignal Processing in Python)

### Todo
- add some training script Argparser
- add data analysis scripts
- test module for the keras models