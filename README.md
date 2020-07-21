# Fingerprint-Restoration
Fingerprint Restoration using Cubic Bezier Curve

This project proposes a noval method of representing fingerprints using Bezier curves, which is conducive to the compression of fingerprint images. The most important thing is to achieve the repair of incomplete fingerprint images. The repaired images can improve the recognition success rate.  

## Installation and Dependencies:  
This project is based on python3.7 , and the python dependencies:
numpy, cv2=3.4.3.18 , scipy  

## Setup  
First, install the above dependencies.  
Second, git clone this project.  
Now, you're ready to start fingerprint restoration.   

## Start  

```
$ git clone git@github.com:tuyanglin/Fingerprint-Restoration.git
$ cd Code
$ cd preprocessing
$ python preprocess.py imagepath outputpath
$ cd ../reconstruction
$ python adjust_fitting.py preprocess_image_path outputpath
```
## Data
The data folder includes SourceAFIS synthetic dataset and FVC2004 fingerprint dataset, each of which includes the original incomplete fingerprint and the fingerprint repaired by our algorithm
