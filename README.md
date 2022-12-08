# Traffic-Sign-Classification
Traffic Sign Recognition (TSR) is undoubtedly one of the most important problems in the field of driverless cars and advanced driver assistance systems (ADAS). TSR enables the front-facing smart cameras mounted on the car to recognize the signboards so that the car can act accordingly. Some examples include recognizing stop signs, speed limit, turn signs etc.

Video presentation: https://liveconcordia-my.sharepoint.com/:f:/g/personal/na_bajwa_live_concordia_ca/EoyQ9c_J-z1FuWL5cFCyi6UBYknpENgmHqOEiG6EskmoBA?e=REVrHb

Requirements for running the code:
<br>
<li>1.Install Anaconda</li>
<li>2.Install Python 3.0+</li>
<li>Other packages that needs to installed are added in the .ipynb files.</li>




In order to run the resnet trainings, you need to run the jupyter notebook of the training you want. Each file is named with the architecture and data set it is trained on. Example:(resnet18_BTSD.ipynb for the training of resnet18 on the data set BTSD.)

In order to run the alexnet trainings, you need to run the jupyter notebook of the training you want. Each file is named with the architecture and data set it is trained on. Example:(Alexnet_BTSD.ipynb for the training of AlexNet on the data set BTSD.)

In order to run the GoogLeNet trainings, you need to run the jupyter notebook of the training you want. Each file is named with the architecture and data set it is trained on. Example:(GoogleNet_BTSD.ipynb for the training of GoogLeNet on the data set BTSD.)

In order to run the VGG-16 trainings, you need to run the jupyter notebook of the training you want. Each file is named with the architecture and data set it is trained on. Example:(vgg16_BTSD.ipynb for the training of VGG-16 on the data set BTSD.)

In order to run the Transfer Learning Model of MobilenetV3:
Go to : Transfer Learning/MobileNetV3/src/AIProject_MobileNetv3.ipynb
Then Run all the code blocks it will automatically call the other files and print the result on the terminal.

In order to run the Transfer Learning Model of GoogleNet:
Go to : Transfer Learning/GoogleNet/GoogleNet.ipynb
Then Run all the code blocks it will print the result on the terminal.

Datasets Download Link:
<br>
GTSRB Train Dataset: https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip
GTSRB Test Dataset:  https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip

BTSD Train Dataset: https://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Training.zip
BTSD Test Dataset:  https://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Testing.zip

rMASTIF Train Dataset: http://www.zemris.fer.hr/~kalfa/Datasets/rMASTIF/rmastif_train.tar.gz
rMASTIF Test Dataset:  http://www.zemris.fer.hr/~kalfa/Datasets/rMASTIF/rmastif_test.tar.gz
