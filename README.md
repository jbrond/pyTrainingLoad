# trainingLoad

<B>Installing pyTrainingLoad package</B>

Installing the pyTrainingLoad package is straight forward:

pip install git+https://github.com/jbrond/pyTrainingLoad.git

Two example files are included for testing one accelerometry (Axivity AX3) and one with heart rate measured with Garmin Venu SQ.

The Axivity data is a raw cwa data file which can be loaded using the Open Movement Python package. The Garmin file is a TCX file contaning both GPS and heart rate data. The Garmin TCX was originally exported using Garmin Connect.