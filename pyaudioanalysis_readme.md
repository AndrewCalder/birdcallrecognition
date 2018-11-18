An audio classification example
More examples and detailed tutorials can be found at the wiki
https://github.com/tyiannak/pyAudioAnalysis/wiki

pyAudioAnalysis provides easy-to-call wrappers to execute audio analysis tasks. Eg, this code first trains an audio segment classifier, given a set of WAV files stored in folders (each folder representing a different class) and then the trained classifier is used to classify an unknown audio WAV file

from pyAudioAnalysis import audioTrainTest as aT
aT.featureAndTrain(["classifierData/music","classifierData/speech"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "svmSMtemp", False)
aT.fileClassification("data/doremi.wav", "svmSMtemp","svm")

Result:
(0.0, array([ 0.90156761,  0.09843239]), ['music', 'speech'])

In addition, command-line support is provided for all functionalities. E.g. the following command extracts the spectrogram of an audio signal stored in a WAV file:

python audioAnalysis.py fileSpectrogram -i data/doremi.wav