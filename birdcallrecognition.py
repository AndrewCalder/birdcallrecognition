#C S 4033-001 Machine Learning
#Short SL Project - Bird Call Recognition
#David McKnight, Cody Degner, Andrew Calder

#Import packages
import glob;
import librosa;
import numpy;

#Initialize variables
#Acceptable file types
fileTypesAllowed = ["mp3"];#["wav", "mp3"];
numFileTypes = len(fileTypesAllowed);
#Training and validation data in these folders
dataPathPrefix = "data_";
trainingDataPaths = ["crow"];#, "cardinal"];
numDataPaths = len(trainingDataPaths);
#Testing data (mixed bird species) in this folder?
testingDataPath = "testing";

#Sample sound file
test = "test.wav";
test2 = trainingDataPaths[0] + "/XC171534-cardinalsong.mp3";

"""
# Beat tracking example
#from __future__ import print_function
#import librosa
# 1. Get the file path to the included audio example
filename = test2
# 2. Load the audio as a waveform `y`
#    Store the sampling rate as `sr`
y, sr = librosa.load(filename)
# 3. Run the default beat tracker
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
print('Estimated tempo: {:.2f} beats per minute'.format(tempo))
# 4. Convert the frame indices of beat events into timestamps
beat_times = librosa.frames_to_time(beat_frames, sr=sr)
print('Saving output to beat_times.csv')
librosa.output.times_csv('beat_times.csv', beat_times)
"""


#
#   LOAD DATA
#
print("Loading data...");
#First find the total number of files in all training data directories
numFiles = [0 for i in range(numDataPaths)];    #tracks number of files per bird
totalNumFiles = 0;
maxNumFiles = 0;
for i in range(numDataPaths):
    for ft in range(numFileTypes):
        files = glob.glob(".\\" + dataPathPrefix + trainingDataPaths[i] + "\\*." + fileTypesAllowed[ft]);
        numFiles[i] += len(files);
totalNumFiles = sum(numFiles);
maxNumFiles = max(numFiles);
#Initialize the training/validation data array
#filenames
samples = [[-1 for j in range(maxNumFiles)] for i in range(numDataPaths)];
#time series
datats = [[-1 for j in range(maxNumFiles)] for i in range(numDataPaths)];
#sampling rates
datasr = [[-1 for j in range(maxNumFiles)] for i in range(numDataPaths)];
#Find all data in the training folders
for i in range(numDataPaths):
    currentSampleIndex = 0; #need this
    for ft in range(numFileTypes):
        files = glob.glob(".\\" + dataPathPrefix + trainingDataPaths[i] + "\\*." + fileTypesAllowed[ft]);
        #Add these files to the data arrays
        for j in range(len(files)):
            samples[i][j + currentSampleIndex] = files[j];
            #Load the file with Librosa
            #"""
            ts, sr = librosa.load(files[j])
            datats[i][j] = ts;  #time series
            datasr[i][j] = sr;  #sample rates
            #"""
            print("\tLoaded sample [" + str(i+1) + "," + str(j+currentSampleIndex+1) + "]...");
        if (len(files) > 0):
            currentSampleIndex += 1;


#TODO: Subdivide audio files into 10-second-or-so samples?


#TODO: filter out background noise?


#
#   PERFORM FEATURE EXTRACTION
#
print("Extracting relevant features from data...");
"""
Features (predictor variables):
    Spectral centroids
    Spectral bandwidth
    Spectral contrast
    Spectral flatness
    Spectral rolloff
Target variables:
    Bird species
"""
allVars = ["spec_cent", "spec_band", "spec_cont", "spec_flat", "spec_roll", "bird_species"];
numVars = len(allVars);
numPredictors = numVars - 1;
#Keep track of all vars in 3d array
data = [[[0.0 for k in range(numVars)] for j in range(maxNumFiles)] for i in range(numDataPaths)];
#Extract features for all data
for i in range(numDataPaths):
    for j in range(numFiles[i]):
        print("\tSample [" + str(i+1) + "," + str(j+1) + "]...");
        """ #might not need this...
        #Check if there's actual data here, or just an empty slot
        print(datats[i][j]);
        if (any(datats[i][j]) == -1):
            continue;
        """
        #Extract each feature
        data[i][j][0] = librosa.feature.spectral_centroid(datats[i][j], datasr[i][j]);
        data[i][j][1] = librosa.feature.spectral_bandwidth(datats[i][j], datasr[i][j]);
        data[i][j][2] = librosa.feature.spectral_contrast(datats[i][j], datasr[i][j]);
        data[i][j][3] = librosa.feature.spectral_flatness(datats[i][j]);
        data[i][j][4] = librosa.feature.spectral_rolloff(datats[i][j], datasr[i][j]);
        data[i][j][5] = trainingDataPaths[i];       #bird species name = directory name


#
#   SPLIT DATA INTO TRAINING + VALIDATION SETS
#
print("Splitting data...");
proportionTraining = 0.75;   #the remaining data will be used for validation
trainSize = round(proportionTraining * totalNumFiles);
valSize = totalNumFiles - trainSize;
#Initialize the data arrays
dataTrain = [0 for j in range(trainSize)];
dataVal = [0 for j in range(valSize)];
#Stock the data arrays with appropriate sections of the master data array
#First n% will be training set, last (100-n)% will be validation set
currentIndex = 0;
for i in range(numDataPaths):
    for j in range(numFiles[i]):
        #Check: which set should this sample be part of?
        if (currentIndex < trainSize):      #training set
            dataTrain[currentIndex] = data[i][j];
        else:                               #validation set
            dataTrain[currentIndex-trainSize] = data[i][j];
        currentIndex += 1;
print("dataTrain[0]:")
print(dataTrain[0]);


#
#   PERFORM TRAINING
#
print("Training SL algorithm (k-means)...");
#Use k-means algorithm
"""
1. Choose the number of clusters (K).
2. Choose the initial seeds (cluster centroids).
3. For each example, find the cluster (C) with the nearest centroid and
assign to C.
4. Based on the new cluster assignment from step 3, recompute the centroids.
5. Repeat steps 3-4 until convergence (the cluster assignment no longer
changes).
"""
#Set number of clusters and initial centroids
numClusters = numDataPaths;     #the number of bird species
clusters = [[0.0 for k in range(numPredictors)] for i in range(numClusters)];
for i in range(numClusters):
    #Set initial centroids
    #Here, a centroid is a set of values, one for each predictor
    #Set centroids = to average of each bird's values
    clusters[i] = 0.0;
#Centroid distance threshold = max distance to be considered part of a cluster
#Set it to average of mean deviation from centroid of all birds
cdt = 1.0;
#Iterate through samples, find nearest centroid for each
#Recompute centroids
#TODO


#
#   PERFORM VALIDATION
#
print("Validating...");
#TODO

#Perform testing
print("Testing...");
#TODO
