#C S 4033-001 Machine Learning
#Short SL Project - Bird Call Recognition
#David McKnight, Cody Degner, Andrew Calder

#Import packages
import glob;
import librosa;
import numpy;

#Initialize variables
#Acceptable file types
fileTypesAllowed = ["mp3"];#[, "wav", "mp3"];
numFileTypes = len(fileTypesAllowed);
#Training, validation, and testing data in these folders
dataPathPrefix = "data_";
trainingDataPaths = ["crow", "cardinal"];
numDataPaths = len(trainingDataPaths);

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
#TODO: SPLIT ALSO INTO TESTING DATA!!!
print("Splitting data...");
proportionTraining = 0.75;   #the remaining data will be used for validation
trainSize = [0 for i in range(numDataPaths)];    #size of training data, split by bird species
trainSizeTotal = 0;
valSize = [0 for i in range(numDataPaths)];
valSizeTotal = 0;
#Calculate sizes of training + validation data sets (again, split by bird species)
for i in range(numDataPaths):
    trainSize[i] = round(proportionTraining * numFiles[i]);
    valSize[i] = numFiles[i] - trainSize[i];
trainSizeTotal = sum(trainSize);
valSizeTotal = sum(valSize);
#Initialize the data arrays
dataTrain = [0 for j in range(trainSizeTotal)];
dataVal = [0 for j in range(valSizeTotal)];
#Stock the data arrays with appropriate sections of the master data array
#First n% will be training set, last (100-n)% will be validation set
currentIndexTrain = 0;
currentIndexVal = 0;
counter = 0;    #used to choose n% of birds for training and (100-n)% for validation
for i in range(numDataPaths):
    counter = 0;    #reset counter, because new bird
    for j in range(numFiles[i]):
        #Check: which set should this sample be part of?
        if (counter < trainSize[i]):    #training set
            dataTrain[currentIndexTrain] = data[i][j];
            currentIndexTrain += 1;
        else:                           #validation set
            dataVal[currentIndexVal] = data[i][j];
            currentIndexVal += 1;
        counter += 1;


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
print("Calculating cluster centroids...");
numClusters = numDataPaths;     #the number of bird species
clusters = [[-1 for j in range(trainSizeTotal)] for c in range(numClusters)];  #enough space for all data in single cluster
centroids = [[0.0 for k in range(numPredictors)] for c in range(numClusters)];
clusterSize = [0 for c in range(numClusters)];
#We'll also use these to store transformed data (each point is average of each predictor value- it turns out those are all arrays)
dataTrainConcise = [[0.0 for k in range(numVars)] for j in range(trainSizeTotal)];
dataValConcise = [[0.0 for k in range(numVars)] for j in range(valSizeTotal)];
for c in range(numClusters):
    #Set initial centroids
    #   Here, a centroid is a set of values, one for each predictor
    #   Set centroids = to average of each bird's values
    #Iterate through all data that has bird species 'i'
    #Then take the average of each predictor var value
    for j in range(trainSizeTotal):
        #Does this sample have the right species?
        if (dataTrain[j][numVars-1] == trainingDataPaths[c]):
            print("\tTRAINING SAMPLE " + str(j) + " MATCHED TO SPECIES " + trainingDataPaths[c]);
            for k in range(numPredictors):
                #Each predictor var is actually a giant array, so take the average of it
                predVarMean = sum(dataTrain[j][k][0])/len(dataTrain[j][k][0]);  #don't ask about the extra [0], it's just there and I'm too scared to try and fix it
                dataTrainConcise[j][k] = predVarMean;
                centroids[c][k] += predVarMean;
                print("\t\tPred var avg: " + str(predVarMean));
            clusterSize[c] += 1
            #TODO: do all this 'concise' business elsewhere (at end of previous step?)
            dataTrainConcise[j][5] = trainingDataPaths[c];
            
    #TODO: DO ALL THIS ELSEWHERE BUT WE NEED IT NOW!!!!
    for j in range(valSizeTotal):
        if (dataVal[j][numVars-1] == trainingDataPaths[c]):
            for k in range(numPredictors):
                #Each predictor var is actually a giant array, so take the average of it
                predVarMean = sum(dataVal[j][k][0])/len(dataVal[j][k][0]);  #don't ask about the extra [0], it's just there and I'm too scared to try and fix it
                dataValConcise[j][k] = predVarMean;
            dataValConcise[j][5] = trainingDataPaths[c];
    #END TODO
            
    #We've stored running sums, now divide them to calculate the final average
    for k in range(numPredictors):
        centroids[c][k] /= clusterSize[c];
print("Cluster centroids:");
print(centroids);
#Centroid distance threshold = max distance to be considered part of a cluster
#Set it to average of mean deviation from centroid of all birds
cdt = [0.0 for k in range(numPredictors)]
for j in range(trainSizeTotal):
    for c in range(numClusters):
        for k in range(numPredictors):
            #The deviation of this example
            dev = abs(dataTrainConcise[j][k] - centroids[c][k]);
            cdt[k] += dev;
            cdt[k] += dev;
            #print("\tdev: " + str(dev));
        #print("");
#We have running sum, now take average
allowanceFactor = 1.0;
for k in range(numPredictors):
    cdt[k] = abs(cdt[k]/trainSizeTotal);
    cdt[k] *= allowanceFactor;
#Do cluster reassignment / centroid recomputation until no centroids are recomputed
prevClusters = 0;
passes = 1;
acceptablePredictorThresholdProportion = 0.75;   #how many features do I need to 'pass' for in order to get in a cluster?
acceptablePredictorThreshold = round(acceptablePredictorThresholdProportion * numPredictors);
while (prevClusters != clusters):
    print("\tCluster pass " + str(passes) + "...");
    #Store previous cluster centroids
    prevClusters = clusters;
    #Wipe cluster data
    for c in range(numClusters):
        for j in range(trainSizeTotal):
           clusters[c][j] = -1;
        clusterSize[c] = 0;
    #Iterate through samples, find if each is near a centroid
    acceptablePredictors = [[0 for c in range(numClusters)] for j in range(trainSizeTotal)];      #to determine which cluster to assign this sample to
    for j in range(trainSizeTotal):
        for c in range(numClusters):
            for k in range(numPredictors):
                dev = abs(dataTrainConcise[j][k] - centroids[c][k]);
                if (dev <= cdt[k]):
                    #Deviation is within threshold, it's a member
                    acceptablePredictors[j][c] += 1;
        #Do we fit in a cluster?
        mostLikelyClusterResult = max(acceptablePredictors[j]);
        mostLikelyClusterIndex = acceptablePredictors[j].index(mostLikelyClusterResult);    #TODO address 'ties', right now it just gets the first index
        print(acceptablePredictors[j]);
        print("\tAcceptable predictors (out of 5) for sample " + str(j) + ": " + str(mostLikelyClusterResult) + ", in cluster " + str(mostLikelyClusterIndex));
        if (mostLikelyClusterResult >= acceptablePredictorThreshold):
            #Accept to this cluster
            clusters[mostLikelyClusterIndex][clusterSize[mostLikelyClusterIndex]] = j;    #just store the index
            clusterSize[mostLikelyClusterIndex] += 1;
    #Recompute centroids
    for c in range(numClusters):
        for k in range(numPredictors):
            for j in range(trainSizeTotal):
                #Store running sum
                centroids[c][k] += dataTrainConcise[j][k];
    #Average, remember?
    for c in range(numClusters):
        for k in range(numPredictors):
            centroids[c][k] /= clusterSize[c];
    print("\t\tPrevClusters:");
    print(prevClusters);
    print("\t\tClusters:");
    print(clusters);
    print("\t\tCentroids:");
    print(centroids);
    passes += 1;


#
#   PERFORM VALIDATION
#
print("Validating...");
#TODO do this....


#
#   PERFORM TESTING
#
print("Testing...");
#TODO
#FOR NOW: We'll use the validation set for testing, because we haven't implemented validation yet
#dataTest = dataVal;
dataTestConcise = dataValConcise;
testSizeTotal = valSizeTotal;
#Classify each element of the testing set
targetClass = ["" for i in range(testSizeTotal)]
acceptablePredictors = [[0 for c in range(numClusters)] for j in range(testSizeTotal)];      #to determine which cluster to assign this sample to
for j in range(testSizeTotal):
    #Find if the sample is near a centroid
    for c in range(numClusters):
        for k in range(numPredictors):
            dev = abs(dataTestConcise[j][k] - centroids[c][k]);
            if (dev <= cdt[k]):
                #Deviation is within threshold, it's a member
                acceptablePredictors[j][c] += 1;
    #Do we fit in a cluster?
    mostLikelyClusterResult = max(acceptablePredictors[j]);
    mostLikelyClusterIndex = acceptablePredictors[j].index(mostLikelyClusterResult);    #TODO address 'ties', right now it just gets the first index
    print(acceptablePredictors[j]);
    print("\tAcceptable predictors (out of 5) for sample " + str(j) + ": " + str(mostLikelyClusterResult) + ", in cluster " + str(mostLikelyClusterIndex));
    if (mostLikelyClusterResult >= acceptablePredictorThreshold):
        #Accept to this cluster
        targetClass[j] = trainingDataPaths[mostLikelyClusterIndex];     #store bird species name of the cluster
print("Target classes:");
print(targetClass);


#Finished
