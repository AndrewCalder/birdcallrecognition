#C S 4033-001 Machine Learning
#Short SL Project - Bird Call Recognition
#David McKnight, Cody Degner, Andrew Calder

#Import packages
import glob;
import librosa;
import math;
import numpy;
import os;
import sklearn.metrics;
import matplotlib.pyplot;

#Initialize variables
#Acceptable file types
fileTypesAllowed = ["wav"];#["mp3"];#[, "wav", "mp3"];
numFileTypes = len(fileTypesAllowed);
#Training, validation, and testing data in these folders
#dataPathPrefix = "";#"data_";
#dataPathSuffix = "";#"_test";
#trainingDataPaths = ["crow", "cardinal"];
dataPath = "kaggledata\\src_wavs";
#dataPath = "kaggledata\\src_wavs_test";
pathFileToFileId = "kaggledata\\rec_id2filename.txt";
pathFileIdToSpeciesId = "kaggledata\\rec_labels_test_hidden.txt";
pathSpeciesIdToSpecies = "kaggledata\\species_list.txt";
numSpecies = 19;
#numDataPaths = len(trainingDataPaths);


#
#   LOAD DATA
#
print("Loading data...");
#First find the total number of files in all training data directories
numFiles = 0;
for ft in range(numFileTypes):
    files = glob.glob(".\\" + dataPath + "\\*." + fileTypesAllowed[ft]);
    numFiles += len(files);

#TODO remove
#numFilesBreak = 10;
#numFiles = numFilesBreak;

#filenames
samples = [-1 for j in range(numFiles)];
#time series
datats = [-1 for j in range(numFiles)];
#sampling rates
datasr = [-1 for j in range(numFiles)];
#Find all data in the training folders
for ft in range(numFileTypes):
    files = glob.glob(".\\" + dataPath + "\\*." + fileTypesAllowed[ft]);
    #Add these files to the data arrays
    for j in range(len(files)):
        samples[j] = os.path.basename(files[j]);
        print(samples[j]);
        #Load the file with Librosa
        #"""
        ts, sr = librosa.load(files[j])
        datats[j] = ts;  #time series
        datasr[j] = sr;  #sample rates
        #"""
        print("\tLoaded sample [" + str(j+1) + " (" + samples[j] + ")]...");

        """
        #TODO remove
        if (j+1 >= numFilesBreak):
            break;
        """



#TODO: filter out background noise?

#TODO: Subdivide audio files into 10-second-or-so samples?


#
#   PERFORM FEATURE EXTRACTION
#
print("Extracting relevant features from data...");
"""
Features (predictor variables):
    Most common frequency
Target variables:
    Bird species
"""
#allVars = ["spec_cent", "spec_band", "spec_cont", "spec_flat", "spec_roll", "bird_species"];
allVars = ["fourier_frequency_mode", "bird_species"];
numVars = len(allVars);
numPredictors = numVars - 1;
#Keep track of all vars in 3d array
data = [[0.0 for k in range(numVars)] for j in range(numFiles)];
#Extract features for all data
for j in range(numFiles):
    print("\tSample [" + str(j+1) + "]...");
    #Perform fourier transform for the current sample
    ft = librosa.core.stft(datats[j]);
    ftlen = len(ft);
    ftwid = len(ft[0]);
    #Find magnitude of each element
    mags = [[0.0 for l in range(ftwid)] for k in range(ftlen)];
    #mags = [0.0 for k in range(ftlen)];
    for k in range(ftlen):
        for l in range(ftwid):
            mags[k][l] = abs(ft[k][l]);
        #mags[k] = abs(ft[k]);
        #if (k%100==0):
            #print(k);
    #Find the mode - the most common frequency
    #First we need to do some calculations...
    #print(datasr[i][j]);
    #print(ftlen-1);
    freqSpacing = math.ceil(float(datasr[j])/(ftlen-1));     #Hz
    #print(freqSpacing)
    #print(numpy.max(mags[0]))
    meanFreqPowers = numpy.mean(mags, axis=1);
    #print(meanFreqPowers)
    #print(len(meanFreqPowers))
    #Find most prevalent frequency
    maxFreqPower = -1.0;
    maxFreqVal = -1.0;
    for f in range(ftlen):
        currentPower = meanFreqPowers[f];
        if (currentPower > maxFreqPower):
            maxFreqPower = currentPower;
            maxFreqVal = f;
    #print(maxFreqPower);
    #print(maxFreqVal);
    #print(maxFreqVal*22);
    #Store most common frequency index (common sample rate assumed)
    data[j][0] = maxFreqVal;
    #data[j][0] = maxFreqPower;


#
#   LOAD SPECIES METADATA
#
print("Loading metadata...");
#This process will tell us which sound files belong to which species
#Load in the mapping array from: file id to filename
print("File ID to file name...");
fileNames = [0 for i in range(numFiles)];
file = open(pathFileToFileId, 'r')
lines = file.readlines()
i = 0 #indexing variable for loading into our array
for line in lines:
    words = line.split(',')
    if (i == 0):    #skip the header
        #print(str(i) + ": " + str(words));
        i += 1;
        continue;
    #Load in data
    fileNames[i-1] = words[1].strip() + ".wav";
    #print(str(i) + ", " + str(fileNames[i-1]));
    i += 1;
    #"""
    #TODO remove
    if (i+1 >= numFiles):
        break;
    #"""
#Load in the mapping array from: file id to species ids contained in file
print("File ID to species IDs...");
speciesIds = [[0 for sp in range(numSpecies)] for i in range(numFiles)];
file = open(pathFileIdToSpeciesId, 'r')
lines = file.readlines()
i = 0 #indexing variable for loading into our array
for line in lines:
    words = line.split(',')
    if (i == 0):    #skip the header
        #print(str(i) + ": " + str(words));
        i += 1;
        continue;
    #Load in data
    words[len(words)-1] = words[len(words)-1].strip();  #get rid of newline
    speciesIds[i-1] = words[1:len(words)];
    #print(str(i) + ", " + str(speciesIds[i-1]));
    i += 1;
    #"""
    #TODO remove
    if (i+1 >= numFiles):
        break;
    #"""
#Load in the mapping array from: file id to species ids contained in file
print("Species ID to species name...");
speciesNames = [[0 for r in range(2)] for i in range(numSpecies+1)];        #both a name and a 4-letter code
speciesNames[numSpecies][0] = "UNKN";       #add this extra species tag for unmarked species in data set
speciesNames[numSpecies][1] = "Unknown Species";
file = open(pathSpeciesIdToSpecies, 'r')
lines = file.readlines()
i = 0 #indexing variable for loading into our array
for line in lines:
    words = line.split(',')
    if (i == 0):    #skip the header
        #print(str(i) + ": " + str(words));
        i += 1;
        continue;
    #Load in data
    speciesNames[i-1][0] = words[1].strip();
    speciesNames[i-1][1] = words[2].strip();
    #print(str(i) + ", " + str(speciesNames[i-1][0]) + ", " + str(speciesNames[i-1][1]));
    i += 1;
#Finally, store the correct species for each bird in the dataset
print("Locating species...");
for j in range(numFiles):
    #Find file ID for this filename
    #print(samples[j]);
    fileIndex = 0;
    if samples[j] in fileNames:
        fileIndex = fileNames.index(samples[j]);
    else:
        #print(samples[j] + " not found.");
        data[j][numVars-1] = ["UNKN", "Unknown Species"];
        continue;
    #We have index; use it to find the species included
    speciesIdsInFile = speciesIds[fileIndex];
    #Is the species unknown?
    if (len(speciesIdsInFile) == 0):
        #Yep
        #print(samples[j] + " rejected from dataset (unknown species).");
        data[j][numVars-1] = ["UNKN", "Unknown Species"];
        continue;
    elif (speciesIdsInFile[0] == '?'):
        #Also yep
        #print(samples[j] + " rejected from dataset (unknown species).");
        data[j][numVars-1] = ["UNKN", "Unknown Species"];
        continue;
    #We're good to go now
    speciesNamesInFile = speciesIdsInFile;
    for sp in range(len(speciesIdsInFile)):
        #Get the name of this species
        #print(speciesIdsInFile[sp])
        speciesNamesInFile[sp] = speciesNames[int(speciesIdsInFile[sp])][1]
    #print(fileIndex);
    #print(speciesNamesInFile);
    data[j][numVars-1] = speciesNamesInFile;


#
#   PRE-PROCESS DATA
#
print("Pre-processing data...");
#Get rid of examples with unknown species
dataNew = [];
i = 0;  #counter variable for index of new dataset
for j in range(numFiles):
    if (data[j][numVars-1][0] != "UNKN"):
        #Known species, this is a good data point to use
        dataNew.append(data[j]);
        #print(data[j]);
#print(dataNew);
data = dataNew;
numFiles = len(dataNew);


#
#   SHUFFLE DATA
#
numIterations = 10;
errorsAvg = [0.0 for ncl in range(numSpecies-1)];
for iter in range(numIterations):
    print("Shuffling data...");
    #print(data);
    numpy.random.shuffle(data);
    #print(data);


    #
    #   SPLIT DATA INTO TRAINING, VALIDATION, AND TESTING SETS
    #
    print("Splitting data...");
    proportionTrain = 0.6;                                      #60%
    proportionVal = 0.0;                                        #0%
    proportionTest = 1.0 - proportionTrain - proportionVal;     #40%
    trainSize = 0;
    valSize = 0;
    testSize = 0;
    #Calculate sizes of training + validation data sets (again, split by bird species)
    trainSize = round(proportionTrain * numFiles);
    valSize = round(proportionVal * numFiles);
    testSize = numFiles - trainSize - valSize;
    #Initialize the data arrays
    dataTrain = [0 for j in range(trainSize)];
    dataVal = [0 for j in range(valSize)];
    dataTest = [0 for j in range(trainSize)];
    #Stock the data arrays with appropriate sections of the master data array
    currentIndexTrain = 0;
    currentIndexVal = 0;
    currentIndexTest = 0;
    for j in range(numFiles):
        #Check: which set should this sample be part of?
        if (j < trainSize):                       #training set
            dataTrain[currentIndexTrain] = data[j];
            currentIndexTrain += 1;
        elif (j < trainSize + valSize):           #validation set
            dataVal[currentIndexVal] = data[j];
            currentIndexVal += 1;
        else:                                           #testing set
            dataTest[currentIndexTest] = data[j];
            currentIndexTest += 1;


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
    #numClusters = numDataPaths;     #the number of bird species
    numClusters = numSpecies;
    errors = [0.0 for ncl in range(numSpecies-1)];
    for ncl in range(1,numSpecies):
        numClusters = ncl+1;
        clusters = [[] for c in range(numClusters)];  #enough space for all data in single cluster
        #centroids = [[0.0 for k in range(numPredictors)] for c in range(numClusters)];
        centroids = [0.0 for c in range(numClusters)];
        clusterSize = [0 for c in range(numClusters)];
        #We'll also use these to store transformed data (each point is average of each predictor value- it turns out those are all arrays)
        #dataTrainConcise = [[0.0 for k in range(numVars)] for j in range(trainSize)];
        #dataValConcise = [[0.0 for k in range(numVars)] for j in range(valSize)];
        for c in range(numClusters):
            #Set initial centroids
            #Iterate through all data that has bird species 'i'
            for j in range(trainSize):
                #Does this sample have the right species? (with repeats)
                if (speciesNames[c][1] in dataTrain[j][numVars-1]):
                    #print("\tTRAINING SAMPLE " + str(j) + " MATCHED TO SPECIES " + speciesNames[c][1]);
                    #Great, add it to running sum for its cluster centroid (will divide later to get mean)
                    clusters[c].append(j);
                    centroids[c] += dataTrain[j][0];
                    clusterSize[c] += 1;
            #We've stored running sums, now divide them to calculate the final average
            if (clusterSize[c] == 0):
                centroids[c] = 0.0; #avoid division by 0
            else:
                centroids[c] /= clusterSize[c];
        """
        print("Cluster centroids:");
        print(centroids);
        """
        #Centroid distance threshold = max distance to be considered part of a cluster
        #Set it to average of mean deviation from centroid of all birds
        #cdt = [[0.0 for k in range(numPredictors)] for c in range(numClusters)]
        cdt = [0.0 for c in range(numClusters)]
        for j in range(trainSize):
            for c in range(numClusters):
                #Is the given sample in this cluster?
                if (speciesNames[c][1] in dataTrain[j][numVars-1]):
                    dev = abs(dataTrain[j][0] - centroids[c]);
                    cdt[c] += dev;
                #Else, continue to the next cluster
        #We have running sum, now take average
        allowanceFactor = 1.0;
        for c in range(numClusters):
            if (clusterSize[c] == 0):
                cdt[c] = 0.0; #avoid division by 0
            else:
                cdt[c] /= clusterSize[c];
            cdt[c] *= allowanceFactor;
        #print("\tCDT");
        #print(cdt);
        #Do cluster reassignment / centroid recomputation until no centroids are recomputed
        prevClusters = 0;
        passes = 1;
        #acceptablePredictorThresholdProportion = 0.65;   #how many features do I need to 'pass' for in order to get in a cluster?
        #acceptablePredictorThreshold = round(acceptablePredictorThresholdProportion * numPredictors);
        clusterCentroidDeviations = [[1000.0 for c in range(numClusters)] for j in range(trainSize)];
        while (prevClusters != clusters):
            print("\tCluster pass " + str(passes) + "...");
            #Store previous clusters
            prevClusters = clusters.copy();
            #Wipe cluster data
            for c in range(numClusters):
                clusters[c] = [];
                clusterSize[c] = 0;
            #Iterate through samples, find if each is near a centroid
            for j in range(trainSize):
                for c in range(numClusters):
                    dev = abs(dataTrain[j][0] - centroids[c]);
                    #Store the deviation
                    clusterCentroidDeviations[j][c] = dev;
                #Do we fit in a cluster?
                mostLikelyClusterResult = min(clusterCentroidDeviations[j]);
                mostLikelyClusterIndex = clusterCentroidDeviations[j].index(mostLikelyClusterResult);    #TODO address 'ties', right now it just gets the first index
                #Check if we're close enough
                if (mostLikelyClusterResult <= cdt[c]):
                    #Deviation is within threshold, it's a member
                    clusters[mostLikelyClusterIndex].append(j);     #just store the index
                    clusterSize[mostLikelyClusterIndex] += 1;
                #else:
                    #Not clear enough result, so mark it as unknown
                    #print("Example " + str(j) + " fit in no clusters.");
                    #TODO testing...
                    #Add it anyway
                    clusters[mostLikelyClusterIndex].append(j);     #just store the index
                    clusterSize[mostLikelyClusterIndex] += 1;
            #Wipe centroid data
            for c in range(numClusters):
                centroids[c] = 0.0;
            #Recompute centroids
            for c in range(numClusters):
                for j in range(clusterSize[c]):
                        #Add each element of cluster to running sum
                        centroids[c] += dataTrain[clusters[c][j]][0];
                #We have running sums, now divide to find the average
                if (clusterSize[c] == 0):
                    centroids[c] = 0.0; #avoid division by 0
                else:
                    centroids[c] /= clusterSize[c];
            """
            print("\t\tPrevClusters:");
            print(prevClusters);
            print("\t\tClusters:");
            print(clusters);
            print("\t\tCentroids:");
            print(centroids);
            """
            passes += 1;


        #
        #   PERFORM TESTING
        #
        print("Testing...");
        #dataTestConcise = dataValConcise;
        #Classify each element of the testing set
        targetClass = [[] for i in range(testSize)];
        targetClassInt = [0 for i in range(testSize)]; #with species index instead of name
        #print(dataTest);
        #acceptablePredictors = [[0 for c in range(numClusters)] for j in range(testSize)];      #to determine which cluster to assign this sample to
        clusterCentroidDeviations = [[1000.0 for c in range(numClusters)] for j in range(trainSize)];

        #TODO remove
        #centroids = [1.6, 2.7, 3.6, 0.5, 1.125, 0.6666666666666666, 1.0, 2.857142857142857, 1.0625, 1.2142857142857142, 1.0731707317073171, 1.625, 3.7777777777777777, 1.0, 1.0, 1.1666666666666667, 1.0, 2.0, 1.125];

        #Test the samples
        for j in range(testSize):
            #Find if the sample is near a centroid
            for c in range(numClusters):
                dev = abs(dataTest[j][0] - centroids[c]);
                #Store the deviation
                clusterCentroidDeviations[j][c] = dev;
            #Do we fit in a cluster?
            mostLikelyClusterResult = min(clusterCentroidDeviations[j]);
            mostLikelyClusterIndex = clusterCentroidDeviations[j].index(mostLikelyClusterResult);    #TODO address 'ties', right now it just gets the first index
            #Check if we're close enough
            if (mostLikelyClusterResult <= cdt[c]):
                #Accept to this cluster
                targetClass[j].append(speciesNames[mostLikelyClusterIndex][1]);         #store bird species name of the cluster
                targetClassInt[j] = mostLikelyClusterIndex;
            #else:
                #Unknown species
                #print("Example " + str(j) + " fit in no clusters.");
                #targetClass[j] = "Unknown Species";
        #If we have some without any species, mark them as unknown species
        for j in range(testSize):
            if len(targetClass[j]) == 0:
                targetClass[j] = "Unknown Species";
        """
        print("Target classes:");
        print(targetClass);
        """


        #
        #   CALCULATE ERROR
        #
        print("Calculating error...");
        #First get the true values
        trueClass = ["" for j in range(testSize)];
        trueClassInt = [0 for j in range(testSize)];
        for j in range(testSize):
            trueClass[j] = dataTest[j][numVars-1];
            #Find the index of the species name
            for sp in range(numSpecies):
                if (speciesNames[sp][1] == dataTest[j][numVars-1][0]):
                    #Found it
                    trueClassInt[j] = sp;
        #mse = sklearn.metrics.mean_squared_error(trueClass, targetClass);
        mse = sklearn.metrics.mean_squared_error(trueClassInt, targetClassInt);
        errors[ncl-1] = mse;
        errorsAvg[ncl-1] += mse;
        print(str(ncl+1) + ", " + str(mse));



#
#   PLOT RESULTS
#
print("Plotting results...");
#First average the running sums of MSEs
for ncl in range(numSpecies-1):
    errorsAvg[ncl-1] /= numIterations;
matplotlib.pyplot.plot(errorsAvg);
matplotlib.pyplot.show();


#Finished
