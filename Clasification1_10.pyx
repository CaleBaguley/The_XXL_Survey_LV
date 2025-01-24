import time
import pandas as pd
import math as maths
from matplotlib import pyplot as plt
import numpy as np
import copy as cp
import sys
sys.path.insert(0, "/homeb/jb14389")
import GPy

# ------------- Main function ------------------

    # -- Main Input Paramiters --
    
        # Files = ['../PipelineV4.2/north42b.csv']"; Source csv files containing data
        
        # MaxDataCount = 1000; Maximum number of dat points the program will use. If the input
        #                      data points exed this then the program will randomly select down
        #                      to the given MaxDataCount. If None all data points used.
        
        # max_iters = 1000; Limit on the number of iterations the GPy optimisation of the
        #                   gausian classifire can make.
        
        # Iterations = 100; The number of iterations conducteed by the MCMC algorithm.
        
        # Cols  = [1,2,3,4,5,10,11,12,13,14]; Id of the columns to be used. Note the first column
        #                                     needs to be the C1C2 column as it is used for the
        #                                     priors.
        
        # LogCols  = []; List of columns to be loged before use. Don't apply this to the C1C2 column
        
        # NormCols  = -1; List of columns to be normalised before use. This maps the data linierly
        #                 to between 0 and 1. Don't apply this to the C1C2 column.
        #                 If -1 then it is set to equal Cols[1:]
        
        # NameCol = 0; Column used to name sources must be unique
        
        # SaveFile = "ClusterProb.csv"; Results are saved to this file
        
        # CopyDataUsedIn  = None; If not null takes the address of a pervious output file and matches
        #                         the data for the run. Use to repeat calculations on a previous data
        #                         set.
        
        # SamplePointsInputFile = None; If not null takes the a pervious output file and matches to 
        #                         the data to create a consistent set of sample points not used for
        #                         training.
        
        # SamplePointsOutputFile = "SamplePointsProb.csv"; File address to save sample point
        #                                                  probabilities to.

        # SamplePointsCount = None; If not none the sample points count is limited to the input with
        #                           sample points not used going into the pool for training selection

        # SamplePointsInputFileNameCol = 1
        
        # ARD = False; Use to enable automatic relevence detection within GPy.
	
	    # C_Val = [0.95, 0.5, 0.05] Prior values for C1, C2 and C0 respectivly.

	    # External_Prior = None, format: [["FileAdres", PriorValue(float)]] Set prior of sources in seprate file

def Main(Files = ['../Data/PipelineV4.3/XXLn_Reduced_Cols.csv'], MaxDataCount = 1000, max_iters = 1000, Iterations = 100, Cols = [3,7,8,9,10,18,19,20,21,22,23,24,25], LogCols = [], NormCols = -1, NameCol = 0, SaveFile = "ClusterProb.csv", Plot = False, CopyDataUsedIn = None, SamplePointsInputFile = None, SamplePointsOutputFile = "SamplePointsProb.csv", SamplePointsCount = None, SamplePointsInputFileNameCol = 1, ARD = False, C_Val = [0.95, 0.5, 0.05],  External_Prior = None):

    if(NormCols == -1):
        NormCols = Cols[1:]
        
    if(LogCols == -1):
        LogCols = Cols[1:]
        
    print("-- Loading data --")
    Data = np.concatenate([pd.read_csv(f).values for f in Files])
        
    print("-- Preping data --")
    Data, Prior, RawData, DataNames, Sample, SampleNames, RawSample = DataSetup(Data, Data[:,NameCol], Cols, LogCols, NormCols, MaxDataCount, CopyDataUsedIn = CopyDataUsedIn, SamplePointsFile = SamplePointsInputFile, C_Val = C_Val, External_Prior = External_Prior, SampleFileNameCol = SamplePointsInputFileNameCol, SamplePointsCount = SamplePointsCount)

    print("-- Running calculation --")
    DataPosterior, SamplePosterior, LengthScale, Variance = MC_Calc(Data, Prior, Iterations, ARD, Sample)
    
    print("-- Saving data to ", SaveFile, " --" )
    
    ResultsOut = [[DataNames[i],DataPosterior[i][0],Prior[i][0]] for i in range(len(DataNames))]
    
    pd.DataFrame(ResultsOut, columns=['Name','Posterior','Prior']).to_csv(SaveFile)
    
    if(SamplePosterior is None):
        pass
    else:
        print("-- Saving Sample data to ", SamplePointsOutputFile, " --")
    
        SampleResultsOut = [[SampleNames[i],SamplePosterior[i][0]] for i in range(len(SampleNames))]
        
        pd.DataFrame(SampleResultsOut, columns=['Name','Posterior']).to_csv(SamplePointsOutputFile)
    
    
    print("-- Results --")

    print("Min Prob;", min(DataPosterior), " Mean Prob;",np.mean(DataPosterior), " Max Prob;",max(DataPosterior))
    
    print("Lengthscales; ")
    
    for i in range(len(LengthScale)):
        print(Cols[i+1],LengthScale[i])
        
    print("Variance; ", Variance)

    if(Plot == True):
        Ploting(RawData, Prior, DataPosterior)
    
    
# --- Main Calculations ---

def MC_Calc(DataPoints, PriorOnLabels, Iterations, ARD = False, SamplePoints = None):

    # -- setup results variables --
    ProbabilityOfDataPoints = np.asarray(([[0.] for i in range(len(DataPoints))]))
    LengthScale = np.asarray([0.])
    Variance = 0.
    
    if(SamplePoints is None): ProbabilityOfSamplePoints = None
    else: ProbabilityOfSamplePoints = np.asarray(([[0.] for i in range(len(SamplePoints))]))
    
    
    StartTime = time.time()
    
    # --- Main calculation loop ---
    for i in range(Iterations):
    
        # Print progress, time taken and predicted time remaining in h:m:s format
        print(100*(i/Iterations),'%')
        
        if(i > 0):
            RunTime = time.time()-StartTime
            RunTimeHours = int(RunTime/(60**2))
            RunTimeMinits = int(RunTime/(60))-60*RunTimeHours
            RunTimeSeconds = int(RunTime) - 60*RunTimeMinits - 3600*RunTimeHours
            
            RemainingTime = RunTime*(Iterations-i)/i
            RemainingTimeHours = int(RemainingTime/(60**2))
            RemainingTimeMinits = int(RemainingTime/(60))-60*RemainingTimeHours
            RemainingTimeSeconds = int(RemainingTime) - 60*RemainingTimeMinits - 3600*RemainingTimeHours
        
            print('Elapsed;', RunTimeHours,'h', RunTimeMinits,'m', RunTimeSeconds, 's', 'Predicted remaining;', RemainingTimeHours, 'h',  RemainingTimeMinits, 'm', RemainingTimeSeconds, 's')
        
        # Run iteration calculations
        Labels = np.add( np.random.uniform(size=len(PriorOnLabels)).reshape((len(PriorOnLabels),1)), PriorOnLabels).astype(int)
        TmpProbabilityOfDataPoints, TmpLengthScale, TmpVariance, TmpProbabilityOfSamplePoints = ProbabilityOfDataGivenLables(DataPoints, Labels, ARD, SamplePoints)
        
        # Update results
        ProbabilityOfDataPoints = np.add(ProbabilityOfDataPoints, TmpProbabilityOfDataPoints)
        LengthScale = np.add(LengthScale, TmpLengthScale)
        Variance += TmpVariance
        if(SamplePoints is None): pass
        else: 
            ProbabilityOfSamplePoints = np.add(ProbabilityOfSamplePoints, TmpProbabilityOfSamplePoints)
            
    # Print total run time in h:m:s format
    RunTime = time.time()-StartTime
    RunTimeHours = int(RunTime/(60**2))
    RunTimeMinits = int(RunTime/(60))-60*RunTimeHours
    RunTimeSeconds = int(RunTime) - 60*RunTimeMinits - 3600*RunTimeHours
    
    print('Total elapsed time;', RunTimeHours,'h', RunTimeMinits,'m', RunTimeSeconds, 's')
        
    # Mean and return results
    if(SamplePoints is None): pass
    else: ProbabilityOfSamplePoints = np.true_divide(ProbabilityOfSamplePoints, Iterations)
    
    return np.true_divide(ProbabilityOfDataPoints, Iterations), ProbabilityOfSamplePoints, np.true_divide(LengthScale, Iterations), Variance/Iterations

# Returns: 
#   Probability for given positions,
#   Paramiter scales,
#   Probability of sample points (None if None entered for SamplePoints)
def ProbabilityOfDataGivenLables(Positions, Labels, ARD = False, SamplePoints = None):
    
    print("Generating modle")
    if (ARD == False):
        model = GPy.models.SparseGPClassification(Positions, Labels)

    else:
        kernel  = GPy.kern.RBF(len(Positions[0]), ARD = 1) #Radial basis function
        ##kernel += GPy.kern.White(len(Positions[0]))
        model = GPy.models.SparseGPClassification(Positions, Labels, kernel = kernel)

    print("Optimising modle")
    model.optimize()
    
    print("Calculating predictions")
    if(SamplePoints is None):
        ProbabilityOfSamplePoints = None
    else:
        ProbabilityOfSamplePoints = model.predict(SamplePoints)[0]
        
    return model.predict(Positions)[0], model.kern.lengthscale, model.kern.variance, ProbabilityOfSamplePoints
    
# --- General setup ---------------------------------------------------------------------------------------------------

def SeperateSamplePoints(SamplePointsFile, Data, DataNames, NameCol, PointsCount = None):
    
    
    if(SamplePointsFile is None):
        return Data, DataNames, None, None
    
    print('\nSeperating sample data listed in file;', SamplePointsFile)
    
    SampleNames = pd.read_csv(SamplePointsFile).values[:,NameCol]
    
    if(len(SampleNames) <= 0):
        print("Error: no data from", SamplePointsFile)
        exit()
    else:
        print("Sample points: {}".format(len(SampleNames)))
        
    if(PointsCount is not None and PointsCount < len(SampleNames)):
        print("Reducing sample data points from {} to {}".format(len(SampleNames), PointsCount))
        Choice = np.random.choice(SampleNames.shape[0], PointsCount, replace = False)
        SampleNames = SampleNames[Choice]
        

    IsSamplePoint = np.in1d(DataNames,SampleNames,assume_unique=True)

    SampleOut = Data[IsSamplePoint]
    SampleNamesOut = DataNames[IsSamplePoint]
    DataOut = Data[np.invert(IsSamplePoint)]
    DataNamesOut = DataNames[np.invert(IsSamplePoint)]

    print('Original data count: ', len(Data), ' Sample points: ', len(SampleNames), ' Data points left: ', len(DataOut))
    
    if(len(SampleOut) == 0):
        SampleOut = None
        SampleNamesOut = None
        
        print('\n -- WARNING : Sample points empty --\n')
      
    return DataOut, DataNamesOut, SampleOut, SampleNamesOut
    
def CopyDataFromFile(File, Data, DataNames):

    if( File is None):
        return Data, DataNames
    
    print("Removing data not in file: ", File)
        
    NamesFromFile = pd.read_csv(File).values[:,1]
    
    if(len(NamesFromFile) <= 0):
        print("Error: no data from", File)
        exit()
        
    IsInFile = np.in1d(DataNames, NamesFromFile, assume_unique=True)
    
    DataOut = Data[IsInFile]
    NamesOut = DataNames[IsInFile]
    
    print("File length; ", len(NamesFromFile), "\nData reduced from; ", len(Data), " to; ", len(DataOut))
    
    return DataOut, NamesOut
    
def RemoveNullValues(Data,Names):
    
    IsNull = pd.isnull(Data)
    ContainsNull = np.any(IsNull,1)
    DataOut = Data[np.invert(ContainsNull)]
    NamesOut = Names[np.invert(ContainsNull)]
    
    return DataOut, NamesOut
    
def RemoveValuesLessThanOrEqualToZero(Data,Names):

    print("Removing values less than or equal to zero")

    IsLessThanOrEqualToZero = np.invert(np.greater(Data[:,1:],0))

    ContainsLessThan = np.any(IsLessThanOrEqualToZero,1)

    DataOut = Data[np.invert(ContainsLessThan)]
    NamesOut = Names[np.invert(ContainsLessThan)]

    print("Resulting reduction from ", len(Names), " to ", len(NamesOut))

    return DataOut, NamesOut
    
def ReduceData(Data, DataNames, MaxDataCount):

    if(MaxDataCount is not None and len(Data) > MaxDataCount):
        print('Reducing data; ', len(Data), ' --> ', MaxDataCount)
        Choice = np.random.choice(Data.shape[0], MaxDataCount, replace = False)
        Data = Data[Choice,:]
        DataNames = DataNames[Choice]
    else:
        print('Data count below or equal to MaxDataCount; ', len(Data), ' <= ', MaxDataCount)

    return Data, DataNames
    
def LogData(Data, Sample, Cols, LogCols):

    if(len(LogCols) <= 0):
        return Data, Sample
    
    print('Taking Log of data')
    
    #Gets an array containing columns in Data to be loged
    IsLogCol = [i for i, x in enumerate(np.isin(Cols,LogCols)) if x]
    
    for i in IsLogCol:
        Data[:,i] = np.log10(Data[:,i].astype(float))
        Sample[:,i] = np.log10(Sample[:,i].astype(float))
   
    return Data, Sample
    
def NormData(Data, Cols, NormCols, Sample):

    if(len(NormCols) <= 0):
        return Data, Sample
    
    print('Normalising data')
    
    #Gets an array containing columns in Data to be normalised
    IsNormCol = [i for i, x in enumerate(np.isin(Cols,NormCols)) if x]
    
    for i in IsNormCol:
        
        Min = min(Data[:,i])
        Max = max(Data[:,i])
        Data[:,i] = (Data[:,i] - Min)/(Max - Min)
        if(Sample is not None):
            Sample[:,i] = (Sample[:,i] - Min)/(Max - Min)
        
   
    return Data, Sample
    
# Convert the C1, C2 and C0 classes to priors on the points being clusters  
def CalcPriors(Class, C_Val = [0.95, 0.5, 0.05]):
    
    print('Converting class to prior value')
    
    PY = [[0.] for i in range(len(Class))]
   
    for i in range(len(Class)):
        if(Class[i] == 2): # C2
            PY[i] = C_Val[1]
        elif(Class[i] == 1): # C1
            PY[i] = C_Val[0]
        else: # C0 and others
            PY[i] = C_Val[2]
    
    return np.reshape(np.asarray(PY),(len(PY),1))

def ChangePriorsForSourcesInFile(Ids, Priors, SourceFile, NewPrior):
    print(" - Seting prior of sources in {} to {} -".format(SourceFile, NewPrior))
    
    NamesFromFile = pd.read_csv(SourceFile).values[:,0]
    
    IsInFile = np.in1d(Ids, NamesFromFile, assume_unique=True)
    
    ArrayValuesToChange = np.argwhere(IsInFile)

    Priors[ArrayValuesToChange] = NewPrior

#---  Prep data to be used ---

def DataSetup(DataIn, DataNamesIn, Cols, LogCols, NormCols, MaxDataCount, CopyDataUsedIn = None, SamplePointsFile = None, C_Val = [0.95, 0.5, 0.05], External_Prior = None, SampleFileNameCol = 1, SamplePointsCount = None):

    Data = DataIn[:,Cols]
    
    DataNames = cp.deepcopy(DataNamesIn)
    
    Data, DataNames, Sample, SampleNames = SeperateSamplePoints(SamplePointsFile, Data, DataNames, NameCol = SampleFileNameCol, PointsCount = SamplePointsCount)
    
    Data, DataNames = CopyDataFromFile(CopyDataUsedIn, Data, DataNames)
    
    Data, DataNames = RemoveNullValues(Data, DataNames)
    if(Sample is not None):
        Sample, SampleNames = RemoveNullValues(Sample, SampleNames)
        
    Data,DataNames = RemoveValuesLessThanOrEqualToZero(Data,DataNames)
    if(Sample is not None):
        Sample, SampleNames = RemoveValuesLessThanOrEqualToZero(Sample, SampleNames)
    
    
    Data, DataNames = ReduceData(Data, DataNames, MaxDataCount)
    
    RawData = cp.deepcopy(Data)
    RawSample = cp.deepcopy(Sample)
    
    # Log specified columns
    Data, Sample = LogData(Data, Sample, Cols, LogCols)
    
    # Normalise data and sample in the same way
    Data, Sample = NormData(Data, Cols, NormCols, Sample)
        
    
    # Calc prior from class
    Priors = CalcPriors(Data[:,0], C_Val = C_Val)

    # Change priors from input files
    if(External_Prior is not None):
        print("-- Chainging prior based on input files --")
        for current in External_Prior:
            ChangePriorsForSourcesInFile(DataNames, Priors, current[0], current[1])
    
    # Remove classes to give training data
    Data = Data[:,1:]
    if(Sample is not None):
        Sample = Sample[:,1:]
    
    return Data, Priors, RawData, DataNames, Sample, SampleNames, RawSample

# -- Helper Functions -----------------------------------------------------------------------------------------------

def Ploting(RawData, Prior, Posterior):

    # -- Plot prior --

    sc = plt.scatter(RawData[:,1],RawData[:,4], c = Prior[:,0])

    plt.clim(0,1)

    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    plt.gca().set_xlim([min(RawData[:,1]*0.9),max(RawData[:,1])*1.1])
    plt.gca().set_ylim([min(RawData[:,4]*0.9),max(RawData[:,4])*1.1])

    plt.plot([min(RawData[:,1]*0.9),max(RawData[:,1])*1.1],[15,15], color = 'black', linewidth = 1, linestyle='dashed')
    plt.plot([min(RawData[:,1]*0.9),max(RawData[:,1])*1.1],[45,45], color = 'black', linewidth = 1, linestyle='dashed')

    plt.plot([ 5, 5],[min(RawData[:,4]*0.9),max(RawData[:,4])*1.1], color = 'black', linewidth = 1, linestyle='dashed')

    cbar = plt.colorbar(sc)

    cbar.ax.set_ylabel('Cluster Probability')
    cbar.ax.set_ylim([0,1])
    
    plt.title('Posterior cluster probability')

    plt.show()

    # -- Plot posterior --

    sc = plt.scatter(RawData[:,1],RawData[:,4], c = Posterior[:,0])

    plt.clim(0,1)

    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    plt.gca().set_xlim([min(RawData[:,1]*0.9),max(RawData[:,1])*1.1])
    plt.gca().set_ylim([min(RawData[:,4]*0.9),max(RawData[:,4])*1.1])

    plt.plot([min(RawData[:,1]*0.9),max(RawData[:,1])*1.1],[15,15], color = 'black', linewidth = 1, linestyle='dashed')
    plt.plot([min(RawData[:,1]*0.9),max(RawData[:,1])*1.1],[45,45], color = 'black', linewidth = 1, linestyle='dashed')

    plt.plot([ 5, 5],[min(RawData[:,4]*0.9),max(RawData[:,4])*1.1], color = 'black', linewidth = 1, linestyle='dashed')

    cbar = plt.colorbar(sc)

    cbar.ax.set_ylabel('Cluster Probability')
    cbar.ax.set_ylim([0,1])
    
    plt.title('Posterior cluster probability')

    plt.show()
    
    # -- Plot change --
    
    change = Posterior[:,0]-Prior[:,0]
    
    print(min(change),np.mean(change),max(change))
    
    sc = plt.scatter(RawData[:,1],RawData[:,4], c = change + 0.5, cmap = 'PRGn')

    plt.clim(0,1)

    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    plt.gca().set_xlim([min(RawData[:,1]*0.9),max(RawData[:,1])*1.1])
    plt.gca().set_ylim([min(RawData[:,4]*0.9),max(RawData[:,4])*1.1])

    plt.plot([min(RawData[:,1]*0.9),max(RawData[:,1])*1.1],[15,15], color = 'black', linewidth = 1, linestyle='dashed')
    plt.plot([min(RawData[:,1]*0.9),max(RawData[:,1])*1.1],[45,45], color = 'black', linewidth = 1, linestyle='dashed')

    plt.plot([ 5, 5],[min(RawData[:,4]*0.9),max(RawData[:,4])*1.1], color = 'black', linewidth = 1, linestyle='dashed')

    cbar = plt.colorbar(sc, ticks = [0.1,0.3,0.5,0.7,0.9])

    cbar.ax.set_ylabel('Posteriror minus prior probability')
    cbar.ax.set_yticklabels([-0.4, -0.2, 0 , 0.2, 0.4])
    
    plt.title('Change in cluster probability')

    plt.show()
    
    return

        
# Calls main function if file called from comand line
if __name__ == "__main__":
    Main(MaxDataCount = 1000, Iterations = 2, Cols = [1,7,8,11,12,20,21,24,25,31,36,37,40,41,44,45,51,52,55,56], ARD = True, SaveFile = 'tmp.csv')
