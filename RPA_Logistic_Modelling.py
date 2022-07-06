import sympy as sp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import itertools
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import math
from datetime import datetime
import os
from  more_itertools import unique_everseen
import random
from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore")


# Define Model Which will be utilised
def GLM(t,K,A,C,Q,B,M,v):
    num = K-A
    denom = (C + Q*np.exp(-B*(t-M)))**float(1.0/v)
    y = A + (num/denom)
    return y


# Def function calculate start-time
def DetermineInflectionPointTangent(xvalues,OptimalParameters):

    # Define symbols
    t,K,A,C,Q,B,M,v = sp.symbols('t K A C Q B M v')

    # Build Identical Model to GLM Function
    GLMSyPyNom = K - A
    GLMSyPyDenom = (C + Q*sp.exp(-B*(-M+t)))**(1.0/v)
    GLM = A + (GLMSyPyNom/GLMSyPyDenom)


    # Determine first and second derivative
    FirstDerrivative = sp.diff(GLM,t)
    SecondDerivative = sp.diff(GLM,t,t)



    # Determine Time of Inflection using Second Derivative
    try:
        ModelInflectionTimePoint = float(sp.solve(SecondDerivative,t)[0].subs([(K,OptimalParameters[0]),
                                               (A,OptimalParameters[1]),
                                               (C,OptimalParameters[2]),
                                               (Q,OptimalParameters[3]),
                                               (B,OptimalParameters[4]),
                                               (M,OptimalParameters[5]),
                                               (v,OptimalParameters[6])]))
    except:
        ModelInflectionTimePoint = float('NaN')



    # Determine Max Gradient - I.e First derivation equation and time of Inflection Point
    try:
        ModelMaxGradient = FirstDerrivative.subs([(t,ModelInflectionTimePoint),
                                       (K,OptimalParameters[0]),
                                       (A,OptimalParameters[1]),
                                       (C,OptimalParameters[2]),
                                       (Q,OptimalParameters[3]),
                                       (B,OptimalParameters[4]),
                                       (M,OptimalParameters[5]),
                                       (v,OptimalParameters[6])])
    except:
        ModelMaxGradient = float('NaN')


    # Determine Y value at inflection point
    InflectionY = float(GLM.subs([(t,ModelInflectionTimePoint),
                                   (K,OptimalParameters[0]),
                                   (A,OptimalParameters[1]),
                                   (C,OptimalParameters[2]),
                                   (Q,OptimalParameters[3]),
                                   (B,OptimalParameters[4]),
                                   (M,OptimalParameters[5]),
                                   (v,OptimalParameters[6])]))


    ## Create Tangent Formula
    w = sp.symbols('w')
    TangentLine = ModelMaxGradient*(w-ModelInflectionTimePoint) + InflectionY


    # Solve the tangent line to identify Reaction Start Point. I.e when the
    # tangent crosses the x-axis i.e. y=0
    ReactionStartTime = ModelInflectionTimePoint - ((InflectionY-0)/ModelMaxGradient)
    InterceptY = InflectionY - ModelMaxGradient*ModelInflectionTimePoint


    return (ReactionStartTime,
            ModelInflectionTimePoint,
            ModelMaxGradient,
            InterceptY,
            InflectionY)



# Function needed to calculate Parameters
def AsymptoteCalculation(array):
    Last10thArray = math.ceil(len(array)/10)
    LastSubArray = array[-Last10thArray:]
    AsCDict = {'Upper Asymptote Max Value':max(array),
               'Upper Asymptote Xth Percentile Max Value':max(LastSubArray),
               'Upper Asymptote Xth Percentile Mean Value':np.mean(LastSubArray),
               'Lower Asymptote Min Value':min(array)
              }
    return AsCDict

# Function needed to calculate Parameters values.
def GradientCalculator(xarray,yarray,step=1):
    GCDict = {'Mean Gradient':0,
              'Max Gradient':0,
              'Max Gradient Time':0}

    GradientValues = []
    for i in list(range(len(xarray)-step)):
        try: # JUST INCASE ONE OF THE VALUES IS MISSING
            Grad = (yarray[i+step]-yarray[i])/((xarray[i+step]-xarray[i]))
        except:
            Grad = float('NaN')
        GradientValues.append(Grad)


    GradArray = np.asarray(GradientValues)
    GCDict['Mean Gradient'] = np.mean(GradArray)
    GCDict['Max Gradient'] = np.max(GradArray)
    #Extract time points
    MaxTempIndex = np.where(GradArray==GCDict['Max Gradient'])[0][0]
    GCDict['Max Gradient Time'] = (xarray[MaxTempIndex] + xarray[MaxTempIndex-1])/2
    GCDict['Median Gradient'] = np.median(GradArray)

    return GCDict



def ExtractToP(ExpID,RawDataFile,BaselineDF):

    # DF ready to store output
    OutputDataFrame = pd.DataFrame()

    SubRaw = RawDataFile[RawDataFile['Experiment_ID']==ExpID]

    # Loop through Unique Tube IDs
    for Tube in SubRaw['Tube_Number'].unique():


        print('\n\n\n {} \n\n\n'.format(ExpID))
        print('\n\n\n {} \n\n\n'.format(Tube))


        # Extract Baseline and associated Std
        BaseDFRow = BaselineDF[(BaselineDF['Experiment_ID']==ExpID)&(BaselineDF['Tube_Number']==Tube)]
        PresetBaselineValue = BaseDFRow['Baseline Fluorescence (mV)'].tolist()[0]
        PresetBaselineStdValue = BaseDFRow['Baseline Fluorescence std (mV)'].tolist()[0]


        # Subset Raw Data for ExpID and Tube, all time points after 60 seconds (Start point for baseline)
        # and fluorescence before limit of detection.
        SubRawTubeSpef = SubRaw[(SubRaw['Tube_Number']==Tube)&(SubRaw['Fluorescence (mV)']<=4000)&(SubRaw['Time (ms)']>=60000)]



        # Top performing model dict
        TopOfSetDict = { 'ExpID': ExpID,
                         'Tube': Tube,
                         'Standard_Error_Mean':100000000000000,
                         'R2_value':0,
                         'Curve Fit Success':False,
                         'Original Baseline Fluorescence (mV)':PresetBaselineValue,
                         'Original Baseline Fluorescene std (mV)':PresetBaselineStdValue}



        # Extract Necesary Time Points Data
        Timepoints = SubRawTubeSpef['Time (ms)'].tolist()
        xdata = np.array(Timepoints)

        # Extract and Normalise Y Data
        NormYData = [float(RawY-PresetBaselineValue) for RawY in SubRawTubeSpef['Fluorescence (mV)'].tolist()]
        ydata = np.array(NormYData)

        # Generate preliminary Values
        GradientDict = GradientCalculator(xdata,ydata)
        AsymptoteDict = AsymptoteCalculation(ydata)

        # Guess parameters - reduced to find best values
        GuessParameterOptions = {'K':[AsymptoteDict['Upper Asymptote Max Value'],
                                      AsymptoteDict['Upper Asymptote Xth Percentile Mean Value'],
                                      AsymptoteDict['Upper Asymptote Max Value']*1.5,
                                      AsymptoteDict['Upper Asymptote Max Value']*2],

                                 'A':[AsymptoteDict['Lower Asymptote Min Value']],

                                 'Q':[0.1,1,5,10,20,50],

                                 'C':[1,2,3,4],

                                 'B':[GradientDict['Mean Gradient'],
                                     GradientDict['Mean Gradient']/2,
                                     GradientDict['Mean Gradient']/5,
                                     GradientDict['Mean Gradient']/10],

                                 'M':[GradientDict['Mean Gradient'],
                                     int(np.median(Timepoints)),
                                     int(np.median(Timepoints)/4),
                                     int(np.median(Timepoints)/2),
                                     int(np.median(Timepoints)*1.5)],

                                 'v':[0.1,2,3,5,10,20,50,100,150]}

        # All possible combinations of guess parameters
        AllGuessCombos = list(itertools.product(*list(GuessParameterOptions.values())))

        random.shuffle(AllGuessCombos)

        # Loop through parameter estimation models
        for Combinations in AllGuessCombos[:1000]:

            K = Combinations[0]
            A = Combinations[1]
            C = Combinations[2]
            Q = Combinations[3]
            B = Combinations[4]
            M = Combinations[5]
            v = Combinations[6]

            # Ensure the Gradient is not negative as fluorescence should not
            # overall decrease even for reactions which dont increase
            if B < 0:
                B = 0

            # Set these as the guess parameters
            p0 = [K,A,C,Q,B,M,v]

            # Gonna always run 'LM' model
            try:
                # Try to run curve fit
                popt, pcov = curve_fit(GLM, xdata, ydata ,p0, method='lm', maxfev = 50000)

                # Obtain Prodicted Y Values, SME, RScore
                PredictedY = GLM(xdata,*popt)
                StandardErrorMean = mean_squared_error(ydata,PredictedY)
                RScore = r2_score(PredictedY,ydata)

                # See if the fitted model is the best for the set
                if StandardErrorMean < TopOfSetDict['Standard_Error_Mean']:
                    TopOfSetDict['R2_value'] = RScore
                    TopOfSetDict['Standard_Error_Mean'] = StandardErrorMean
                    TopOfSetDict['Curve Fit Success'] = True
                    TopOfSetDict['Paramter Guess Methods']= ','.join([str(h) for h in Combinations])
                    TopOfSetDict['Parameter_Values'] = popt
                    TopOfSetDict['Predicted_Y_Values'] = PredictedY

            except:
                # If a curve fit is not possible
                Success = False



        TopOfSetDict['Raw_Data_Max_Fluorescence_Value_(mV)'] = max(NormYData)


        print('\n\n Now Assessing Best Model \n\n')

        # Only run if reaction was success
        if TopOfSetDict['Curve Fit Success'] == True:


            # Now best model has been found determine all Model Specific Values
            TopModelValues = DetermineInflectionPointTangent(Timepoints,TopOfSetDict['Parameter_Values'])


            TopOfSetDict['Model_Data_Onset_Time_(ms)'] = TopModelValues[0]
            TopOfSetDict['Model_Data_Inflection_Time_(ms)'] = TopModelValues[1]
            TopOfSetDict['Model_Data_Max_Gradient_(mV)'] = TopModelValues[2]


            # Generate necessary plot for experiment
            plt.plot(xdata,ydata,'b-' ,label="Raw Data") # Raw Data - Blue Line
            plt.plot(xdata,TopOfSetDict['Predicted_Y_Values'], 'r-' ,label="Model") # Top Model Y data - Red Line
            plt.plot(xdata, TopOfSetDict['Model_Data_Max_Gradient_(mV)']*xdata+TopModelValues[3],'g--',label="Inflection Point Tangent")
            plt.plot(TopModelValues[0],0,'kx',label='Reaction Onset') # Reaction Onset Marker - Black X


            plt.vlines(TopOfSetDict['Model_Data_Inflection_Time_(ms)'],0,TopModelValues[4],'b--') # Vertical line makring the onset
            plt.plot(TopOfSetDict['Model_Data_Inflection_Time_(ms)'],0,'kx',label='Time to Max Gradient')
            #plt.plot([], [], ' ', label="RMSE Value: {}".format(str(TopOfSetDict['Standard_Error_Mean']))) # Extract label on legend (R2 value)


            # Add necessary Labels and Legends
            plt.title('{}_{}'.format(ExpID,Tube))
            plt.xlabel('Time (ms)')
            plt.ylabel('fluorescence (mV)')
            plt.legend(loc="upper left")
            plt.ylim(bottom=-200)

            # Save figure
            plt.savefig('./Enhanced_Model_Images/{}_{}_1.png'.format(ExpID,Tube))
            plt.axis([0, 1200000, -200, 8000])
            plt.savefig('./Enhanced_Model_Images/{}_{}_2.png'.format(ExpID,Tube))

            # Clear figure ready for next tube.
            plt.clf()

            # Remove model Y values as no longer needed
            TopOfSetDict.pop('Predicted_Y_Values',None)



        # Add Data To Data Frame (if unsuccessful will be just be simple dict)
        OutputDataFrame = OutputDataFrame.append(TopOfSetDict,ignore_index=True)


    OutputDataFrame.to_csv('ExpMetrics/{}.csv'.format(ExpID),index=None)

    return OutputDataFrame



################################################################################
################################################################################
################################################################################
################################################################################


RawDataFile = pd.read_csv('Raw_Data.csv')

RawDataFile = RawDataFile.dropna()

BaselineDF = pd.read_csv('Baseline_Values.csv')

ExpIDList = RawDataFile['Experiment_ID'].unique().tolist()


ZippedInput = list(zip(ExpIDList,
                       [RawDataFile]*len(ExpIDList),
                       [BaselineDF]*len(ExpIDList)))


with Pool(processes=1) as pool:
    OutDFsList = pool.starmap(ExtractToP,ZippedInput)


FinalDF = pd.concat(OutDFsList)

FinalDF.to_csv('Reaction_Metrics.csv',index=None)
