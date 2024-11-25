### 
### This code is developed by Christian P. Janssen of Utrecht University
### It is intended for students from the Master's course Cognitive Modeling
### Large parts are based on the following research papers:
### Janssen, C. P., & Brumby, D. P. (2010). Strategic adaptation to performance objectives in a dual‐task setting. Cognitive science, 34(8), 1548-1560. https://onlinelibrary.wiley.com/doi/full/10.1111/j.1551-6709.2010.01124.x
### Janssen, C. P., Brumby, D. P., & Garnett, R. (2012). Natural break points: The influence of priorities and cognitive and motor cues on dual-task interleaving. Journal of Cognitive Engineering and Decision Making, 6(1), 5-29. https://journals.sagepub.com/doi/abs/10.1177/1555343411432339
###
### If you want to use this code for anything outside of its intended purposes (training of AI students at Utrecht University), please contact the author:
### c.p.janssen@uu.nl


###
### import packages
###

import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from statistics import mean
import random

###
###
### Global parameters. These can be called within functions to change (Python: make sure to call GLOBAL)
###
###


###
### Car / driving related parameters
###
steeringUpdateTime = 250  # in ms ## How long does one steering update take? (250 ms consistent with Salvucci 2005 Cognitive Science)
timeStepPerDriftUpdate = 50  ### msec: what is the time interval between two updates of lateral position?
startingPositionInLane = 0.27  # assume that car starts already slightly away from lane centre (in meters) (cf. Janssen & Brumby, 2010)

# parameters for deviations in car drift due the simulator environment: See Janssen & Brumby (2010) page 1555
gaussDeviateMean = 0
gaussDeviateSD = 0.13  ##in meter/sec

### The car is controlled using a steering wheel that has a maximum angle. Therefore, there is also a maximum to the lateral velocity coming from a steering update
maxLateralVelocity = 1.7  # in m/s: maximum lateral velocity: what is the maximum that you can steer?
minLateralVelocity = -1 * maxLateralVelocity

startvelocity = 0  # a global parameter used to store the lateral velocity of the car

###
### Switch related parameters
###
retrievalTimeWord = 200  # ms. ## How long does it take to think of the next word when interleaving after a word (time not spent driving, but drifting)
retrievalTimeSentence = 300  # ms. ## how long does it take to retrieve a sentence from memory (time not spent driving, but drifting)

###
### parameters for typing task
###
timePerWord = 0  ### ms ## How much time does one word take
wordsPerMinuteMean = 39.33  # parameters that control typing speed: when typing two fingers, on average you type this many words per minute. From Jiang et al. (2020; CHI)
wordsPerMinuteSD = 10.3  ## this si standard deviation (Jiang et al, 2020)


## Function to reset all parameters. Call this function at the start of each simulated trial. Make sure to reset GLOBAL parameters.
def resetParameters():
    global timePerWord
    global retrievalTimeWord
    global retrievalTimeSentence
    global steeringUpdateTime
    global startingPositionInLane
    global gaussDeviateMean
    global gaussDeviateSD
    global gaussDriveNoiseMean
    global gaussDriveNoiseSD
    global timeStepPerDriftUpdate
    global maxLateralVelocity
    global minLateralVelocity
    global startvelocity
    global wordsPerMinuteMean
    global wordsPerMinuteSD

    timePerWord = 0  ### ms

    retrievalTimeWord = 200  # ms
    retrievalTimeSentence = 300  # ms

    steeringUpdateTime = 250  # in ms
    startingPositionInLane = 0.27  # assume that car starts already away from lane centre (in meters)

    gaussDeviateMean = 0
    gaussDeviateSD = 0.13  ##in meter/sec
    gaussDriveNoiseMean = 0
    gaussDriveNoiseSD = 0.1  # in meter/sec
    timeStepPerDriftUpdate = 50  ### msec: what is the time interval between two updates of lateral position?
    maxLateralVelocity = 1.7  # in m/s: maximum lateral velocity: what is the maximum that you can steer?
    minLateralVelocity = -1 * maxLateralVelocity
    startvelocity = 0  # a global parameter used to store the lateral velocity of the car
    wordsPerMinuteMean = 39.33
    wordsPerMinuteSD = 10.3


##calculates if the car is not accelerating more than it should (maxLateralVelocity) or less than it should (minLateralVelocity)  (done for a vector of numbers)
def velocityCheckForVectors(velocityVectors):
    global maxLateralVelocity
    global minLateralVelocity

    velocityVectorsLoc = velocityVectors

    if (type(velocityVectorsLoc) is list):
        ### this can be done faster with for example numpy functions
        velocityVectorsLoc = velocityVectors
        for i in range(len(velocityVectorsLoc)):
            if (velocityVectorsLoc[i] > 1.7):
                velocityVectorsLoc[i] = 1.7
            elif (velocityVectorsLoc[i] < -1.7):
                velocityVectorsLoc[i] = -1.7
    else:
        if (velocityVectorsLoc > 1.7):
            velocityVectorsLoc = 1.7
        elif (velocityVectorsLoc < -1.7):
            velocityVectorsLoc = -1.7

    return velocityVectorsLoc


## Function to determine lateral velocity (controlled with steering wheel) based on where car is currently positioned. See Janssen & Brumby (2010) for more detailed explanation.
## Lateral velocity update depends on current position in lane. Intuition behind function: the further away you are, the stronger the correction will be that a human makes
def vehicleUpdateActiveSteering(LD):
    latVel = 0.2617 * LD * LD + 0.0233 * LD - 0.022
    returnValue = velocityCheckForVectors(latVel)

    if LD > 0:
        returnValue = -returnValue

    return returnValue


### function to update lateral deviation in cases where the driver is NOT steering actively (when they are distracted by typing for example). Draw a value from a random distribution. This can be added to the position where the car is already.
def vehicleUpdateNotSteering():
    global gaussDeviateMean
    global gaussDeviateSD

    vals = np.random.normal(loc=gaussDeviateMean, scale=gaussDeviateSD, size=1)[0]
    returnValue = velocityCheckForVectors(vals)
    return returnValue


### Function to run a trial. Needs to be defined by students (section 2 and 3 of assignment)

def runTrial(nrWordsPerSentence=17, nrSentences=10, nrSteeringMovementsWhenSteering=4, interleaving="word"):
    global timePerWord
    resetParameters()
    locPos = [startingPositionInLane]
    trialTime = 0
    locColor = ["blue"]

    # Sample words per minute from normal distribution
    s = max(0, np.random.normal(wordsPerMinuteMean, wordsPerMinuteSD))
    timePerWord = 1 / (s / 60) * 1000

    if interleaving == "bonus":
        numberOfSteeringDrifts = math.floor(
            nrSteeringMovementsWhenSteering * steeringUpdateTime / timeStepPerDriftUpdate)

        words_left = nrWordsPerSentence * nrSentences  # total words to process

        while words_left > 0:
            # determine number of words to read in current batch; assume we read at least one word everytime
            words_to_read = random.choice(range(1, min(words_left, nrWordsPerSentence * 2))) if words_left > 1 else 1

            trialTime += timePerWord * words_to_read

            # add retrieval time for sentence or word depending on or current position in sentence
            if words_left % nrWordsPerSentence == 0:
                trialTime += retrievalTimeSentence
            else:
                trialTime += retrievalTimeWord

            # add sentence retrieval time if a new sentence start is encountered in our current word batch
            for w in range(words_left - words_to_read, words_left):
                if w % nrWordsPerSentence == 0:
                    trialTime += retrievalTimeSentence

            # simulate the car drifts while typing
            drift_steps = math.floor(timePerWord * words_to_read / timeStepPerDriftUpdate)
            for _ in range(drift_steps):
                locPos.append((locPos[-1] + vehicleUpdateNotSteering() * 0.05))
                locColor.append("red")

            # actively steer after one batch of words
            for _ in range(numberOfSteeringDrifts):
                locPos.append(locPos[-1] + vehicleUpdateActiveSteering(locPos[-1]) * 0.05)
                locColor.append("blue")

            words_left = words_left - words_to_read  # update remaining word count in email

    if interleaving == "word":
        numberOfDriftsPerWord = math.floor((timePerWord + retrievalTimeWord) / timeStepPerDriftUpdate)
        numberOfDriftsFirstWord = math.floor(
            (timePerWord + retrievalTimeWord + retrievalTimeSentence) / timeStepPerDriftUpdate)
        numberOfSteeringDrifts = math.floor(
            nrSteeringMovementsWhenSteering * steeringUpdateTime / timeStepPerDriftUpdate)

        for i in range(nrSentences):
            trialTime += retrievalTimeSentence
            for j in range(nrWordsPerSentence):
                trialTime += timePerWord
                trialTime += retrievalTimeWord
                trialTime += nrSteeringMovementsWhenSteering * steeringUpdateTime

                if j == 0:
                    n_drifts = numberOfDriftsFirstWord
                else:
                    n_drifts = numberOfDriftsPerWord

                for _ in range(n_drifts):
                    locPos.append((locPos[-1] + vehicleUpdateNotSteering() * 0.05))
                    locColor.append("red")

                if j < (nrWordsPerSentence - 1):
                    for _ in range(numberOfSteeringDrifts):
                        locPos.append(locPos[-1] + vehicleUpdateActiveSteering(locPos[-1]) * 0.05)
                        locColor.append("blue")

    if interleaving == "sentence":
        timePerSentence = timePerWord * nrWordsPerSentence
        numberOfDriftsPerSentence = math.floor(timePerWord * nrWordsPerSentence / timeStepPerDriftUpdate)
        numberOfSteeringDrifts = math.floor(
            nrSteeringMovementsWhenSteering * steeringUpdateTime / timeStepPerDriftUpdate)

        for i in range(nrSentences):
            trialTime += retrievalTimeSentence
            trialTime += timePerSentence
            trialTime += nrSteeringMovementsWhenSteering * steeringUpdateTime

            for _ in range(numberOfDriftsPerSentence):
                locPos.append((locPos[-1] + vehicleUpdateNotSteering() * 0.05))
                locColor.append("red")

            if i < (nrSentences - 1):
                for _ in range(numberOfSteeringDrifts):
                    locPos.append(locPos[-1] + vehicleUpdateActiveSteering(locPos[-1]) * 0.05)
                    locColor.append("blue")

    if interleaving == "drivingOnly":
        timePerEmail = timePerWord * nrWordsPerSentence * nrSentences + (nrSentences * retrievalTimeSentence)
        totalNumberOfSteers = math.floor(timePerEmail / timeStepPerDriftUpdate)

        trialTime = timePerWord * nrWordsPerSentence * nrSentences
        for _ in range(totalNumberOfSteers):
            locPos.append(locPos[-1] + vehicleUpdateActiveSteering(locPos[-1]) * 0.05)
            locColor.append("blue")

    if interleaving == "none":
        timePerEmail = timePerWord * nrWordsPerSentence * nrSentences + (nrSentences * retrievalTimeSentence)
        totalNumberOfDrifts = math.floor(timePerEmail / timeStepPerDriftUpdate)

        trialTime = timePerWord * nrWordsPerSentence * nrSentences
        for _ in range(totalNumberOfDrifts):
            locPos.append((locPos[-1] + vehicleUpdateNotSteering() * 0.05))
            locColor.append("red")

    mean_lat_dev = mean(locPos)
    max_lat_dev = max(map(abs, locPos))

    # scatter_plot(locPos, locColor, mean_lat_dev, max_lat_dev, trialTime)

    return trialTime, mean_lat_dev, max_lat_dev


def scatter_plot(locPos, locColor, mean_lat_dev, max_lat_dev, trialTime):
    plt.figure(figsize=(10, 6))
    timeAxis = np.arange(0, len(locPos)) * 50

    df = pd.DataFrame({'time': timeAxis, 'position': locPos})
    sns.scatterplot(data=df, x='time', y='position', hue=locColor, palette=['tab:blue', 'tab:red'], s=8)
    plt.xlabel("Time (ms)")
    plt.ylabel("Position (m)")
    mean_line = plt.axhline(y=mean_lat_dev, color='y',
                            label='Mean position = ' + str(round(mean_lat_dev, 4)))
    max_line = plt.axhline(y=max_lat_dev, color='g',
                           label='Max position (Absolute) = ' + str(round(max_lat_dev, 4)))
    plt.title("Lane Position Over Time (Interleaving Strategy 'word')")

    trial_time_line = plt.Line2D([0], [0], marker='', color='w',
                                 label='Total trial time = ' + str(round(trialTime, 2)) + ' ms')

    # Combine the lines for the legend
    plt.legend(handles=[mean_line, max_line, trial_time_line])
    plt.savefig("scatter_plot_driving_8.png")
    plt.show()


### function to run multiple simulations. Needs to be defined by students (section 3 of assignment)
def runSimulations(nrSims=100):
    totalTime = []
    meanDeviation = []
    maxDeviation = []
    Condition = []

    set_choices = range(15, 21)
    nrSentences = 10
    nrSteeringMovementsWhenSteering = 4

    for s in ["word", "sentence", "drivingOnly", "none", "bonus"]:
        for i in range(nrSims):
            nrWordsPerSentence = random.choice(set_choices)  # uniformly selected
            trialTime, mean_lat_dev, max_lat_dev = runTrial(nrWordsPerSentence, nrSentences,
                                                            nrSteeringMovementsWhenSteering, interleaving=s)
            totalTime.append(trialTime)
            meanDeviation.append(mean_lat_dev)
            maxDeviation.append(max_lat_dev)
            Condition.append(s)

    return totalTime, meanDeviation, maxDeviation, Condition


def plot_simulations(totalTime, meanDeviation, maxDeviation, Condition):
    plt.figure(figsize=(10, 6))
    marker_styles = ['o', 's', 'D', 'v', "p"]
    df = pd.DataFrame(
        {'time': totalTime, 'meanDeviation': meanDeviation, 'maxDeviation': maxDeviation, 'Condition': Condition})
    # plt.plot(totalTime, maxDeviation)
    sns.scatterplot(data=df, x='time', y='maxDeviation', style='Condition', markers=marker_styles, color='grey')

    avg_max_dev = []
    avg_trial_times = []
    std_max_dev = []
    std_time = []

    # mean per condition
    for s in ["word", "sentence", "drivingOnly", "none", "bonus"]:
        df_cond = df.loc[(df['Condition'] == s)]
        avg_max_dev.append(df_cond["maxDeviation"].mean())
        avg_trial_times.append(df_cond["time"].mean())
        std_max_dev.append(np.std(df_cond["maxDeviation"]))
        std_time.append(np.std(df_cond["time"]))

    df_cond = pd.DataFrame({'avg_max_dev': avg_max_dev, 'avg_trial_times': avg_trial_times,
                            'Condition': ["word (mean)", "sentence (mean)", "drivingOnly (mean)", "none (mean)",
                                          "bonus (mean)"]})
    sns.scatterplot(data=df_cond, x='avg_trial_times', y='avg_max_dev', hue='Condition', style='Condition',
                    markers=marker_styles, palette=["red", "blue", "green", "orange", "purple"], s=70)

    plt.errorbar(avg_trial_times, avg_max_dev, xerr=std_time, yerr=std_max_dev, fmt="none", capsize=5)

    plt.xlabel("Total Trial Time (ms)")
    plt.ylabel("Max Lateral Deviation (m)")
    plt.title("Simulations of Maximum Lateral Deviation Over Time")
    #    plt.legend(["word", "sentence", "drivingOnly", "none", "word (mean)", "sentence (mean)", "drivingOnly (mean)", "none (mean)"])
    plt.savefig("simulation_run_driving.png")
    plt.show()


# runTrial(interleaving="bonus")

totalTime, meanDeviation, maxDeviation, Condition = runSimulations(100)
plot_simulations(totalTime, meanDeviation, maxDeviation, Condition)
