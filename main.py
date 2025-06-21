import readnumbersFromDatabase
import calculateStatistics
import AIcalculations

version = '1.4'
versionAI = '2.1'

# define size of set
lengthOfSet       = 16 # int(input('select the size of the set to bi analyzed: '))
lengthOfSetAI     = 16 # int(input('select the size of the set to bi analyzed: '))
lengthOfRows      = 56 # int(input('select the amount of rows to be played: '))
timeStampforPlay  = '2025-06-24' # input('enter the date to be played: ')
# mod  = 5
# automated version
for models in range(6):   
    mod = models

    # if input('create Model') == 'y':
    #     AIcalculations.createModel()
    # mod = input('select model: 1.4 statistics 0, 2.1 AI 1 (all sets), 2.2 AI 2 (selected sets), 2.3 AI 3 no random, 2.4 AI no rand + dayOfWeek: ')  
    if mod == 0:
        # get set of numbers from DB
        ListOfNumbers = readnumbersFromDatabase.getDBValues(lengthOfSet)

        # calculate the best numbers from past
        SuggestionList = calculateStatistics.calculateStats(ListOfNumbers)   
        # get randomized values from suggestion list
        RowsToBeplayed = calculateStatistics.randomizeSugetsionListValues(SuggestionList,lengthOfRows)

        # save results ?
        readnumbersFromDatabase.saveResultsRequest(RowsToBeplayed, timeStampforPlay, version)

    elif mod == 3 or mod == 4:
        if mod == 3 :
            versionAI = '2.3'
            ListOfNumbers = readnumbersFromDatabase.getDBValues(lengthOfSetAI)
        elif  mod == 4:
            versionAI = '2.4'
            ListOfNumbers = readnumbersFromDatabase.getDBValuesDOW(lengthOfSetAI)

        cntOfListItems = 0
        RowsToBeplayed = []
        locLoopcntr = 0

        while cntOfListItems < lengthOfRows and locLoopcntr <= 200 :
            if mod == 3:
                SuggestionListAI = AIcalculations.AIcalculaionsProcedure(ListOfNumbers)
            elif mod == 4:
                SuggestionListAI = AIcalculations.AIcalculaionsProcedureDOW(ListOfNumbers)

            locLoopcntr += 1

            if RowsToBeplayed == [] or not calculateStatistics.checkListIsallreadyused(RowsToBeplayed,SuggestionListAI):
                RowsToBeplayed.insert(cntOfListItems,SuggestionListAI)
                cntOfListItems += 1
                print('****************************\n' + str(cntOfListItems))

        if locLoopcntr >= 200:
            locLoopcntr = 0
            RowsToBeplayed = calculateStatistics.AddNewItemsToList(RowsToBeplayed,lengthOfRows)

        # save results ?
        readnumbersFromDatabase.saveResultsRequest(RowsToBeplayed, timeStampforPlay, versionAI)


    elif mod == 5:
        version = '1.5'

        # get set of numbers from DB
        ListOfNumbers = readnumbersFromDatabase.getDBValues(lengthOfSet)

        # calculate the best numbers from past
        SuggestionList = calculateStatistics.calculateCorrelation(ListOfNumbers, lengthOfSet)   
        # get randomized values from suggestion list
        RowsToBeplayed = calculateStatistics.randomizeCorelatedValues(SuggestionList,lengthOfRows)

        if RowsToBeplayed.__len__() < lengthOfRows:
            RowsToBeplayed = calculateStatistics.AddNewItemsToList(RowsToBeplayed,lengthOfRows)

        # save results ?
        readnumbersFromDatabase.saveResultsRequest(RowsToBeplayed, timeStampforPlay, version)

    else:
        # get set of numbers from DB
        if mod == 1:  # all sets 2.1
            lengthOfSetAI = 1000
        else:           # 16 last sets 2.2
            versionAI = '2.2'
        
        ListOfNumbers = readnumbersFromDatabase.getDBValuesDOW(lengthOfSetAI)
        SuggestionListAI = AIcalculations.AIcalculaionsProcedureDOW(ListOfNumbers)

        # get randomized values from suggestion list
        RowsToBeplayed = calculateStatistics.randomizeSugetsionValues(SuggestionListAI,lengthOfRows)

        # save results ?
        readnumbersFromDatabase.saveResultsRequest(RowsToBeplayed, timeStampforPlay, versionAI)