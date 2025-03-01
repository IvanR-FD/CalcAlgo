import readnumbersFromDatabase
import calculateStatistics
import AIcalculations

version = '1.4'
versionAI = '2.1'

# define size of set
lengthOfSet       = 16 # int(input('select the size of the set to bi analyzed: '))
lengthOfSetAI     = 16 # int(input('select the size of the set to bi analyzed: '))
lengthOfRows      = 56 # int(input('select the amount of rows to be played: '))
timeStampforPlay  = '2025-03-04' # input('enter the date to be played: ')

# if input('create Model') == 'y':
#     AIcalculations.createModel()
mod = input('select model: 1.4 statistics 0, 2.1 AI 1 (all sets), 2.2 AI 2 (selected sets), 2.3 AI 3 no random, 2.4 AI no rand + dayOfWeek: ')  
if mod == '0':
    # get set of numbers from DB
    ListOfNumbers = readnumbersFromDatabase.getDBValues(lengthOfSet)

    # calculate the best numbers from past
    SuggestionList = calculateStatistics.calculateStats(ListOfNumbers)   
    # get randomized values from suggestion list
    RowsToBeplayed = calculateStatistics.randomizeSugetsionListValues(SuggestionList,lengthOfRows)

    # save results ?
    readnumbersFromDatabase.saveResultsRequest(RowsToBeplayed, timeStampforPlay, version)

elif mod == '3':
    versionAI = '2.3'
    lengthOfSetAI = 1000

    ListOfNumbers = readnumbersFromDatabase.getDBValues(lengthOfSetAI)

    cntOfListItems = 0
    RowsToBeplayed = []
    while cntOfListItems < lengthOfRows:
        SuggestionListAI = AIcalculations.AIcalculaionsProcedure(ListOfNumbers)

        if RowsToBeplayed == [] or not calculateStatistics.checkListIsallreadyused(RowsToBeplayed,SuggestionListAI):
            RowsToBeplayed.insert(cntOfListItems,SuggestionListAI)
            cntOfListItems += 1
            print('****************************\n' + str(cntOfListItems))

    # save results ?
    readnumbersFromDatabase.saveResultsRequest(RowsToBeplayed, timeStampforPlay, versionAI)

elif mod == '4':
    versionAI = '2.4'
    lengthOfSetAI = 1000

    ListOfNumbers = readnumbersFromDatabase.getDBValuesDOW(lengthOfSetAI)

    cntOfListItems = 0
    RowsToBeplayed = []
    while cntOfListItems < lengthOfRows:
        SuggestionListAI = AIcalculations.AIcalculaionsProcedureDOW(ListOfNumbers)

        if RowsToBeplayed == [] or not calculateStatistics.checkListIsallreadyused(RowsToBeplayed,SuggestionListAI):
            RowsToBeplayed.insert(cntOfListItems,SuggestionListAI)
            cntOfListItems += 1
            print('****************************\n' + str(cntOfListItems))

    # save results ?
    readnumbersFromDatabase.saveResultsRequest(RowsToBeplayed, timeStampforPlay, versionAI)


else:
     # get set of numbers from DB
    if mod == '1':  # all sets 2.1
        lengthOfSetAI = 1000
    else:           # 16 last sets 2.2
        versionAI = '2.2'
    
    ListOfNumbers = readnumbersFromDatabase.getDBValues(lengthOfSetAI)
    SuggestionListAI = AIcalculations.AIcalculaionsProcedure(ListOfNumbers)

    # get randomized values from suggestion list
    RowsToBeplayed = calculateStatistics.randomizeSugetsionValues(SuggestionListAI,lengthOfRows)

    # save results ?
    readnumbersFromDatabase.saveResultsRequest(RowsToBeplayed, timeStampforPlay, versionAI)