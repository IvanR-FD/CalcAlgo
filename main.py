import readnumbersFromDatabase
import calculateStatistics
import AIcalculations

version = '1.4'
versionAI = '2.1'

# define size of set
lengthOfSet       = 16 # int(input('select the size of the set to bi analyzed: '))
lengthOfRows      = 56 # int(input('select the amount of rows to be played: '))
timeStampforPlay  = '2025-01-31' # input('enter the date to be played: ')

# if input('create Model') == 'y':
#     AIcalculations.createModel()
mod = input('select model: statistics 0, AI 1 (all sets), AI 2 (selected sets), AI 3 no random: ')  
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
    lengthOfSet = 1000

    ListOfNumbers = readnumbersFromDatabase.getDBValues(lengthOfSet)

    cntOfListItems = 0
    RowsToBeplayed = []
    while cntOfListItems < lengthOfRows:
        SuggestionListAI = AIcalculations.AIcalculaionsProcedure(ListOfNumbers)

        if RowsToBeplayed == [] or not calculateStatistics.checkListIsallreadyused(RowsToBeplayed,SuggestionListAI):
            RowsToBeplayed.insert(cntOfListItems,SuggestionListAI)
            cntOfListItems += 1

    # save results ?
    readnumbersFromDatabase.saveResultsRequest(RowsToBeplayed, timeStampforPlay, versionAI)

else:
     # get set of numbers from DB
    if mod == '1':  # all sets 2.1
        lengthOfSet = 1000
    else:           # 16 last sets 2.2
        versionAI = '2.2'
    
    ListOfNumbers = readnumbersFromDatabase.getDBValues(lengthOfSet)
    SuggestionListAI = AIcalculations.AIcalculaionsProcedure(ListOfNumbers)

    # get randomized values from suggestion list
    RowsToBeplayed = calculateStatistics.randomizeSugetsionValues(SuggestionListAI,lengthOfRows)

    # save results ?
    readnumbersFromDatabase.saveResultsRequest(RowsToBeplayed, timeStampforPlay, versionAI)