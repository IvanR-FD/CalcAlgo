from collections import Counter
import random
import numpy as np
import math

def calculateStats(ListOfNumbers):
    try:
        # get all numbers from int arrays separated for 50s and 12s groups
        ListOf50s = [subarray[:5] for subarray in ListOfNumbers]

        for num in ListOfNumbers:
            print(str(num) + '\n')

        # get 1st column + right neighbour
        Col12  = []
        Col123 = []
        Col234 = []       
        Col345 = []
        Col45  = []
        Col12345  = []
        Col67   = []
        
        SuggestionList = []

        for i in range(list(ListOfNumbers).__len__()):
            Col12.insert( Col12.__len__(), ListOfNumbers[i][0])
            Col12.insert( Col12.__len__(), ListOfNumbers[i][1])
            Col123.insert(Col123.__len__(), ListOfNumbers[i][0])
            Col123.insert(Col123.__len__(), ListOfNumbers[i][1])
            Col123.insert(Col123.__len__(), ListOfNumbers[i][2])
            Col234.insert(Col234.__len__(), ListOfNumbers[i][1])
            Col234.insert(Col234.__len__(), ListOfNumbers[i][2])
            Col234.insert(Col234.__len__(), ListOfNumbers[i][3])
            Col345.insert(Col345.__len__(), ListOfNumbers[i][2])
            Col345.insert(Col345.__len__(), ListOfNumbers[i][3])
            Col345.insert(Col345.__len__(), ListOfNumbers[i][4])
            Col45.insert( Col45.__len__(), ListOfNumbers[i][3])
            Col45.insert( Col45.__len__(), ListOfNumbers[i][4])
            Col12345.insert( Col12345.__len__(), ListOfNumbers[i][0])
            Col12345.insert( Col12345.__len__(), ListOfNumbers[i][1])
            Col12345.insert( Col12345.__len__(), ListOfNumbers[i][2])
            Col12345.insert( Col12345.__len__(), ListOfNumbers[i][3])
            Col12345.insert( Col12345.__len__(), ListOfNumbers[i][4])
            Col67.insert( Col67.__len__(), ListOfNumbers[i][5])
            Col67.insert( Col67.__len__(), ListOfNumbers[i][6])

        short = False
        for i in range(6):
            match i:
                case 0:
                    LocColumn = Col12
                case 1:
                    LocColumn = Col123
                case 2:
                    LocColumn = Col234
                case 3:
                    LocColumn = Col345
                case 4:
                    LocColumn = Col45
                case 5:
                    LocColumn = Col67
                    short = True
                
            SuggestionList.insert(i, getNvaluesForSuggests(LocColumn,short))

        return SuggestionList


    except Exception as e:
        print(f"error occured: {e}")
        return None
    
def calculateStatsAll(ListOfNumbers):
    try:
        # get all numbers from int arrays separated for 50s and 12s groups
        for num in ListOfNumbers:
            print(str(num) + '\n')
        
        Col12345  = []
        Col67   = []
        
        SuggestionList = []

        for i in range(list(ListOfNumbers).__len__()):
            Col12345.insert( Col12345.__len__(), ListOfNumbers[i][0])
            Col12345.insert( Col12345.__len__(), ListOfNumbers[i][1])
            Col12345.insert( Col12345.__len__(), ListOfNumbers[i][2])
            Col12345.insert( Col12345.__len__(), ListOfNumbers[i][3])
            Col12345.insert( Col12345.__len__(), ListOfNumbers[i][4])
            Col67.insert( Col67.__len__(), ListOfNumbers[i][5])
            Col67.insert( Col67.__len__(), ListOfNumbers[i][6])

        short = False
        for i in range(7):
            match i:
                case 0:
                    LocColumn = Col12345
                case 1:
                    LocColumn = Col12345
                case 2:
                    LocColumn = Col12345
                case 3:
                    LocColumn = Col12345
                case 4:
                    LocColumn = Col12345
                case 5 | 6:
                    LocColumn = Col67
                    short = True

            match i:
                case 0| 1 | 2 | 3 | 4:
                    SuggestionList.insert(i, getNvaluesForSuggestsSeparated(LocColumn,short))
                    # removes the best values from the set
                    Col12345 = [x for x in Col12345 if x not in SuggestionList[i]]  
                case 5:
                   SuggestionList.insert(i, getNvaluesForSuggests(LocColumn,short))
                   Col67 = [x for x in Col67 if x not in SuggestionList[i]]  
                
                case 6:
                    SuggestionList[i-1].extend(getNvaluesForSuggests(LocColumn,short))
                
        return SuggestionList


    except Exception as e:
        print(f"error occured: {e}")
        return None

def calculateCorrelation(ListOfNumbers, lengthOfSet):
    # autocerrelation
    lenOftops = 3
    SuggestionList = []
    Colmin = 1
    Colmax = [50,12]

    for m in range(7):
        Col  = []
    
        for i in range(list(ListOfNumbers).__len__() - lengthOfSet,list(ListOfNumbers).__len__()):
            Col.insert( Col.__len__(), ListOfNumbers[i][m])
        
        coeff = []
        for k in range(lengthOfSet):
            coeff.insert(list(coeff).__len__(), calculateRk(Col, k))

        autocoeff = []
        for l in range(1,lengthOfSet):
            autocoeff.insert(list(autocoeff).__len__(), coeff[l]/coeff[0])


        top_indices = [i for i, _ in sorted(enumerate(autocoeff), key=lambda x: x[1], reverse=True)[:lenOftops]]
        
        changinngOfTops = []
        for i  in range(list(top_indices).__len__()):
            changinngOfTops.insert(list(changinngOfTops).__len__(), Col[Col.__len__() - (top_indices[i] + 1)] - Col[Col.__len__() - (top_indices[i] + 2)] )
            
        guessedNumbers = []
        for i in range(list(changinngOfTops).__len__()):
            localNumber = Col[Col.__len__() - 1] + changinngOfTops[i]
            if m < 5:
                localNumber = max(Colmin, min(Colmax[0], localNumber))
            else:
                localNumber = max(Colmin, min(Colmax[1], localNumber))

            if not guessedNumbers or not guessedNumbers.__contains__(localNumber):
                guessedNumbers.insert(list(guessedNumbers).__len__(), localNumber)
        
        SuggestionList.append(guessedNumbers)

    return SuggestionList

def calculateRk(ValArray, lag):
    LocSum = 0
    for i in range(list(ValArray).__len__() - lag):
       LocSum += ValArray[i] * ValArray[i + lag]
    return LocSum

    
        
def ListListcontainsValue(MainList, SearchValue):
    for SubList in MainList:
        if SearchValue in SubList:
            return True
    return False

def getNvaluesForSuggests(LocList, short):
    # calculate the frequence of single numbers in 1st two columns
    frequence = Counter(LocList[:len(LocList)])
    frequence = frequence.most_common()

    # collect the classified groups
    collections = []
    quantity    = []
    
    for i in range(len(frequence)):
        localList = []
        
        # check if new member is allready in one of the group
        jumpOver = ListListcontainsValue(collections,frequence[i][0])

        if not jumpOver:
            localList.insert(0, frequence[i][0])
            localquantity = frequence[i][1]

            # compare the next value with the one which is selected in upper loop
            for k in range(len(frequence)):
                # check if the second value is allready in one of the group
                jumpOver = ListListcontainsValue(collections,frequence[k][0])
                
                neighborLen = 2
                if short:
                    neighborLen = 1

                if (abs(frequence[k][0] - localList[0]) <= neighborLen and not jumpOver) and (i != k):
                    localList.append(frequence[k][0])
                    localquantity += frequence[k][1]

            collections.append(localList)    
            quantity.append(localquantity) 

    # get maximalquantity to select best set of collextions
    max_value = max(quantity)
    all_maxima = [index for index, value in enumerate(quantity) if value == max_value]
    
    # get the best selections
    bestCollection = []
    for bestIndex in all_maxima:
        for num in collections[bestIndex]:
                if num not in bestCollection:
                    bestCollection.append(num)

    # if short:
    #     # if len(all_maxima) > 1:
    #     #     for num in range(len(all_maxima) - 1,-1,-1):
    #     #         quantity.pop(all_maxima[num])
    #     #         collections.pop(all_maxima[num])

    #     # else:
    #     #     quantity.pop(all_maxima[0])
    #     #     collections.pop(all_maxima[0])

    #     max_value = max(quantity)
    #     all_maxima = [index for index, value in enumerate(quantity) if value == max_value]

    #     for bestIndex in all_maxima:
    #         for num in collections[bestIndex]:
    #             bestCollection.append(num)

    return bestCollection

# dont concats groups togehter
def getNvaluesForSuggestsSeparated(LocList, short):
    # calculate the frequence of single numbers in 1st two columns
    frequence = Counter(LocList[:len(LocList)])
    frequence = frequence.most_common()

    # collect the classified groups
    collections = []
    quantity    = []
    
    for i in range(len(frequence)):
        localList = []
        
        # check if new member is allready in one of the group
        jumpOver = ListListcontainsValue(collections,frequence[i][0])

        if not jumpOver:
            localList.insert(0, frequence[i][0])
            localquantity = frequence[i][1]

            # compare the next value with the one which is selected in upper loop
            for k in range(len(frequence)):
                # check if the second value is allready in one of the group
                jumpOver = ListListcontainsValue(collections,frequence[k][0])
                
                neighborLen = 2
                if short:
                    neighborLen = 1

                if (abs(frequence[k][0] - localList[0]) <= neighborLen and not jumpOver) and (i != k):
                    localList.append(frequence[k][0])
                    localquantity += frequence[k][1]

            collections.append(localList)    
            quantity.append(localquantity) 

    # get the first maximalquantity to select the best set of collextions
    all_maxima = quantity.index(max(quantity))
    
    # get the best selections
    bestCollection = []
    for num in collections[all_maxima]:
        if num not in bestCollection:
            bestCollection.append(num)
        
    return bestCollection

def randomizeSugetsionValues(Suggests, numOfRow):
    SugestionList = []

    # get original guessed values aswell in the last position
    localSuggestsCollection = []
    for num in Suggests:
        localSuggestsCollection.append(num)

    SugestionList.insert(0,localSuggestsCollection)
    
    for sugCnt in range(1,numOfRow):
        localSuggestsCollection = []

        for cntOfNumbers in range(7):
            tempNmuber = 0
            match cntOfNumbers:
                case n if 0 <= n <= 4:
                    condition = 50
                    offset = 3    
                case _: 
                    condition = 12       
                    offset = 2

            match cntOfNumbers:
                case n if 0 <= n <= 6:
                    indexOfNumber = cntOfNumbers

                case _:
                    indexOfNumber = 6
            while tempNmuber <= 0 or tempNmuber > condition or tempNmuber in localSuggestsCollection:
                # get the indexed value and add an random offset
                randNumber = (int(round((random.random()*2 - 1))) * offset)

                # get float  number between 0 and 1 and normalize to perform index  
                if type(Suggests[indexOfNumber]) == int:
                    tempNmuber = Suggests[indexOfNumber] + randNumber
                else:
                    tempNmuber = int(round(random.random() * (len(Suggests[indexOfNumber]) - 1)))
                    tempNmuber = list(Suggests[indexOfNumber])[tempNmuber] + randNumber

            # create one suggestion
            localSuggestsCollection.append(tempNmuber)
        
        SugestionList.insert(sugCnt,localSuggestsCollection)

    return SugestionList
    
def randomizeSugetsionListValues(Suggests, numOfRow):
    SugestionList = []
    randSug_contentCntr = 0

    while randSug_contentCntr < numOfRow:

        # # get original guessed values aswell in the last position       
        # for sugCnt in range(0,numOfRow):
        localSuggestsCollection = []

        for cntOfNumbers in range(7):
            tempNmuber = 0
            match cntOfNumbers:
                case n if 0 <= n <= 4:
                    condition = 50
                    offset = 2    
                case _: 
                    condition = 12       
                    offset = 1

            match cntOfNumbers:
                case n if 0 <= n <= 5:
                    indexOfNumber = cntOfNumbers

                case _:
                    indexOfNumber = 5

            while tempNmuber <= 0 or tempNmuber > condition or tempNmuber in localSuggestsCollection:
                # get the indexed value and add an random offset
                randNumber = (int(round((random.random() *2 - 1))) * offset)

                # get float  number between 0 and 1 and normalize to perform index  
                if type(Suggests[indexOfNumber]) == int:
                    tempNmuber = Suggests[indexOfNumber] + randNumber
                else:
                    tempNmuber = int(round(random.random() * (len(Suggests[indexOfNumber]) - 1)))
                    tempNmuber = list(Suggests[indexOfNumber])[tempNmuber] + randNumber

            # create one suggestion
            localSuggestsCollection.append(tempNmuber)
            if  cntOfNumbers == 4: # sort list
                localSuggestsCollection.sort()
            elif cntOfNumbers == 6 :    # switch add numbers if first number is bigger
                if localSuggestsCollection[5] > localSuggestsCollection[6]:
                    localSuggestsCollection[6] = localSuggestsCollection[5]
                    localSuggestsCollection[5] = tempNmuber

        if SugestionList == [] or not checkListIsallreadyused(localSuggestsCollection,SugestionList):
            SugestionList.insert(randSug_contentCntr,localSuggestsCollection)
            randSug_contentCntr += 1
        

    return SugestionList

def randomizeSugetsionListValuesMixed(Suggests, numOfRow):
    SugestionList = []
    randSug_contentCntr = 0

    while randSug_contentCntr < numOfRow:

        # # get original guessed values aswell in the last position       
        # for sugCnt in range(0,numOfRow):
        localSuggestsCollection = []
        
        for numberindex in range(7):
            tempNmuber = 0

            # set limits
            match numberindex:
                case n if 0 <= n <= 4:
                    condition = 50
                    offset = 2    
                case _: 
                    condition = 12       
                    offset = 1

            # select collection for pick a numberfor randomizing
            match numberindex:
                case n if 0 <= n <= 4:
                    valueSet = False
                    locCounter = numberindex
                    selectionProbability = 0.5
                    while not valueSet:
                        randValue = np.random.rand() 
                        if randValue > selectionProbability:
                            locCounter = locCounter - 1 if locCounter > 0 else 0
                            selectionProbability = math.sqrt(selectionProbability)
                        else:
                            loclacollection = Suggests[locCounter]
                            valueSet = True                    

                case 5 | 6:
                    loclacollection = Suggests[5]            

            while tempNmuber <= 0 or tempNmuber > condition or tempNmuber in localSuggestsCollection:
                # get the indexed value and add an random offset
                randNumber = int(round(np.random.rand() * offset))

                # get float  number between 0 and 1 and normalize to perform index  
                if type(loclacollection) == int:
                    tempNmuber = loclacollection + randNumber
                else:
                    tempNmuber = int(round(np.random.rand() * (len(loclacollection) - 1)))
                    tempNmuber = list(loclacollection)[tempNmuber] + randNumber

            # create one suggestion
            localSuggestsCollection.append(tempNmuber)
            if  numberindex == 4: # sort list
                localSuggestsCollection.sort()
            elif numberindex == 6 :    # switch add numbers if first number is bigger
                if localSuggestsCollection[5] > localSuggestsCollection[6]:
                    localSuggestsCollection[6] = localSuggestsCollection[5]
                    localSuggestsCollection[5] = tempNmuber

        if SugestionList == [] or not checkListIsallreadyused(localSuggestsCollection,SugestionList):
            SugestionList.insert(randSug_contentCntr,localSuggestsCollection)
            randSug_contentCntr += 1
        

    return SugestionList

def checkListIsallreadyused(mainList, subList):
    return any(subList == sublist for sublist in mainList)


def AddNewItemsToList(ListOfNumsAI, lengthOFlist): 

    localList   = ListOfNumsAI
    Suggests    = ListOfNumsAI[0]
    for sugCnt in range(list(ListOfNumsAI).__len__(), lengthOFlist):
                
        gotValidValues = False
        while not gotValidValues :
            localSuggestsCollection = []
            for cntOfNumbers in range(7):
                tempNmuber = 0
                match cntOfNumbers:
                    case n if 0 <= n <= 4:
                        condition = 50
                        offset = 3    
                    case _: 
                        condition = 12       
                        offset = 2

                match cntOfNumbers:
                    case n if 0 <= n <= 6:
                        indexOfNumber = cntOfNumbers

                    case _:
                        indexOfNumber = 6

                while tempNmuber <= 0 or tempNmuber > condition or tempNmuber in localSuggestsCollection:
                    # get the indexed value and add an random offset
                    randNumber = (int(round((random.random()*2 - 1))) * offset)

                    # get float  number between 0 and 1 and normalize to perform index  
                    if type(Suggests[indexOfNumber]) == int:
                        tempNmuber = Suggests[indexOfNumber] + randNumber
                    else:
                        tempNmuber = int(round(random.random() * (len(Suggests[indexOfNumber]) - 1)))
                        tempNmuber = list(Suggests[indexOfNumber])[tempNmuber] + randNumber

                # create one suggestion
                localSuggestsCollection.append(tempNmuber)

            # check if the suggestion allready in the list
            if not checkListIsallreadyused(ListOfNumsAI,localSuggestsCollection):
                gotValidValues = True
                localList.insert(sugCnt,localSuggestsCollection)

    return localList

def randomizeCorelatedValues(Suggests, numOfRow):
    SugestionList = []
    localCounter = 0
    randomizedNumbers = 0

    # mixing values with numbers from each column
    while localCounter < 40000 and randomizedNumbers < numOfRow:
        localCounter += 1
        localSuggestsCollection = []

        for column in Suggests :
            numberSet = False
            while not numberSet:
                colIdx = random.randint(0, list(column).__len__() - 1 )
                guessedNumber = column[colIdx]
                if localSuggestsCollection == [] or guessedNumber not in localSuggestsCollection:
                    # create one suggestion
                    localSuggestsCollection.append(guessedNumber)
                    numberSet = True

        if SugestionList == [] or localSuggestsCollection not in SugestionList:
            SugestionList.insert(randomizedNumbers,localSuggestsCollection)
            randomizedNumbers += 1

    return SugestionList