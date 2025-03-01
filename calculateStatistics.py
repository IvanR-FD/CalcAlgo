from collections import Counter
import random

def calculateStats(ListOfNumbers):
    try:
        # get all numbers from int arrays separated for 50s and 12s groups
        ListOf50s = [subarray[:5] for subarray in ListOfNumbers]
        ListOf12s = [subarray[5:] for subarray in ListOfNumbers]

        for num in ListOfNumbers:
            print(str(num) + '\n')


        FlatArray50 = [num for subarray in ListOf50s for num in subarray]
        
        # get 1st column + right neighbour
        Col12  = []
        Col123 = []
        Col234 = []       
        Col345 = []
        Col45  = []
        Col67   = []
        
        SuggestionList = []

        for i in range(list(ListOfNumbers).__len__()):
            Col12.insert( Col12.__len__(), ListOfNumbers[i][0])
            Col12.insert( Col12.__len__(), ListOfNumbers[i][1])
            #Col123.insert(Col123.__len__(), ListOfNumbers[i][0])
            Col123.insert(Col123.__len__(), ListOfNumbers[i][1])
            Col123.insert(Col123.__len__(), ListOfNumbers[i][2])
            #Col234.insert(Col234.__len__(), ListOfNumbers[i][1])
            Col234.insert(Col234.__len__(), ListOfNumbers[i][2])
            Col234.insert(Col234.__len__(), ListOfNumbers[i][3])
            Col345.insert(Col345.__len__(), ListOfNumbers[i][2])
            Col345.insert(Col345.__len__(), ListOfNumbers[i][3])
            # Col345.insert(Col345.__len__(), ListOfNumbers[i][4])
            Col45.insert( Col45.__len__(), ListOfNumbers[i][3])
            Col45.insert( Col45.__len__(), ListOfNumbers[i][4])
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

            # compare the next value with the on which is selected in uuper lop
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

    if short:
        if len(all_maxima) > 1:
            for num in range(len(all_maxima) - 1,-1,-1):
                quantity.pop(all_maxima[num])
                collections.pop(all_maxima[num])

        else:
            quantity.pop(all_maxima[0])
            collections.pop(all_maxima[0])

        max_value = max(quantity)
        all_maxima = [index for index, value in enumerate(quantity) if value == max_value]

        for bestIndex in all_maxima:
            for num in collections[bestIndex]:
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
                case n if 0 <= n <= 5:
                    indexOfNumber = cntOfNumbers

                case _:
                    indexOfNumber = 5
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

    # get original guessed values aswell in the last position       
    for sugCnt in range(0,numOfRow):
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
        
        SugestionList.insert(sugCnt,localSuggestsCollection)

    return SugestionList

def checkListIsallreadyused(mainList, subList):
    return any(subList == sublist for sublist in mainList)