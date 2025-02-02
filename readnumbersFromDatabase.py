import mysql.connector
import readnumbersFromDatabase

def getDBValues(lengthOfSet):
    try:
        connection = mysql.connector.connect(
        host="localhost",
        user="admin",
        password="admin",
        database="mytestdbscheme"
        )

        cursor = connection.cursor()

        command = 'CALL `mytestdbscheme`.`getNumbers`('+ str(lengthOfSet) + ', @out1);\n '
        cursor.execute(command)
        cursor.fetchall()

        command = 'SELECT  @out1;'
        cursor.execute(command)
        result = cursor.fetchall()

        # split arrays from object
        ResultList = str(result).split(';')

        # remove useless chars from strings
        ResultList[0] = ResultList[0][3:]
        ResultList[-1] = ResultList[-1][: ResultList[-1].__len__() - 4]
        
        IntResultArray = []

        # split strings in int arrays
        for sublist in ResultList:       
            IntResultArray.insert(IntResultArray.__len__(),list(map(int, sublist.split(','))))

        return IntResultArray


    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None
    
    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()
            print("Database connection closed")

# get values from DB with day of week
def getDBValuesDOW(lengthOfSet):
    try:
        connection = mysql.connector.connect(
        host="localhost",
        user="admin",
        password="admin",
        database="mytestdbscheme"
        )

        cursor = connection.cursor()

        command = 'CALL `mytestdbscheme`.`getNumbersWithDayOfWeek`('+ str(lengthOfSet) + ', @out1);\n '
        cursor.execute(command)
        cursor.fetchall()

        command = 'SELECT  @out1;'
        cursor.execute(command)
        result = cursor.fetchall()

        # split arrays from object
        ResultList = str(result).split(';')

        # remove useless chars from strings
        ResultList[0] = ResultList[0][3:]
        ResultList[-1] = ResultList[-1][: ResultList[-1].__len__() - 4]
        
        IntResultArray = []

        # split strings in int arrays
        for sublist in ResultList:       
            IntResultArray.insert(IntResultArray.__len__(),list(map(int, sublist.split(','))))

        return IntResultArray


    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None
    
    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()
            print("Database connection closed")

def writeSuggestionsintoDB(RowToPlay, timeStamp, vers):
    try:
        connection = mysql.connector.connect(
        host="localhost",
        user="admin",
        password="admin",
        database="mytestdbscheme"
        )

        cursor = connection.cursor()

        # create SQL querry
        sql_query = 'CALL `mytestdbscheme`.`storeNumbers`('+ str(RowToPlay[0]) + ', ' + str(RowToPlay[1])
        sql_query += ', ' + str(RowToPlay[2]) + ', ' + str(RowToPlay[3]) + ', ' + str(RowToPlay[4]) 
        sql_query += ', ' + str(RowToPlay[5]) + ', ' + str(RowToPlay[6]) + ', ' + '\'' + timeStamp + '\', '
        sql_query += '\'' + vers + '\'' + ');\n '

        # sql_query = "INSERT INTO lastplayednumbers (1stNum, 2ndNum, 3rdNum, 4thNum, 5thNum, 6thNum, 7thNum, timeStamp)"
        # sql_query += " VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"

        # get values
        # data_array = [str(RowToPlay[0]), str(RowToPlay[1]), str(RowToPlay[2]), 
        #             str(RowToPlay[3]),str(RowToPlay[4]),str(RowToPlay[5]),str(RowToPlay[6]), timeStamp]

        # execute querry
        cursor.execute(sql_query)

        # Änderungen bestätigen
        connection.commit()
        return None


    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None
    
    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()

def saveResultsRequest(RowResults, timeStamp, Version):
    print('****************************\n\n')
    print(str(Version) + ' done')

    cnt = 1
    for Row in RowResults:
        print(str(cnt) + ') ' + str(Row) + ' \n')
        cnt +=  1

    cnt = 1
    if input('use vales: y or n? ') == 'y':
        for Row in RowResults:
            readnumbersFromDatabase.writeSuggestionsintoDB(Row, timeStamp,Version)
            cnt +=  1

   