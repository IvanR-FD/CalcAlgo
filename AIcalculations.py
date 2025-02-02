
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import Input, Dense
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import numpy as np
import random

# import tensorflow
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Dense

def AIcalculaionsProcedure(ListOfNumsAI):
    # AIpredictions = createModel(ListOfNumsAI)
    # AIpredictions = createModelNv(ListOfNumsAI)
    AIpredictions = createModelNvWithError(ListOfNumsAI)
        
    # round values
    AIpredictions = np.round(AIpredictions)
    AIpredictionsAsInt = []

    for k in range(7):
        match k: 
            case n if 0 <= n <= 4:
                min = 1
                max = 50
                offset = 2
            case _: 
                min = 1
                max = 12
                offset = 1

        while AIpredictions[0][k] < min or AIpredictions[0][k] > max or AIpredictions[0][k] in AIpredictionsAsInt:
            if AIpredictions[0][k] < min:
                AIpredictions[0][k] += 1
            elif AIpredictions[0][k] > max:
                 AIpredictions[0][k] -= 1
            else:
                AIpredictions[0][k] += (int(round((random.random()*2 - 1))) * offset)
                
        AIpredictionsAsInt.append(int(AIpredictions[0][k]))

    return AIpredictionsAsInt   

# calculations with day of week
def AIcalculaionsProcedureDOW(ListOfNumsAI):
    AIpredictions = createModelNvWithErrorDOW(ListOfNumsAI)
        
    # round values
    AIpredictions = np.round(AIpredictions)
    AIpredictionsAsInt = []

    for k in range(7):
        match k: 
            case n if 0 <= n <= 4:
                min = 1
                max = 50
                offset = 2
            case _: 
                min = 1
                max = 12
                offset = 1

        while AIpredictions[0][k] < min or AIpredictions[0][k] > max or AIpredictions[0][k] in AIpredictionsAsInt:
            if AIpredictions[0][k] < min:
                AIpredictions[0][k] += 1
            elif AIpredictions[0][k] > max:
                 AIpredictions[0][k] -= 1
            else:
                AIpredictions[0][k] += (int(round((random.random()*2 - 1))) * offset)
                
        AIpredictionsAsInt.append(int(AIpredictions[0][k]))

    return AIpredictionsAsInt   

def createModel(Values):
    try:
        # genereate input and output data
        Input = []
        Output = []

        # pre define sequential model
        model = keras.Sequential()

        a = 1 == 0
        # define ins and outs
        if a :
            for cntOfSamplse in range(len(Values) - 1):
                Input.append(Values[cntOfSamplse])
                Output.append(Values[cntOfSamplse + 1])
                
            model.add(keras.layers.Dense(164, activation='relu', input_shape=(7,)))
        else:
            for cntOfSamplse in range(len(Values) - 2):
                # current_input = Values[cntOfSamplse] + Values[cntOfSamplse + 1]
                Input.append([Values[cntOfSamplse] + Values[cntOfSamplse + 1]])
                Output.append(Values[cntOfSamplse + 2])

            model.add(keras.layers.Dense(164, activation='relu', input_shape=(1,14)))

        Input   = np.array(Input)
        Output  = np.array(Output)

        X_train, X_test, Y_train, Y_test = train_test_split(Input, Output, test_size=0.2, random_state=42)

        
        
        model.add(keras.layers.Dense(264, activation='relu'))
        model.add(keras.layers.Dense(264, activation='relu'))
        model.add(keras.layers.Dense(7))

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        model.summary()
        model.save('AImodels/Seqential.keras')

        # x_train = tf.constant(X_train)
        # y_train = tf.constant(Y_train)
        # x_test  = tf.constant(X_test)
        # y_test  = tf.constant(Y_test)

        x_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        y_train = tf.convert_to_tensor(Y_train, dtype=tf.float32)
        x_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
        y_test = tf.convert_to_tensor(Y_test, dtype=tf.float32)

        # training of the model
        history = model.fit(
            x_train, y_train,
            epochs=45,
            batch_size=32,
            validation_data=(x_test, y_test),
            verbose=1
        )

        # validation of the model
        plt.plot(history.history['loss'], label='training loss')
        plt.plot(history.history['val_loss'], label='validation loss')
        plt.title('Model loss')
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.legend()
        plt.show()

        # create input for use (last numbers)
        if a :
            real_input = Output[len(Output)-1]
        else:
            real_input = Output[len(Output)-2].tolist() + Output[len(Output)-1].tolist()

        real_input_tensor = tf.convert_to_tensor(real_input, dtype=tf.float32)


        # adds one Batch-Dimension 
        real_input_tensor = tf.expand_dims(real_input_tensor, axis=0) 

        if not a:
            real_input_tensor = tf.expand_dims(real_input_tensor, axis=0) 

        predictions = model.predict(real_input_tensor) 
        
        if not a:
            predictions = np.squeeze(predictions, axis=0)

        return predictions

    except Exception as e:
        print(f"error occured: {e}")
        return None

def createModelNv(Values):
    try:
        # genereate input and output data
        Input = []
        Output = []

        # pre define sequential model
        model = keras.Sequential()

    
        # define ins and outs
        lenOfIrows = 3
        for cntOfSamplse in range(len(Values) - lenOfIrows):
            # current_input = Values[cntOfSamplse] + Values[cntOfSamplse + 1]
            Input.append([Values[cntOfSamplse] + Values[cntOfSamplse + 1] + Values[cntOfSamplse + 2]])
            Output.append(Values[cntOfSamplse + lenOfIrows])

        model.add(keras.layers.Dense(164, activation='relu', input_shape=(1,21)))

        Input   = np.array(Input)
        Output  = np.array(Output)

        X_train, X_test, Y_train, Y_test = train_test_split(Input, Output, test_size=0.2, random_state=42)

        
        
        model.add(keras.layers.Dense(264, activation='relu'))
        model.add(keras.layers.Dense(264, activation='relu'))
        model.add(keras.layers.Dense(7))

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        model.summary()
        model.save('AImodels/Seqential.keras')

        x_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        y_train = tf.convert_to_tensor(Y_train, dtype=tf.float32)
        x_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
        y_test = tf.convert_to_tensor(Y_test, dtype=tf.float32)

        # training of the model
        history = model.fit(
            x_train, y_train,
            epochs=45,
            batch_size=32,
            validation_data=(x_test, y_test),
            verbose=1
        )

        # validation of the model
        plt.plot(history.history['loss'], label='training loss')
        plt.plot(history.history['val_loss'], label='validation loss')
        plt.title('Model loss')
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.legend()
        plt.show()

        # create input for use (last numbers)
        real_input = Output[len(Output)-3].tolist() + Output[len(Output)-2].tolist() + Output[len(Output)-1].tolist()

        real_input_tensor = tf.convert_to_tensor(real_input, dtype=tf.float32)


        # adds one Batch-Dimension 
        real_input_tensor = tf.expand_dims(real_input_tensor, axis=0) 

        real_input_tensor = tf.expand_dims(real_input_tensor, axis=0) 

        predictions = model.predict(real_input_tensor) 
        predictions = np.squeeze(predictions, axis=0)

        return predictions

    except Exception as e:
        print(f"error occured: {e}")
        return None
    

def createModelNvWithError(Values):
    try:
        # genereate input and output data
        Input = []
        Output = []

        # pre define sequential model
        model = keras.Sequential()


        # define ins and outs
        lenOfIrows = 3
        for cntOfSamplse in range(len(Values) - lenOfIrows):
            # current_input = Values[cntOfSamplse] + Values[cntOfSamplse + 1]
            Input.append([Values[cntOfSamplse] + Values[cntOfSamplse + 1] + Values[cntOfSamplse + 2]])
            Output.append(Values[cntOfSamplse + lenOfIrows])

        model.add(keras.layers.Dense(164, activation='relu', input_shape=(1,21)))

        Input   = np.array(Input)
        Output  = np.array(Output)

        X_train, X_test, Y_train, Y_test = train_test_split(Input, Output, test_size=0.2, random_state=42)

        model.add(keras.layers.Dense(264, activation='relu'))
        model.add(keras.layers.Dense(264, activation='relu'))
        model.add(keras.layers.Dense(264, activation='relu'))
        model.add(keras.layers.Dense(7))

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        model.summary()
        model.save('AImodels/Seqential.keras')

        x_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        y_train = tf.convert_to_tensor(Y_train, dtype=tf.float32)
        x_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
        y_test = tf.convert_to_tensor(Y_test, dtype=tf.float32)

        # training of the model
        history = model.fit(
            x_train, y_train,
            epochs=60,
            batch_size=32,
            validation_data=(x_test, y_test),
            verbose=1
        )

        # validation of the model
        # plt.plot(history.history['loss'], label='training loss')
        # plt.plot(history.history['val_loss'], label='validation loss')
        # plt.title('Model loss')
        # plt.ylabel('loss')
        # plt.xlabel('epochs')
        # plt.legend()
        # plt.show()

        # create input for use (last numbers)
         # define same ins like for training before and get calculated outs to get the error 
        lenOfIrows = 3
        PredictedOutput = None
        for cntOfSamplse in range(len(Input)):
            real_input = []
            real_input_tensor = []
            predictions = []

            real_input = Input[cntOfSamplse]
            real_input_tensor = tf.convert_to_tensor(real_input, dtype=tf.float32)
            
            # adds one Batch-Dimension 
            real_input_tensor = tf.expand_dims(real_input_tensor, axis=0) 
            
            predictions = model.predict(real_input_tensor) 
            predictions = np.squeeze(predictions, axis=0)

            # get result in the predicted array
            if PredictedOutput is None:
                PredictedOutput = predictions
            else:
                PredictedOutput = np.vstack((PredictedOutput, predictions))


        # calculate error between predicted output and real results
        ErrorOnPredicted = None
        for cntOfPredOut in range(len(Output)):
            if ErrorOnPredicted is None:
                ErrorOnPredicted = Output[cntOfPredOut] - PredictedOutput[cntOfPredOut]
            else:
                ErrorOnPredicted = np.vstack((ErrorOnPredicted, Output[cntOfPredOut] - PredictedOutput[cntOfPredOut]))

        # define ins and outs for the 2nd network
        # genereate input and output data
        InputErr = []
        OutputErr = []

        # pre define sequential model
        modelErr = keras.Sequential()


        # define same ins like for the 1. net but set the error as output 
        for cntOfSamplse in range(len(Input)):
            InputErr.append(Input[cntOfSamplse])
            

        modelErr.add(keras.layers.Dense(164, activation='relu', input_shape=(1,21)))    


        # Input is still the same
        InputErr   = np.array(InputErr)
        OutputErr  = ErrorOnPredicted

        X_trainErr, X_testErr, Y_trainErr, Y_testErr = train_test_split(InputErr, OutputErr, test_size=0.2, random_state=42)

        modelErr.add(keras.layers.Dense(264, activation='relu'))
        modelErr.add(keras.layers.Dense(264, activation='relu'))
        modelErr.add(keras.layers.Dense(264, activation='relu'))
        modelErr.add(keras.layers.Dense(7))

        modelErr.compile(optimizer='adam', loss='mse', metrics=['mae'])
        modelErr.summary()
        modelErr.save('AImodels/Seqential.keras')

        x_trainErr = tf.convert_to_tensor(X_trainErr, dtype=tf.float32)
        y_trainErr = tf.convert_to_tensor(Y_trainErr, dtype=tf.float32)
        x_testErr = tf.convert_to_tensor(X_testErr, dtype=tf.float32)
        y_testErr = tf.convert_to_tensor(Y_testErr, dtype=tf.float32)

        # training of the model
        historyErr = modelErr.fit(
            x_trainErr, y_trainErr,
            epochs=60,
            batch_size=32,
            validation_data=(x_testErr, y_testErr),
            verbose=1
        )

        # plt.plot(historyErr.history['loss'], label='training loss Err')
        # plt.plot(historyErr.history['val_loss'], label='validation loss Err')
        # plt.title('Model loss')
        # plt.ylabel('loss')
        # plt.xlabel('epochs')
        # plt.legend()
        # plt.show()

        # create input for use (last numbers)
        real_inputErr = Output[len(Output)-3].tolist() + Output[len(Output)-2].tolist() + Output[len(Output)-1].tolist()
        real_inputErr_tensor = tf.convert_to_tensor(real_inputErr, dtype=tf.float32)

        # adds one Batch-Dimension 
        real_inputErr_tensor = tf.expand_dims(real_inputErr_tensor, axis=0) 
        real_inputErr_tensor = tf.expand_dims(real_inputErr_tensor, axis=0) 

        predictionsErr = []
        predictionsErr = modelErr.predict(real_inputErr_tensor) 
        predictionsErr = np.squeeze(predictionsErr, axis=0)

        # get last outputs for latest prediction
        # create input for use (last numbers)
        real_input = []
        real_input_tensor = []
        predictions       = []
        real_input = Output[len(Output)-3].tolist() + Output[len(Output)-2].tolist() + Output[len(Output)-1].tolist()

        real_input_tensor = tf.convert_to_tensor(real_input, dtype=tf.float32)


        # adds one Batch-Dimension 
        real_input_tensor = tf.expand_dims(real_input_tensor, axis=0) 
        real_input_tensor = tf.expand_dims(real_input_tensor, axis=0) 

        predictions = model.predict(real_input_tensor) 
        predictions = np.squeeze(predictions, axis=0)

        predictionsRefactored = predictions + predictionsErr

        return predictionsRefactored

    except Exception as e:
        print(f"error occured: {e}")
        return None
    
# same calculations but with day of week
def createModelNvWithErrorDOW(Values):
    try:
        # genereate input and output data
        Input = []
        Output = []
        OutputDOW = []

        # pre define sequential model
        model = keras.Sequential()


        # define ins and outs
        lenOfIrows = 3
        for cntOfSamplse in range(len(Values) - lenOfIrows):
            # current_input = Values[cntOfSamplse] + Values[cntOfSamplse + 1]
            Input.append([Values[cntOfSamplse] + Values[cntOfSamplse + 1] + Values[cntOfSamplse + 2]])
            Output.append(Values[cntOfSamplse + lenOfIrows][0:7])
            OutputDOW.append(Values[cntOfSamplse + lenOfIrows])

        model.add(keras.layers.Dense(164, activation='relu', input_shape=(1,24)))

        Input       = np.array(Input)
        Output      = np.array(Output)
        OutputDOW   = np.array(OutputDOW)

        X_train, X_test, Y_train, Y_test = train_test_split(Input, Output, test_size=0.2, random_state=42)

        model.add(keras.layers.Dense(264, activation='relu'))
        model.add(keras.layers.Dense(264, activation='relu'))
        model.add(keras.layers.Dense(264, activation='relu'))
        model.add(keras.layers.Dense(7))

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        model.summary()
        model.save('AImodels/Seqential.keras')

        x_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        y_train = tf.convert_to_tensor(Y_train, dtype=tf.float32)
        x_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
        y_test = tf.convert_to_tensor(Y_test, dtype=tf.float32)

        # training of the model
        history = model.fit(
            x_train, y_train,
            epochs=60,
            batch_size=32,
            validation_data=(x_test, y_test),
            verbose=1
        )

        # validation of the model
        # plt.plot(history.history['loss'], label='training loss')
        # plt.plot(history.history['val_loss'], label='validation loss')
        # plt.title('Model loss')
        # plt.ylabel('loss')
        # plt.xlabel('epochs')
        # plt.legend()
        # plt.show()

        # create input for use (last numbers)
         # define same ins like for training before and get calculated outs to get the error 
        lenOfIrows = 3
        PredictedOutput = None
        for cntOfSamplse in range(len(Input)):
            real_input = []
            real_input_tensor = []
            predictions = []

            real_input = Input[cntOfSamplse]
            real_input_tensor = tf.convert_to_tensor(real_input, dtype=tf.float32)
            
            # adds one Batch-Dimension 
            real_input_tensor = tf.expand_dims(real_input_tensor, axis=0) 
            
            predictions = model.predict(real_input_tensor) 
            predictions = np.squeeze(predictions, axis=0)

            # get result in the predicted array
            if PredictedOutput is None:
                PredictedOutput = predictions
            else:
                PredictedOutput = np.vstack((PredictedOutput, predictions))


        # calculate error between predicted output and real results
        ErrorOnPredicted = None
        for cntOfPredOut in range(len(Output)):
            if ErrorOnPredicted is None:
                ErrorOnPredicted = Output[cntOfPredOut] - PredictedOutput[cntOfPredOut]
            else:
                ErrorOnPredicted = np.vstack((ErrorOnPredicted, Output[cntOfPredOut] - PredictedOutput[cntOfPredOut]))

        # define ins and outs for the 2nd network
        # genereate input and output data
        InputErr = []
        OutputErr = []

        # pre define sequential model
        modelErr = keras.Sequential()


        # define same ins like for the 1. net but set the error as output 
        for cntOfSamplse in range(len(Input)):
            InputErr.append(Input[cntOfSamplse])
            

        modelErr.add(keras.layers.Dense(164, activation='relu', input_shape=(1,24)))    


        # Input is still the same
        InputErr   = np.array(InputErr)
        OutputErr  = ErrorOnPredicted

        X_trainErr, X_testErr, Y_trainErr, Y_testErr = train_test_split(InputErr, OutputErr, test_size=0.2, random_state=42)

        modelErr.add(keras.layers.Dense(264, activation='relu'))
        modelErr.add(keras.layers.Dense(264, activation='relu'))
        modelErr.add(keras.layers.Dense(264, activation='relu'))
        modelErr.add(keras.layers.Dense(7))

        modelErr.compile(optimizer='adam', loss='mse', metrics=['mae'])
        modelErr.summary()
        modelErr.save('AImodels/Seqential.keras')

        x_trainErr = tf.convert_to_tensor(X_trainErr, dtype=tf.float32)
        y_trainErr = tf.convert_to_tensor(Y_trainErr, dtype=tf.float32)
        x_testErr = tf.convert_to_tensor(X_testErr, dtype=tf.float32)
        y_testErr = tf.convert_to_tensor(Y_testErr, dtype=tf.float32)

        # training of the model
        historyErr = modelErr.fit(
            x_trainErr, y_trainErr,
            epochs=60,
            batch_size=32,
            validation_data=(x_testErr, y_testErr),
            verbose=1
        )

        # plt.plot(historyErr.history['loss'], label='training loss Err')
        # plt.plot(historyErr.history['val_loss'], label='validation loss Err')
        # plt.title('Model loss')
        # plt.ylabel('loss')
        # plt.xlabel('epochs')
        # plt.legend()
        # plt.show()

        # create input for use (last numbers)
        real_inputErr = OutputDOW[len(OutputDOW)-3].tolist() + OutputDOW[len(OutputDOW)-2].tolist() + OutputDOW[len(OutputDOW)-1].tolist()
        real_inputErr_tensor = tf.convert_to_tensor(real_inputErr, dtype=tf.float32)

        # adds one Batch-Dimension 
        real_inputErr_tensor = tf.expand_dims(real_inputErr_tensor, axis=0) 
        real_inputErr_tensor = tf.expand_dims(real_inputErr_tensor, axis=0) 

        predictionsErr = []
        predictionsErr = modelErr.predict(real_inputErr_tensor) 
        predictionsErr = np.squeeze(predictionsErr, axis=0)

        # get last outputs for latest prediction
        # create input for use (last numbers)
        real_input = []
        real_input_tensor = []
        predictions       = []
        real_input = OutputDOW[len(OutputDOW)-3].tolist() + OutputDOW[len(OutputDOW)-2].tolist() + OutputDOW[len(OutputDOW)-1].tolist()

        real_input_tensor = tf.convert_to_tensor(real_input, dtype=tf.float32)


        # adds one Batch-Dimension 
        real_input_tensor = tf.expand_dims(real_input_tensor, axis=0) 
        real_input_tensor = tf.expand_dims(real_input_tensor, axis=0) 

        predictions = model.predict(real_input_tensor) 
        predictions = np.squeeze(predictions, axis=0)

        predictionsRefactored = predictions + predictionsErr

        return predictionsRefactored

    except Exception as e:
        print(f"error occured: {e}")
        return None
    

def AIwithoutRandom(ListOfNumsAI):
    AIpredictions = createModelNvWithError(ListOfNumsAI)
    
    # round values
    AIpredictions = np.round(AIpredictions)
    
    AIpredictionsAsInt = []


    return AIpredictionsAsInt   