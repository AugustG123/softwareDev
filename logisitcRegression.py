import torch
import numpy as np
import openpyxl as xl

print("hello")
all_data

def imported(filename):
    wb = xl.open_workbook(filename)
    sheet = wb.active
    all_data=np.empty(sheet.max_row, sheet.max_col)

    for row in range(1,sheet.max_row+1):
        for data in range(0, sheet.max_col+1):
            all_data[row][col] = sheet.cell(row,col)
    return all_data
            


def arrayMultiplication(constant, array):
    for i in range(len(array)):
        array[i]*=constant
    return array


#array1 and array2 need to have the same size
def arrayAddition(array1, array2):
    array_final=[]
    for i in range(len(array1)):
        array_final.append(array1[i] + array2[i])
    return array_final


def dot_product(array1, array2):
    total = 0.0
    for i in range(len(array1)):
        total += (float)(array1[i]*array2[i])
    return total


def logti_function(z):
    return (float) (1/((float)(1+e**(-z))))


def average_cost(weights,y): #damage level =1,2,3 or 4
        total_cost = 0
        for i in range(len(y)):
            total_cost+=-log(abs((float)y[i] - logti_function(dot_product(weights, alldata[i]))), 10)
        return total_cost/len(y)


def random_list(length):
    array = []
    for i in range(length):
        array.append(100.0 * random.random())
    return array


class LogisticRegressionModel:

    def __init__(self, data, num_iterations, learning_rate, model_number, y):
        self.data = data
        self.data_transposed = alldata.transpose(alldata, len(alldata[0]), len(alldata))
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.model_number = model_number
        self.y = y #output list
        self.weights = np.ones(len(data[0]))


    def gradient_descent():
        for i in range(100):
            new_weights = random_list(len(self.data[0]))
            for i in range(self.num_iterations):
                all_predictions = []
                for j in range(len(self.data)):
                    all_predictions.append(logti_function(dot_product(alldata[j],new_weights)))
                error_array = arrayAddition(all_predictions, arrayMultiplication(-1,self.y))
        
                gradients = []
                for k in range(len(self.alldata_transposed)):
                    gradients.append(dot_product(self.alldata_transposed[k], error_array))
                gradients = arrayMultiplication((1/len(self.data)),gradients)
        
                new_weights = arrayAddition(new_weights, arrayMultiplication(-1*self.learning_rate, gradients))
            if average_cost(new_weights,self.y) >= average_cost(self.weights, self.y):
                self.weights = new_weights
        return self.weights
            

data=imported("pvalue_test2.xlsx")
print(all_data)
