import torch
import numpy as np
import openpyxl as xl

print("hello")
all_data

def imported(filename):
    wb = xl.open_workbook(filename)
    sheet = wb.active

    for row in range(1,sheet.max_row+1):
        for data in range(0, sheet.max_col+1):
            all_data=np.empty(sheet.max_row, sheet.max_col)
            all_data[row][col] = sheet.cell(row,col)

            
def gradient_descent(number_of_bounds, num_iterations=200, learning_rate=.1):
    final_values = numpy.ones(number_of_bounds) #creates array of 1's
    for i in range(num_iterations):
        final_values = arrayAddition(final_values, arrayMultiplication(learning_rate*-1,gradient(final_values)))
        #subtracts a fixed portion of the gradient based on the current values
        #need to create a graident() function
    return final_values


def arrayMultiplication(constant, array):
    for value in array:
        value*=constant
    return array


#array1 and array2 need to have the same size
def arrayAddition(array1, array2):
    array_final
    for i in range(len(array1)):
        array_final.append(array1[i] + array2[i])
    return array_final
        
def gradient(current_values_array):
    #tb made later
            
imported("pvalue_test2.xlsx")
print(all_data)
