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

imported("pvalue_test2.xlsx")
print(all_data)
