#!/usr/bin/python
from PIL import Image
import os, sys, xlwt
from pathlib import Path

workbook = xlwt.Workbook()

sheet = workbook.add_sheet("Sheet Name")
style_red = xlwt.easyxf('font: bold 1, color red;')
style_blue = xlwt.easyxf('font: color blue;')
style_bold = xlwt.easyxf('font: bold 1')

folder_name = "Cars_test"

strPath = str(Path(__file__).resolve().parent) + "\\" + folder_name

def resize():
    column=1
    row=1
    for folders in os.listdir(strPath):
        sheet.write(0, column, folders, style_bold)
        print(folders)
        for files in os.listdir(strPath + "\\" + folders):
            sheet.write(row, 0, files)
            for point in range (1,size+1):
                if point == column:
                    sheet.write(row, point, 1)
                else:
                    sheet.write(row, point, 0)
            row += 1
        column += 1
    workbook.save("labels.csv")
	
if os.path.isdir(strPath):
    print("Folder found")
    size = len(os.listdir(strPath))
    resize()
else:
    print("Folder not found")
