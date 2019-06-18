# import the necessary packages
import json
import os
import random
import cv2 as cv
import keras.backend as K
import numpy as np
import scipy.io
import pathlib
import xlwt 
from resnet_152 import resnet152_model

workbook = xlwt.Workbook()  
  
sheet = workbook.add_sheet("Sheet Name")
style_red = xlwt.easyxf('font: bold 1, color red;')
style_green = xlwt.easyxf('font: bold 1, color green;')
style_blue = xlwt.easyxf('font: color blue;')
style_bold = xlwt.easyxf('font: bold 1')


def make_prediction():
    global count_right
    print('Start processing image: {}'.format(file_name))
    bgr_img = cv.imread(file_name)
    bgr_img = cv.resize(bgr_img, (img_width, img_height), cv.INTER_CUBIC)
    rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
    rgb_img = np.expand_dims(rgb_img, 0)
    preds = model.predict(rgb_img)
    prob = np.max(preds)
    index=0
    for data in preds[0]:
      if data == prob:
        break
      index += 1
    print(preds)
    print("Predictes as " + table[index])
    if table[index] == folder_name:
      sheet.write(row, column, str(file_name)[len(image_dir + "/" + folder_name)+1:] + " %" + str(prob)[0:4], style_blue)
      count_right += 1
    else:
      sheet.write(row, column, str(file_name)[len(image_dir + "/" + folder_name)+1:] + " %" + str(prob)[0:4], style_red)

if __name__ == '__main__':
    print("Loading model")
    table = ["Cort","Fender","Gibson","Ibanez","Jackson"]
    model_weights_path = 'models/model.09-0.82.hdf5'
    img_width, img_height = 299, 299
    num_channels = 3
    num_classes = 5
    model = resnet152_model(img_height, img_width, num_channels, num_classes)
    model.load_weights(model_weights_path, by_name=True)
    
    image_dir = "Guitar_test_resized"
    file_name = ""
    if image_dir != "":  
      if len(os.listdir(image_dir)) == 0:
        print("Directory is empty")
      py = pathlib.Path().glob(image_dir + "/*")
      column=0
      avg=0
      for folder in sorted(py):
          folder_name = str(folder)[len(image_dir)+1:]
          sheet.write(0, column, folder_name, style_bold)
          print(folder_name)
          row=1
          total=0
          count_right=0
          py = pathlib.Path().glob(image_dir + "/" + folder_name + "/*")
          for file in sorted(py):
              file_name = str(file)
              #print(file_name)
              make_prediction()
              row += 1
              total += 1
          print(str(count_right) + " out of " + str(total) + " pictures were correctly estimated")
          print("Success rate " + str((count_right/total)*100)[:5] + "\n")
          sheet.write(row, column, "%" + str((count_right/total)*100)[:5], style_green)
          avg += (count_right/total)*100
          column += 1
      print("Average success rate " + str(avg/column)[:5])
      sheet.write(1, column, "Average success rate %" + str(avg/column)[:5], style_green)
      workbook.save("sample.xls")

    #class_id = np.argmax(preds)
    #text = ('Predict: {}, prob: {}'.format(class_names[class_id][0][0], prob))
    #results.append({'label': class_names[class_id][0][0], 'prob': '{:.4}'.format(prob)})
    #cv.imwrite('images/{}_out.png'.format(i), bgr_img)

    #print(results)
    #with open('results.json', 'w') as file:
    #    json.dump(results, file, indent=4)

    K.clear_session()