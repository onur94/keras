import keras
import numpy as np
import json
import os
import random
import cv2 as cv
import keras.backend as K
import scipy.io
import pathlib
import xlwt 
from keras.applications import ResNet50, densenet, mobilenet_v2, xception, inception_v3
from keras.models import Sequential, Model, load_model
from keras.layers import Activation, Dropout, Flatten, Dense, AveragePooling2D, GlobalAveragePooling2D
from keras.optimizers import SGD, Adam
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from PlotConfusionMatrix import plot_confusion_matrix
from console_progressbar import ProgressBar
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import requests

img_width, img_height = 224,224
num_channels = 3
train_data = 'skin_cancer_malignant_benign/train'
valid_data = 'skin_cancer_malignant_benign/validation'
test_data  = 'skin_cancer_malignant_benign/test'
table = ["benign","malignant"]
dataset = "Skin Cancer"
model_name = "InceptionV3"
num_classes = 2
num_train_samples = 2437
num_valid_samples = 200
num_test_samples = 660
batch_size = 32
num_epochs = 30
patience = 50
verbose = 1

TOKEN = ""

workbook = xlwt.Workbook()  
  
sheet = workbook.add_sheet("Sheet Name")
style_red = xlwt.easyxf('font: bold 1, color red;')
style_green = xlwt.easyxf('font: bold 1, color green;')
style_blue = xlwt.easyxf('font: color blue;')
style_bold = xlwt.easyxf('font: bold 1')

def make_prediction():
    global count_right
    #print('Start processing image: {}'.format(file_name))
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
    #print(preds)
    #print("Predictes as " + table[index])
    if table[index] == folder_name:
      sheet.write(row, column, str(file_name)[len(image_dir + "/" + folder_name)+1:] + " %" + str(prob)[0:4], style_blue)
      count_right += 1
    else:
      sheet.write(row, column, str(file_name)[len(image_dir + "/" + folder_name)+1:] + " %" + str(prob)[0:4], style_red)

if __name__ == '__main__':
    # build a classifier model
    #model = resnet50_model(img_width, img_height, num_channels, num_classes)
    #model = densenet121_model(img_rows=img_width, img_cols=img_height, color_type=num_channels, num_classes=num_classes)
    
    #x = base_model.output
    #predictions = Dense(num_classes, activation='softmax')(x)
    #model = Model(inputs=base_model.input, outputs=predictions)
    
    base_model = inception_v3.InceptionV3(input_shape=(img_width, img_height, 3), weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    #model.summary()

    # prepare data augmentation configuration
    train_data_gen = ImageDataGenerator(rotation_range=20.,
                                        width_shift_range=0.1,
                                        height_shift_range=0.1,
                                        zoom_range=0.2,
                                        horizontal_flip=True)
    valid_data_gen = ImageDataGenerator()
    # callbacks
    tensor_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    log_file_path = 'logs/training.log'
    csv_logger = CSVLogger(log_file_path, append=False)
    early_stop = EarlyStopping('val_acc', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_acc', factor=0.1, patience=int(patience / 4), verbose=1)
    trained_models_path = 'models/model'
    model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, monitor='val_acc', verbose=1, save_best_only=True)
    callbacks = [tensor_board, csv_logger, early_stop, reduce_lr]

    # generators
    train_generator = train_data_gen.flow_from_directory(train_data, (img_width, img_height), batch_size=batch_size,
                                                         class_mode='categorical')
    valid_generator = valid_data_gen.flow_from_directory(valid_data, (img_width, img_height), batch_size=batch_size,
                                                         class_mode='categorical')

    # fine tune the model
    H = model.fit_generator(
          train_generator,
          steps_per_epoch=num_train_samples / batch_size,
          validation_data=valid_generator,
          validation_steps=num_valid_samples / batch_size,
          epochs=num_epochs,
          callbacks=callbacks,
          verbose=verbose)
    
    print("Loss=" + str(H.history["loss"]))
    print("Val_Loss=" + str(H.history["val_loss"]))
    print("Acc=" + str(H.history["acc"]))
    print("Val_Acc=" + str(H.history["val_acc"]))
    N = num_epochs
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    #plt.savefig('Loss-acc.png')
    
    print("Loading model")
    cm = np.zeros(shape=(2,2))
    pb = ProgressBar(num_test_samples, decimals=1, length=30, fill='=', zfill='.')
    
    image_dir = test_data
    file_name = ""
    if image_dir != "":  
      if len(os.listdir(image_dir)) == 0:
        print("Directory is empty")
      py = pathlib.Path().glob(image_dir + "/*")
      column=0
      avg=0
      counter=0
      for folder in sorted(py):
          folder_name = str(folder)[len(image_dir)+1:]
          sheet.write(0, column, folder_name, style_bold)
          print("Predicting images in folder: " + str(folder_name))
          row=1
          total=0
          count_right=0
          py = pathlib.Path().glob(image_dir + "/" + folder_name + "/*")
          for file in sorted(py):
              file_name = str(file)
              #print(file_name)
              pb.print_progress_bar(counter)
              counter += 1
              make_prediction()
              row += 1
              total += 1
          print(str(count_right) + " out of " + str(total) + " pictures were correctly estimated")
          print("Success rate " + str((count_right/total)*100)[:5])
          sheet.write(row, column, "%" + str((count_right/total)*100)[:5], style_green)
          avg += (count_right/total)*100
          cm[column][column] = count_right
          if column == 0:
            cm[column][column+1] = total-count_right
          elif column == 1:
            cm[column][column-1] = total-count_right
          column += 1
      sheet.write(1, column, "Average success rate %" + str(avg/column)[:5], style_green)
      #workbook.save("sample.xls")

    K.clear_session()
    print(cm)
    total = sum(sum(cm))
    acc = (cm[0, 0] + cm[1, 1]) / total
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

    # show the confusion matrix, accuracy, sensitivity, and specificity
    print("Acc: {:.4f}".format(acc))
    print("Sensitivity: {:.4f}".format(sensitivity))
    print("Specificity: {:.4f}".format(specificity))
    target_names = ['0', '1']
    plot_confusion_matrix(cm, target_names, True)
    
    MSG  = "Dataset: " + dataset + "\n"
    MSG += "Model: " + model_name + "\n"
    MSG += "N.of train: " + str(num_train_samples) + "\n"
    MSG += "N.of valid: " + str(num_valid_samples) + "\n"
    MSG += "N.of test : " + str(num_test_samples) + "\n"
    MSG += "Train;\nN.of epoch: " + str(num_epochs) + "\nVal_Acc=" + str(np.max(H.history["val_acc"]))[:6] + "\n"
    MSG += "Test;\n"
    MSG += "Acc: {:.4f}".format(acc) + "\n" + "Sensitivity: {:.4f}".format(sensitivity) + "\n" + "Specificity: {:.4f}".format(specificity)
    req = requests.post('https://api.rpinotify.it/message/' + TOKEN + '/', data={'text': MSG})
