#!/usr/bin/python
from PIL import Image
import os, sys
from console_progressbar import ProgressBar

path = "test"

if not os.path.exists(path+ "_resized"):
    os.mkdir(path + "_resized")

def resize():
    for folders in os.listdir(path):
        if not os.path.exists(path+ "_resized/" + folders):
            os.mkdir(path + "_resized/" + folders)
        print(folders)
        pb = ProgressBar(total=len(os.listdir(path+"/"+folders)), decimals=1, length=30, fill='X', zfill='-')
        i=0
        for files in os.listdir(path+"/"+folders):
                im = Image.open(path+"/"+folders+"/"+files)
                f, e = os.path.splitext(path+"/"+folders+"/"+files)
                imResize = im.resize((224,224), Image.ANTIALIAS)
                imResize.save(path + "_resized/" + folders + "/" + files, 'PNG', quality=90)
                pb.print_progress_bar(i)
                print(' ')
                i=i+1
resize()
print("---Finished---")
