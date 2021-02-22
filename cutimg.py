import numpy as np
from PIL import Image
import os
import shutil
import random

def cutimg(imgpath,imgsave,size=256):
    dirlist = os.listdir(imgpath)
    for img_name in dirlist:
        print(img_name)
        img = Image.open(imgpath+'/'+img_name)
        h,w = img.size
        num = 0
        for i in range(0,w,size):
            if w-i <= size:
                continue
            for j in range(0,h,size):
                if h-j<= size:
                    continue
                img = np.array(img)
                tmp_img = img[i:i+size,j:j+size,:]
                tmp_img = Image.fromarray(tmp_img)
                tmp_img.save(imgsave+'/'+ img_name+'_'+ str(num)+'.tif')
                num+=1
        print(num)

def cutlabel(labelpath, labelsave, size=256):
    dirlist = os.listdir(labelpath)
    for img_name in dirlist:
        print(img_name)
        img = Image.open(labelpath + '/' + img_name)
        h, w = img.size
        num = 0
        for i in range(0, w, size):
            if w - i <= size:
                continue
            for j in range(0, h, size):
                if h - j <= size:
                    continue
                img = np.array(img)
                tmp_img = img[i:i + size, j:j + size, :]
                tmp_img = Image.fromarray(tmp_img)
                tmp_img.save(labelsave + '/' + img_name + '_' + str(num) + '.tif')
                #print(labelsave + '/' + img_name + '_' + str(num) + '.tif')
                num += 1
        print(num)


def select(path):
    name =[]
    dirlist = os.listdir(path)
    for img_name in dirlist:
        label = Image.open(path + '/' + img_name)
        label = np.array(label)
        max= np.max(label)
        if max>0:
            name.append(img_name)
    return name

def moveimg(imgsave,labelsave,imgselect,labelselect,names):
    for name in names:
        shutil.copy(imgsave+'/'+name,imgselect+'/'+name)
        shutil.copy(labelsave+'/'+name,labelselect+'/'+name)

def processlabel(path):
    dirlist = os.listdir(path)
    for img_name in dirlist:
        label = Image.open(path + '/' + img_name)
        label = np.array(label)
        label[label < 255] = 0
        label = Image.fromarray(label)
        label.save(path + '/' + img_name)

def randomselect2val(img,label,img2val,label2val):
    dirlist = os.listdir(img)
    rad = random.sample(dirlist, 30)
    for img_name in rad:
        shutil.move(img+'/'+img_name,img2val+'/'+img_name)
        shutil.move(label + '/' + img_name, label2val + '/' + img_name)

if __name__ == '__main__':
    imgpath = '/home/qzp/Photovoltaic_power_station/光伏数据集2/train'
    labelpath = '/home/qzp/Photovoltaic_power_station/光伏数据集2/label'
    imgsave = '/home/qzp/Photovoltaic_power_station/光伏数据集2/imgcut'
    labelsave = '/home/qzp/Photovoltaic_power_station/光伏数据集2/labelcut'
    imgselect = '/home/qzp/Photovoltaic_power_station/光伏数据集2/imgselect'
    labelselect = '/home/qzp/Photovoltaic_power_station/光伏数据集2/labelselect'
    size = 256
    #cutimg(imgpath,imgsave,size=size)
    #cutlabel(labelpath, labelsave, size=size)
    # processlabel(labelsave)
    # name = select(labelsave)
    # print(len(name))
    # moveimg(imgsave,labelsave,imgselect,labelselect,name)
    img = '/home/qzp/Photovoltaic_power_station/DATA/ps/train/img'
    label = '/home/qzp/Photovoltaic_power_station/DATA/ps/train/label'
    img2val = '/home/qzp/Photovoltaic_power_station/DATA/ps/val/img'
    label2val = '/home/qzp/Photovoltaic_power_station/DATA/ps/val/label'
    randomselect2val(img,label,img2val,label2val)



    # img = Image.open('/home/qzp/Photovoltaic_power_station/光伏数据集2/labelselect/1.tif_3.tif')
    # img = np.array(img)
    # print(img)
    # img[img<255]=0
    # print(img)