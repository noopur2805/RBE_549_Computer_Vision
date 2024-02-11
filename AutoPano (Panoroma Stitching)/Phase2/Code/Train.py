#!/usr/bin/env python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.optim import AdamW

from torchvision import transforms as tf
from Network.Network import Net
import cv2
import sys
import os
import numpy as np
import random
import skimage
import PIL
import os
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from Misc.MiscUtils import *
from Misc.DataUtils import *
from torchvision.transforms import ToTensor
import argparse
import shutil
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm


sys.dont_write_bytecode = True
if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

def plot_losses(losses,filename):
    # losses = [x['loss'] for x in history]
    plt.plot(losses, '-x')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Train Loss vs. No. of epochs')
    # plt.show()  
    # plt.savefig('/home/uthira/usivaraman_hw0/Phase2/Code/Results/Loss.png')
    plt.savefig(filename)
# def processData():

#    mainPath = "/home/usivaraman/CV/CV_P1/Phase2/Data/"
#    mPath= "/home/usivaraman/CV/CV_P1/Phase2/Data"
#    labelPath = "/home/usivaraman/CV/CV_P1/Phase2/Code/TxtFiles/"   

   
# #    if(dataType=="Train"):    
# #     ZipFileStrings= ["Train","Val"]
# #     numFilesList =[5000,1000]
# #    elif(dataType=="Test"):    
# #     ZipFileStrings= ["Test"]
# #     numFilesList =[1000]

# #    for i,ZipFileString in enumerate(ZipFileStrings):  
    
#     IPATH = mainPath + ZipFileString  +'/Img/'
#     PPATH = mainPath + ZipFileString  + '/Patch/' 
#     ZString = ZipFileString+".zip"
#     ZPATH = os.path.join(mPath , ZString)
#     PatchAPath= PPATH +"/PatchA/"
#     PatchBPath= PPATH +"/PatchB/"
#     # print("IPATH :",IPATH)
#     # print("PPATH :",PPATH)
#     # numFiles = numFilesList[i]
#     # print("ZipfileStringPath :",ZPATH)
#     # for i in range(1, numFiles+1):
#     #     with zipfile.ZipFile(ZPATH, 'r') as zfile:
#     #         data        = zfile.read(ZipFileString+'/'+str(i)+'.jpg')
#     #         img         = cv2.imdecode(np.frombuffer(data, np.uint8), 1)    
#     #         # cv2.imshow("Image",img)
#     #         cv2.imwrite(IPATH+str(i)+'.jpg', img)
    
#     # for i in range(1,numFiles+1):
#     #     print("i :",i)
#     #     warped_img, PatchA_corners, PatchB_corners, PatchA, PatchB, H4Pt = createPatch(cv2.imread(IPATH+str(i)+'.jpg'), patchSize,pert) 
#     #     patchA_img = os.path.join(PatchAPath,str(i)+".png")
#     #     cv2.imwrite(patchA_img,PatchA)
#     #     patchB_img = os.path.join(PatchBPath,str(i)+".png")
#     #     cv2.imwrite(patchB_img,PatchB)             
#     #     H4Pt = H4Pt.flatten(order = 'F')
#     #     h4_filename = labelPath + "/" + str(i) + ".csv"
#     #     np.savetxt(h4_filename, H4Pt, delimiter = ",")

#     transformImg=tf.Compose([tf.ToTensor(),
#                             tf.Normalize((0.5,0.5),(0.5,0.5), inplace = True)])
    
#     PatchA_list = os.listdir(PatchAPath)
#     PatchB_list = os.listdir(PatchBPath)
#     PatchA_path, PatchB_path = [], []
#     for i in range(len(PatchA_list)):
#         PatchA_path = os.path.join(PatchAPath,PatchA_list[i])
#         PatchB_path = os.path.join(PatchBPath,PatchB_list[i])
#         PatchA_path.append(PatchA_path)
#         PatchB_path.append(PatchB_path)

#     images1 = [cv2.imread(i,0) for i in images1_path]
#     images2 = images1.copy()
#     images2 = [cv2.imread(i,0) for i in images2_path]
#     trainsetA = np.array(images1)
#     trainsetB = np.array(images2)
#     X_train = []
#     Y_train = []
#     count = 0
#     for i in range(0,len(trainsetA)):
#     # for i in range(0,len(trainsetA)):
#         count+=1
#         print(count, end = "\r")
#         img1 = trainsetA[i]
#         img1 = np.expand_dims(img1, 2)
#         img2 = trainsetB[i]
#         img2 = np.expand_dims(img2, 2)
#         img = np.concatenate((img1, img2), axis = 2)
#         # print(img.shape)
#         Img = transformImg(img)
    

  
def GenerateBatch( TImagesPath1,TImagesPath2,VImagesPath1,VImagesPath2, TLabelPath, VLabelPath, ImageSize, MiniBatchSize):
    """
    Inputs:
    BasePath - Path to COCO folder without "/" at the end
    DirNamesTrain - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainCoordinates - Coordinatess corresponding to Train
    NOTE that TrainCoordinates can be replaced by Val/TestCoordinatess for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize - Size of the Image
    MiniBatchSize is the size of the MiniBatch
    Outputs:
    I1Batch - Batch of images
    CoordinatesBatch - Batch of coordinates
    
    """
    # print("INside /////////")
    transforms=tf.Compose([tf.ToTensor(),
                            tf.Normalize((0.5,0.5),(0.5,0.5), inplace = True)])
    TI1Batch = []
    TCoordinatesBatch = []
    VI1Batch = []
    VCoordinatesBatch = []

    # image1_directory_path = os.path.join("..","Data_ag")
    # image2_directory_path = os.path.join("..","Data_bg")
    if os.path.exists(TImagesPath1): 
        Timage1_list = os.listdir(TImagesPath1)
        # print("Timage1_list : ",Timage1_list)
    else:
        raise Exception ("Directory Image1 doesn't exist")
    if os.path.exists(TImagesPath2): 
        Timage2_list = os.listdir(TImagesPath2)
    else:
        raise Exception ("Directory Image2 doesn't exist")

    Timages1_path, Timages2_path = [], []
    for i in range(len(Timage1_list)):
        Timage1_path = TImagesPath1+'/'+Timage1_list[i]
        Timage2_path = TImagesPath2+'/'+Timage2_list[i]
        Timages1_path.append(Timage1_path)
        Timages2_path.append(Timage2_path)
        # print("Timage1_path",Timages1_path,'\n')/
        # if i == 10:
        #     break
    # print(os.listdir('../Data/Train/Patch/PatchA/'))
    # print(Timages1_path)
    Timages1 = [cv2.imread(i, 0) for i in Timages1_path]
    # Timages2 = images1.copy()
    # print("Timages1 : ",Timages1)
    Timages2 = [cv2.imread(i,0) for i in Timages2_path]
    TImageset1= Timages1
    TImageset2 = Timages2
    # print("TImageSet1 : ",TImageset1)

    if os.path.exists(VImagesPath1): 
        Vimage1_list = os.listdir(VImagesPath1)
    else:
        raise Exception ("Directory Image1 doesn't exist")
    if os.path.exists(VImagesPath2): 
        Vimage2_list = os.listdir(VImagesPath2)
    else:
        raise Exception ("Directory Image2 doesn't exist")

    Vimages1_path, Vimages2_path = [], []
    for i in range(len(Vimage1_list)):
        Vimage1_path = os.path.join(VImagesPath1,Vimage1_list[i])
        Vimage2_path = os.path.join(VImagesPath2,Vimage2_list[i])
        Vimages1_path.append(Vimage1_path)
        Vimages2_path.append(Vimage2_path)

    Vimages1 = [cv2.imread(i,0) for i in Vimages1_path]
    # Vimages2 = images1.copy()
    Vimages2 = [cv2.imread(i,0) for i in Vimages2_path]
    VImageset1=Vimages1
    VImageset2 = Vimages2
    # print("lennnnnnnnn")
    # # X_train = []
    # # Y_train = []
    # # count = 0
    # # for i in range(0,len(trainsetA)):
    # # # for i in range(0,len(trainsetA)):
    # #     count+=1
    # #     print(count, end = "\r")
    # #     img1 = trainsetA[i]
    # #     img1 = np.expand_dims(img1, 2)
    # #     img2 = trainsetB[i]
    # #     img2 = np.expand_dims(img2, 2)
    # #     img = np.concatenate((img1, img2), axis = 2)
    # #     # print(img.shape)
    # #     Img = transformImg(img)
    # #     X_train.append(Img)

    ImageNum = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        TIdx = random.randint(0, len(TImageset1)-1)
        VIdx = random.randint(0, len(VImageset1)-1)


        # print("Tdx :",TIdx)
        # TImageName1 = TImagesPath1 + str(TIdx+1) + ".png"
        # VImageName1 = VImagesPath1 + str(VIdx+1) + ".png"
        # TImageName2 = TImagesPath2 + str(TIdx+1) + ".png"
        # VImageName2 = VImagesPath2 + str(VIdx+1) + ".png"
        # print("TImageName1",TImageName1)
        # print("TImageName2",TImageName2)
        TImg1 = TImageset1[TIdx]
        TImg2 = TImageset2[TIdx]
        VImg1 = VImageset1[VIdx]
        VImg2 = VImageset2[VIdx]
        
        # print("TImageName1",TImg1.shape)
        # print("TImageName2",TImg2.shape)
        TImg1 = np.expand_dims(TImg1, 2)
        TImg2 = np.expand_dims(TImg2, 2)        
        TImg = np.concatenate((TImg1, TImg2), axis = 2)        
        TImg = transforms(TImg)

        VImg1 = np.expand_dims(VImg1, 2)
        VImg2 = np.expand_dims(VImg2, 2)        
        VImg = np.concatenate((VImg1, VImg2), axis = 2)  
        # VImg = np.concatenate((VImageName1, VImageName2), axis = 2)        
        VImg = transforms(VImg)
        
        ##########################################################
        # Add any standardization or data augmentation here!
        ##########################################################
        # TI1 = np.float32(np.load(TRandImageName,allow_pickle=True))
        Tlabel = np.genfromtxt(TLabelPath + str(TIdx+1) + '.csv', delimiter=',')
        TCoordinates = torch.from_numpy(Tlabel)
        # To_Tensor       = ToTensor()
        # I1              = To_Tensor(patch)

        # Append All Images and Mask
        # To_Tensor       = ToTensor()
        # TImg             = torch.tensor(VImg)
        TImg =  np.float32(TImg)
        TImg = torch.from_numpy(TImg)
        TI1Batch.append(TImg)
        TCoordinatesBatch.append(TCoordinates)

        # VI1 = np.float32(np.load(VRandImageName,allow_pickle=True))
        Vlabel = np.genfromtxt(VLabelPath + str(VIdx+1) + '.csv', delimiter=',')
        VCoordinates = torch.from_numpy(Vlabel)
        
        VImg =  np.float32(VImg)
        VImg = torch.from_numpy(VImg)
        # Append All Images and Mask
        VI1Batch.append(VImg)
        VCoordinatesBatch.append(torch.tensor(VCoordinates, dtype=torch.float32))
        ImageNum += 1
        # print("TI1BATCH :",TI1Batch)

    # TBATCH= TI1Batch,TCoordinatesBatch
    # VBATCH = VI1Batch,VCoordinatesBatch
    # print("TBATCH Size :",TBATCH.shape)

    return torch.stack(TI1Batch).to(device), torch.stack(TCoordinatesBatch).to(device)
    # return [],[]

# def GenerateBatch(TImagesPath1,TImagesPath2,VImagesPath1,VImagesPath2, TLabelPath, VLabelPath,, ImageSize, MiniBatchSize):
#     """
#     Inputs: 
#     TrainSet - Variable with Subfolder paths to train files
#     NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
#     TrainLabels - Labels corresponding to Train
#     NOTE that TrainLabels can be replaced by Val/TestLabels for generating batch corresponding to validation (held-out testing in this case)/testing
#     ImageSize is the Size of the Image
#     MiniBatchSize is the size of the MiniBatch
   
#     Outputs:
#     I1Batch - Batch of images
#     LabelBatch - Batch of one-hot encoded labels 
#     """
#     ransforms=tf.Compose([tf.ToTensor(),
#                             tf.Normalize((0.5,0.5),(0.5,0.5), inplace = True)])
#     TI1Batch = []
#     TCoordinatesBatch = []
#     VI1Batch = []
#     VCoordinatesBatch = []

#     # image1_directory_path = os.path.join("..","Data_ag")
#     # image2_directory_path = os.path.join("..","Data_bg")
#     if os.path.exists(TImagesPath1): 
#         Timage1_list = os.listdir(TImagesPath1)
#         # print("Timage1_list : ",Timage1_list)
#     else:
#         raise Exception ("Directory Image1 doesn't exist")
#     if os.path.exists(TImagesPath2): 
#         Timage2_list = os.listdir(TImagesPath2)
#     else:
#         raise Exception ("Directory Image2 doesn't exist")

#     Timages1_path, Timages2_path = [], []
#     for i in range(len(Timage1_list)):
#         Timage1_path = TImagesPath1+'/'+Timage1_list[i]
#         Timage2_path = TImagesPath2+'/'+Timage2_list[i]
#         Timages1_path.append(Timage1_path)
#         Timages2_path.append(Timage2_path)
#         # print("Timage1_path",Timages1_path,'\n')/
#         # if i == 10:
#         #     break
#     # print(os.listdir('../Data/Train/Patch/PatchA/'))
#     # print(Timages1_path)
#     Timages1 = [cv2.imread(i, 0) for i in Timages1_path]
#     # Timages2 = images1.copy()
#     # print("Timages1 : ",Timages1)
#     Timages2 = [cv2.imread(i,0) for i in Timages2_path]
#     TImageset1= Timages1
#     TImageset2 = Timages2
#     # print("TImageSet1 : ",TImageset1)

#     if os.path.exists(VImagesPath1): 
#         Vimage1_list = os.listdir(VImagesPath1)
#     else:
#         raise Exception ("Directory Image1 doesn't exist")
#     if os.path.exists(VImagesPath2): 
#         Vimage2_list = os.listdir(VImagesPath2)
#     else:
#         raise Exception ("Directory Image2 doesn't exist")

#     Vimages1_path, Vimages2_path = [], []
#     for i in range(len(Vimage1_list)):
#         Vimage1_path = os.path.join(VImagesPath1,Vimage1_list[i])
#         Vimage2_path = os.path.join(VImagesPath2,Vimage2_list[i])
#         Vimages1_path.append(Vimage1_path)
#         Vimages2_path.append(Vimage2_path)

#     Vimages1 = [cv2.imread(i,0) for i in Vimages1_path]
#     # Vimages2 = images1.copy()
#     Vimages2 = [cv2.imread(i,0) for i in Vimages2_path]
#     VImageset1=Vimages1
#     VImageset2 = Vimages2
#     # X_train = []
#     # Y_train = []
#     # count = 0
#     # for i in range(0,len(trainsetA)):
#     # # for i in range(0,len(trainsetA)):
#     #     count+=1
#     #     print(count, end = "\r")
#     #     img1 = trainsetA[i]
#     #     img1 = np.expand_dims(img1, 2)
#     #     img2 = trainsetB[i]
#     #     img2 = np.expand_dims(img2, 2)
#     #     img = np.concatenate((img1, img2), axis = 2)
#     #     # print(img.shape)
#     #     Img = transformImg(img)
#     #     X_train.append(Img)

#     for i in range(len(Timage1_list)):
#         # Generate random image
#         TIdx = i
#         VIdx = i


#         # print("Tdx :",TIdx)
#         # TImageName1 = TImagesPath1 + str(TIdx+1) + ".png"
#         # VImageName1 = VImagesPath1 + str(VIdx+1) + ".png"
#         # TImageName2 = TImagesPath2 + str(TIdx+1) + ".png"
#         # VImageName2 = VImagesPath2 + str(VIdx+1) + ".png"
#         # print("TImageName1",TImageName1)
#         # print("TImageName2",TImageName2)
#         TImg1 = TImageset1[TIdx]
#         TImg2 = TImageset2[TIdx]
#         VImg1 = VImageset1[VIdx]
#         VImg2 = VImageset2[VIdx]
        
#         # print("TImageName1",TImg1.shape)
#         # print("TImageName2",TImg2.shape)
#         TImg1 = np.expand_dims(TImg1, 2)
#         TImg2 = np.expand_dims(TImg2, 2)        
#         TImg = np.concatenate((TImg1, TImg2), axis = 2)        
#         TImg = transforms(TImg)

#         VImg1 = np.expand_dims(VImg1, 2)
#         VImg2 = np.expand_dims(VImg2, 2)        
#         VImg = np.concatenate((VImg1, VImg2), axis = 2)  
#         # VImg = np.concatenate((VImageName1, VImageName2), axis = 2)        
#         VImg = transforms(VImg)
        
#         ##########################################################
#         # Add any standardization or data augmentation here!
#         ##########################################################
#         # TI1 = np.float32(np.load(TRandImageName,allow_pickle=True))
#         Tlabel = np.genfromtxt(TLabelPath + str(TIdx+1) + '.csv', delimiter=',')
#         TCoordinates = torch.from_numpy(Tlabel)
#         # To_Tensor       = ToTensor()
#         # I1              = To_Tensor(patch)

#         # Append All Images and Mask
#         TI1Batch.append(TImg)
#         TCoordinatesBatch.append(TCoordinates)

#         # VI1 = np.float32(np.load(VRandImageName,allow_pickle=True))
#         Vlabel = np.genfromtxt(VLabelPath + str(VIdx+1) + '.csv', delimiter=',')
#         VCoordinates = torch.from_numpy(Vlabel)
        

#         # Append All Images and Mask
#         TI1Batch.append(VImg)
#         TCoordinatesBatch.append(VCoordinates)

#     BATCH= np.array(TI1Batch),np.array(TCoordinatesBatch)
#     # VBATCH = np.array(VI1Batch),np.array(VCoordinatesBatch)
#     trainset =
#     random_seed = 50
#     torch.manual_seed(random_seed)


#     val_size = int(0.2*len(TrainSet))
#     train_size = len(TrainSet) - val_size

#     train_ds, val_ds = random_split(TrainSet, [train_size, val_size])
#     batch_size = MiniBatchSize
#     train_loader = DataLoader(train_ds, batch_size, shuffle = True)
#     val_loader = DataLoader(val_ds, batch_size)

#     return train_loader, val_loader  


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print("Number of Epochs Training will run for " + str(NumEpochs))
    print("Factor of reduction in training data is " + str(DivTrain))
    print("Mini Batch Size " + str(MiniBatchSize))
    print("Number of Training Images " + str(NumTrainSamples))
    if LatestFile is not None:
        print("Loading latest checkpoint with the name " + LatestFile)


def TrainOperation(
    TImagesPath1,
    TImagesPath2,
    VImagesPath1, 
    VImagesPath2,
    TLabelPath, 
    VLabelPath,
    NumTrainSamples,
    ImageSize,
    NumEpochs,
    MiniBatchSize,
    SaveCheckPoint,
    CheckPointPath,
    DivTrain,
    LatestFile,    
    LogsPath,
    ModelType,
):
    """
    Inputs:
    ImgPH is the Input Image placeholder
    DirNamesTrain - Variable with Subfolder paths to train files
    TrainCoordinates - Coordinates corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    BasePath - Path to COCO folder without "/" at the end
    LogsPath - Path to save Tensorboard Logs
        ModelType - Supervised or Unsupervised Model
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """

    # Predict output with forward pass
    model = Net(2,128,128)
    model = model.to(device)
    # print("MOdel line")
    Optimizer = torch.optim.SGD(model.parameters(), lr = 0.0005, momentum = 0.9)
    

    # Tensorboard
    # Create a summary to monitor loss tensor
    Writer = SummaryWriter(LogsPath)

    if LatestFile is not None:
        CheckPoint = torch.load(CheckPointPath + LatestFile + ".ckpt")
        # Extract only numbers from the name
        StartEpoch = int("".join(c for c in LatestFile.split("a")[0] if c.isdigit()))
        model.load_state_dict(CheckPoint["model_state_dict"])
        print("Loaded latest checkpoint with the name " + LatestFile + "....")
    else:
        StartEpoch = 0
        print("New model initialized....")
    train =[]
    val =[]
    for Epochs in tqdm(range(StartEpoch, NumEpochs)):
    # for Epochs in range(0,1):
        TBatches =[]
        VBatches =[]
        NumIterationsPerEpoch = int(NumTrainSamples/ MiniBatchSize / DivTrain)
        for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
        # for PerEpochCounter in range(0,1):
            # print("Iteration Number :",PerEpochCounter)
            TBatch = GenerateBatch(TImagesPath1,TImagesPath2,VImagesPath1,VImagesPath2, TLabelPath, VLabelPath, ImageSize, MiniBatchSize)
            # TBatch = TBatch[0]
            # VBatch= VBatch[0]
            # print("LIne after generate batch")
            # Predict output with forward pass
            # print("TBATCH :",TBatch)
            LossThisBatch = model.training_step(TBatch)

            Optimizer.zero_grad()
            LossThisBatch.backward()
            Optimizer.step()

            
            TBatches.append(TBatch)
            # VBatches.append(VBatch)
            # print("Iteration Counter :",PerEpochCounter)
            # print("LossThisBatch :",LossThisBatch)
            # result = model.validation_step(TBatch)
            # model.epoch_end(Epochs*NumIterationsPerEpoch + PerEpochCounter, result)
        model.eval()
        Toutputs = [model.validation_step(batch) for batch in TBatches]
        # Voutputs = [model.validation_step(batch) for batch in VBatches]
        result = model.validation_epoch_end(Toutputs)
        #  result = evaluate(model, val_loader)
        # val_loss = model.validation_epoch_end(Voutputs)
        # result['train_loss'] = train_loss
        # result['val_loss'] = val_loss
        model.epoch_end(Epochs, result)
        train.append(result['train_loss'])
        # val.append(val_loss)
        # print("loss done")
        # Tensorboard
        # Writer.add_scalar('LossEveryEpoch', train_result["loss"], Epochs)
        # Writer.add_scalar('Accuracy', train_result["acc"], Epochs)
        # # If you don't flush the tensorboard doesn't update until a lot of iterations!
        # Writer.flush()

        # Save model every epoch
        SaveName = CheckPointPath + str(Epochs) + "model.ckpt"
        torch.save(
            {
                "epoch": Epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": Optimizer.state_dict(),
                "loss": LossThisBatch,
            },
            SaveName,
        )
        print("\n" + SaveName + " Model Saved...")

    return train


def train_sup():
    """
    Inputs:
    # None
    # Outputs:
    # Runs the Training and testing code based on the Flag
    #"""
    # Parse Command Line arguments
    print("Started Sup Train")
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--BasePath",
        default="../Phase2/Code/Data",
        help="Base path of images, Default:/home/lening/workspace/rbe549/YourDirectoryID_p1/Phase2/Data",
    )
    Parser.add_argument(
        "--TImagesPath1",
        default="../Phase2/Code/Data/Train/Patch/PatchA",
        help="Base path of images, Default:/home/lening/workspace/rbe549/YourDirectoryID_p1/Phase2/Data",
    )
    Parser.add_argument(
        "--TImagesPath2",
        default="../Phase2/Code/Data/Train/Patch/PatchB",
        help="Base path of images, Default:/home/lening/workspace/rbe549/YourDirectoryID_p1/Phase2/Data",
    )
    Parser.add_argument(
        "--VImagesPath1",
        default="../Phase2/Code/Data/Val/Patch/PatchA",
        help="Base path of images, Default:/home/lening/workspace/rbe549/YourDirectoryID_p1/Phase2/Data",
    )
    Parser.add_argument(
        "--VImagesPath2",
        default="../Phase2/Code/Data/Val/Patch/PatchB",
        help="Base path of images, Default:/home/lening/workspace/rbe549/YourDirectoryID_p1/Phase2/Data",
    )
    Parser.add_argument(
        "--TLabelPath",
        default="../Phase2/Code/TxtFiles/Train/",
        help="Base path of images, Default:/home/lening/workspace/rbe549/YourDirectoryID_p1/Phase2/Data",
    )
    Parser.add_argument(
        "--VLabelPath",
        default="../Phase2/Code/TxtFiles/Val/",
        help="Base path of images, Default:/home/lening/workspace/rbe549/YourDirectoryID_p1/Phase2/Data",
    )
    Parser.add_argument(
        "--CheckPointPath",
        default="../Phase2/Code/CheckPoints/",
        help="Path to save Checkpoints, Default: ../Checkpoints/",
    )

    Parser.add_argument(
        "--ModelType",
        default="Unsup",
        help="Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup",
    )
    Parser.add_argument(
        "--NumEpochs",
        type=int,
        default=1,
        help="Number of Epochs to Train for, Default:50",
    )
    Parser.add_argument(
        "--DivTrain",
        type=int,
        default=1,
        help="Factor to reduce Train data by per epoch, Default:1",
    )
    Parser.add_argument(
        "--MiniBatchSize",
        type=int,
        default=64,
        help="Size of the MiniBatch to use, Default:1",
    )
    Parser.add_argument(
        "--LoadCheckPoint",
        type=int,
        default=0,
        help="Load Model from latest Checkpoint from CheckPointsPath?, Default:0",
    )
    Parser.add_argument(
        "--LogsPath",
        default="Logs/",
        help="Path to save Logs for Tensorboard, Default=Logs/",
    )


    # processData()
    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    TImagesPath1 = Args.TImagesPath1
    VImagesPath1 = Args.VImagesPath1
    TImagesPath2 = Args.TImagesPath2
    VImagesPath2 = Args.VImagesPath2
    TLabelPath = Args.TLabelPath
    VLabelPath = Args.VLabelPath
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    ModelType = Args.ModelType
    # print("..........")
    # Setup all needed parameters including file reading
    (
        DirNamesTrain,
        SaveCheckPoint,
        ImageSize,
        NumTrainSamples,
        TrainCoordinates,
        NumClasses,
    ) = SetupAll(BasePath, CheckPointPath)

    # Find Latest Checkpoint File
    if LoadCheckPoint == 1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None

    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)
    # print(".dsdsdcxcxc")
    train = TrainOperation(
        TImagesPath1,
        TImagesPath2,
        VImagesPath1, 
        VImagesPath2,
        TLabelPath, 
        VLabelPath,
        NumTrainSamples,
        ImageSize,
        NumEpochs,
        MiniBatchSize,
        SaveCheckPoint,
        CheckPointPath,
        DivTrain,
        LatestFile,    
        LogsPath,
        ModelType,
    )
    plot_losses(train,'/home/usivaraman/CV/CV_P1/Phase2/Code/Results/Train_Loss.png')
    # plot_losses(val,'/home/uthira/usivaraman_hw0/Phase2/Code/Results/Val_Loss.png')

# if __name__ == "__main__":
#     # print("STARTINGGGGGG")
#     main()
