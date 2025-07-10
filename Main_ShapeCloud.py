#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image, ImageOps,ImageFilter
import torch
import numpy as np
from torch import nn
import torchvision
import torch.nn.functional as F
from scipy import ndimage
import os
import cv2
from scipy.spatial import distance as distscipy
import time
import glob
import torchvision.transforms as transforms
import math
from tqdm import tqdm, trange
from natsort import natsorted 
import random
from gumbel_sinkhorn import my_sinkhorn_ops
from scipy.optimize import linear_sum_assignment
import ast


# In[2]:


def preprocess(img,threshold=0.5, directory="", tag = "", resolutionP=0):
    img = ImageOps.grayscale(img)
    img_with_border = img
    obj = np.array(img_with_border.resize((resolutionP, resolutionP)))
    obj = torch.tensor(obj, dtype=torch.float32) /255
    obj = torch.reshape(obj, (1, 1, resolutionP, resolutionP))
    pic = torchvision.transforms.functional.to_pil_image(obj.reshape(resolutionP, resolutionP))
    return obj


# In[3]:


ordernumber='animal'
datasetnumber='4'
objposition='Dataset/'+ordernumber+'/'+datasetnumber+'/image'+'/*.png'
holeposition='Dataset/'+ordernumber+'/oursbackground-0.png'
with open('Dataset/'+ordernumber+'/'+datasetnumber+'/ordervalue.txt', 'r') as f:
    ordervalue = f.read()


# # Processing Data

# In[4]:


recordvalue = []
objectnumber = int(len(glob.glob(objposition)))
resolutionF = 300
HolePil = Image.open(holeposition)
Hole = cv2.cvtColor(np.asarray(HolePil), cv2.COLOR_RGB2BGR)
HoleGray = cv2.cvtColor(Hole, cv2.COLOR_BGR2GRAY)
HoleGray = cv2.resize(HoleGray, (resolutionF,resolutionF), interpolation=cv2.INTER_AREA)

outret, outthresh = cv2.threshold(HoleGray, 0, 255, cv2.THRESH_BINARY) 

outHoleGray = 255*np.ones((HoleGray.shape[0],HoleGray.shape[1]), dtype=np.uint8)-outthresh
outdist = cv2.distanceTransform(outHoleGray, cv2.DIST_L2, 5)
distouthole_distance = torch.tensor(outdist).cuda()
distouthole_distance =  distouthole_distance.reshape(HoleGray.shape[0],HoleGray.shape[1])


HoleGray2 = cv2.cvtColor(np.asarray(HolePil), cv2.COLOR_RGBA2BGRA)
HoleGray2 = cv2.resize(HoleGray2, (resolutionF,resolutionF), interpolation=cv2.INTER_AREA)
HoleGray2[:, :, 3] = 0

recordtest = 0 

recordvaluewidth=[]
    
root_directory = "data"
ObjImage_list= torch.tensor([])
ObjDilateImage_list= torch.tensor([])
ObjBImage_list= torch.tensor([])
ObjBdistanceImage_list= torch.tensor([])
loader = transforms.Compose([transforms.ToTensor()])

countobject= 0 
for filename in natsorted(glob.glob(objposition)):
    objR_pic = Image.open(filename)
    objR_pic = objR_pic.convert("RGBA")
    sourcew,sourceh=objR_pic.size
    
    Finalresolution = ((resolutionF/(math.sqrt(sourcew*sourcew+sourceh*sourceh))))    
    objR_pic = objR_pic.resize((int(sourcew*Finalresolution),int(sourceh*Finalresolution)))
    
    
    Finalpic = Image.new("RGBA", (resolutionF, resolutionF))
    Finalpic.paste(objR_pic, (0, 0),objR_pic)
    objR_pic = Finalpic
    
    
    
    #--blankeage
    objL_picA = objR_pic.copy()
    objw,objh=objR_pic.size
    for x in range(objw):
        for y in range(objh):
            
            R,G,B,A = objR_pic.getpixel((x,y))
            
            if A<255:
                objL_picA.putpixel((x,y),(0,0,0,0))
                objR_pic.putpixel((x,y),(R,G,B,0))
            else:
                objL_picA.putpixel((x,y),(255,255,255,255))
                objR_pic.putpixel((x,y),(R,G,B,255))
    objB_pic = objL_picA.filter(ImageFilter.MaxFilter(3))
    for i in range(3) :
        objB_pic = objB_pic.filter(ImageFilter.MaxFilter(3))
    objB_pic.paste(objR_pic, (0, 0),objR_pic)
    
    
    
    objBBinary_pic = objB_pic.convert('L') 
    objw,objh=objB_pic.size
    for x in range(objw):
        for y in range(objh):
            
            R,G,B,A = objB_pic.getpixel((x,y))
            if A<255:
                objBBinary_pic.putpixel((x,y),0)
                
            else:
                objBBinary_pic.putpixel((x,y),255)
    objDilate_pic = objBBinary_pic
    #--endblankeage
    
    
    
    obj = loader(objR_pic).float().unsqueeze(0)
    ObjImage_list = torch.cat(([ObjImage_list,obj.unsqueeze(0)]),0) 
    
    objL_pic = objR_pic.convert('L') 
    objw,objh=objR_pic.size
    for x in range(objw):
        for y in range(objh):
            
            R,G,B,A = objR_pic.getpixel((x,y))
            if A<255:
                objL_pic.putpixel((x,y),0)
                
            else:
                objL_pic.putpixel((x,y),255)
    objL_picF=objL_pic
    
    retobj, threshobj = cv2.threshold(np.asarray(objL_pic), 0, 255, cv2.THRESH_BINARY) 
    distobj = cv2.distanceTransform(threshobj, cv2.DIST_L2, 5)
    
    
    
    distobj_pil = torch.tensor(distobj)#Image.fromarray(distobj)
    objL_picFdistance = distobj_pil
    ret, thresh = cv2.threshold(HoleGray, 0, 255, cv2.THRESH_BINARY) 
    dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5) 
    
    disthole_distance = torch.tensor(dist).cuda()
    
    
    
    
    H, L = dist.shape
    Height=0
    Width=0
    values=0
    distvalues = 0
    pixeltotal = 0 
    for h in range(H):
        for l in range(L):
            color_1 = dist[h][l]
            if color_1 >= distvalues:
                Height=h
                Width=l
                values=dist[h][l]
                distvalues = color_1
             
                
                
                
    if Height+1==resolutionF and Width+1==resolutionF:
        
        x,y = np.where(outthresh ==255)
        i = np.random.randint(len(x))
        Height=x[i]
        Width=y[i]
    
    
    
    
    ourdata = ast.literal_eval(ordervalue)  
    valuesF = resolutionF*math.sqrt(ourdata[countobject])*0.3
    values = valuesF 
    
    
#--------end 20241008 tvcg
   
    
    
    recordvalue.append([((resolutionF/2)-Width)/(resolutionF/2),((resolutionF/2)-Height)/(resolutionF/2),(resolutionF/2)/math.sqrt(math.pow(int(valuesF)*2,2)/2)])#250/math.sqrt(math.pow(int(values)*2,2)/2)
    cv2.circle(HoleGray, (Width,Height), int(valuesF), (0,0,0), -1)
    cv2.circle(HoleGray2, (Width,Height), int(valuesF), (102,102,255-countobject*6,255), -1)
    cv2.imwrite("HoleGray.png",HoleGray)
    cv2.imwrite("HoleGray2.png",HoleGray2)
    

    countobject= countobject +1
    recordvaluewidth.append(math.sqrt(math.pow(int(valuesF)*2,2)/2))
    
    
    
    
    objL_picF = loader(objL_picF).float()
    objL_picFdistance = objL_picFdistance.float().unsqueeze(0)
    obj_Dilate = loader(objDilate_pic).float()#
    
    objL_picF = objL_picF#.to(device)    
    ObjBImage_list = torch.cat([ObjBImage_list,objL_picF.unsqueeze(0)],0)
    ObjBdistanceImage_list = torch.cat([ObjBdistanceImage_list,objL_picFdistance.unsqueeze(0)],0)
    ObjDilateImage_list = torch.cat([ObjDilateImage_list,obj_Dilate.unsqueeze(0)],0) 
hole = preprocess(HolePil, directory=root_directory, tag="hole",resolutionP=resolutionF)


# In[5]:


class PackNet(nn.Module):
        
    def __init__(self, num_items, initial_value, image_resolution, deformation_resolution,image):
        super().__init__()
        self.num_items = num_items
        self.image = image
        
        
        self.resolution = image_resolution
        self.deformation_resolution = deformation_resolution
        
        self.deformation_offset =  nn.Parameter(torch.zeros(num_items, self.deformation_resolution, self.deformation_resolution,2), requires_grad=True)#deformation_resolution
        
        self.initial_paramangle = torch.tensor(torch.zeros(num_items), requires_grad=False).float().to('cuda')
        self.parameters_offsetangle= nn.Parameter((torch.zeros_like(self.initial_paramangle).float()+1e-4), requires_grad=True)
        
        
        self.initial_paramscore = torch.tensor((1/1.5)/np.asarray(initial_value)[:,2], requires_grad=False).float().to('cuda')
        self.parameters_offsetscorex = nn.Parameter((torch.ones_like(self.initial_paramscore).float()), requires_grad=True)#nn.Parameter(torch.zeros_like(self.initial_paramscore).float(), requires_grad=True)

        
        
        self.initial_paramtransx = torch.tensor(-1*np.asarray(initial_value)[:,0], requires_grad=False).float().to('cuda')
        self.initial_paramtransy = torch.tensor(-1*np.asarray(initial_value)[:,1], requires_grad=False).float().to('cuda')
        self.parameters_offsettransx = nn.Parameter(torch.zeros_like(self.initial_paramtransx).float()+1e-4, requires_grad=True)
        self.parameters_offsettransy = nn.Parameter(torch.zeros_like(self.initial_paramtransy).float()+1e-4, requires_grad=True)
       
       
        
        self.RF_Source = torch.tensor(torch.zeros(num_items), requires_grad=False).float().to('cuda')
        self.SFX_Source = torch.tensor(torch.ones(num_items), requires_grad=False).float().to('cuda')
        self.TFX_Source = torch.tensor(torch.zeros(num_items), requires_grad=False).float().to('cuda')
        self.TFY_Source = torch.tensor(torch.zeros(num_items), requires_grad=False).float().to('cuda')
        
        
        self.FCT = nn.Sequential(
            nn.Linear(num_items,num_items) ,
            nn.ReLU(),
            nn.Linear(num_items, num_items) ,
            nn.Tanh(),
        )
        self.FCT[2].weight.data.zero_()
        self.FCT[2].bias.data.copy_(torch.zeros(num_items))
        self.FCS = nn.Sequential(
            nn.Linear(num_items,num_items) ,
            nn.ReLU(),
            nn.Linear(num_items, 1) ,
        )
        self.FCS[2].weight.data.zero_()
        self.FCS[2].bias.data.copy_(torch.ones(1))
        self.FCR = nn.Sequential(
            nn.Linear(num_items,num_items) ,
            nn.ReLU(),
            
            nn.Linear(num_items, num_items) ,
            #nn.Sigmoid(),
            nn.Tanh(),
        )
        self.FCR[2].weight.data.zero_()
        self.FCR[2].bias.data.copy_(torch.zeros(num_items))
        
                
        self.log_alpha = nn.Parameter(torch.rand((1, num_items, num_items), requires_grad = True, device="cuda"))
        self.temperature = 0.9
        self.noise_factor= 0.01
        self.n_iter_sinkhorn = 20
        
    
   
    def source_parameters(self,ObjBImage):
        num=ObjBImage.size()[0]
        Soinit=nn.ReLU()(self.FCS(self.parameters_offsetscorex)[0])*torch.ones(num).cuda()
        SoX = (Soinit)*(self.initial_paramscore)
        
        
        RF = (self.FCR((self.parameters_offsetangle)))
        
        SFX =  SoX
        
        
        TFX = self.FCT((self.parameters_offsettransx))+torch.tensor(self.initial_paramtransx)
        TFY = self.FCT((self.parameters_offsettransy))+torch.tensor(self.initial_paramtransy)#self.FC
        self.RF_Source = RF.clone().detach()
        self.SFX_Source = SFX.clone().detach()
        self.TFX_Source = TFX.clone().detach()
        self.TFY_Source = TFY.clone().detach()
        
        
        return RF,SFX,TFX,TFY
    def source_parameters_NoMLP(self,ObjBImage):
        num=ObjBImage.size()[0]
        Soinit=nn.ReLU()(torch.clamp(((self.parameters_offsetscorex)[0]),min=1e-2))*torch.ones(num).cuda()
        SoX = (Soinit)*(self.initial_paramscore)
        
        RF = nn.Tanh()((self.parameters_offsetangle))
        
        SFX =  SoX
        
        TFX = nn.Tanh()((self.parameters_offsettransx))+torch.tensor(self.initial_paramtransx)
        TFY = nn.Tanh()((self.parameters_offsettransy))+torch.tensor(self.initial_paramtransy)
        self.RF_Source = RF.clone().detach()
        self.SFX_Source = SFX.clone().detach()
        self.TFX_Source = TFX.clone().detach()
        self.TFY_Source = TFY.clone().detach()
        
        return RF,SFX,TFX,TFY
    def theta_matrix(self,RF,SFX,TFX,TFY):
        
        ######################################################################
        initial_paramR = self.initial_paramangle+math.pi*(RF)
        
        row1 =torch.stack([torch.cos(initial_paramR).float().cuda(),
                     -torch.sin(initial_paramR).float().cuda(),
                     torch.zeros(self.num_items).float().cuda()],dim=1)
        row2 =torch.stack([torch.sin(initial_paramR).float().cuda(),
                     torch.cos(initial_paramR).float().cuda(),
                     torch.zeros(self.num_items).float().cuda()],dim=1)
        row3 =torch.stack([torch.zeros(self.num_items).float().cuda(),
                     torch.zeros(self.num_items).float().cuda(),
                     torch.ones(self.num_items).float().cuda()],dim=1)
        rowcombineR=torch.stack([row1, row2,row3],dim=1).resize(self.num_items,3,3)
        angle_theta = torch.linalg.inv(torch.cat([rowcombineR], dim=0))[:,0:2,:]
        ######################################################################
        ######################################################################
        initial_paramSXP = SFX
        initial_paramSXP = initial_paramSXP
        initial_paramSX = initial_paramSXP
        initial_paramSYP = SFX
        initial_paramSYP = initial_paramSYP
        initial_paramSY = initial_paramSYP
        row1 =torch.stack([initial_paramSX.float().cuda(),
                     torch.zeros(self.num_items).float().cuda(),
                     torch.zeros(self.num_items).float().cuda()],dim=1)
        row2 =torch.stack([torch.zeros(self.num_items).float().cuda(),
                     initial_paramSY.float().cuda(),
                     torch.zeros(self.num_items).float().cuda()],dim=1)
        row3 =torch.stack([torch.zeros(self.num_items).float().cuda(),
                     torch.zeros(self.num_items).float().cuda(),
                     torch.ones(self.num_items).float().cuda()],dim=1)
        rowcombineS=torch.stack([row1, row2,row3],dim=1).resize(self.num_items,3,3)
        scale_theta = torch.linalg.inv(torch.cat([rowcombineS], dim=0))[:,0:2,:]
        ######################################################################
        ######################################################################
        
        initial_paramTXP = torch.clamp((TFX) , min=-1.0,max=1.0)
        
        initial_paramTX = initial_paramTXP
    
        initial_paramTYP =torch.clamp((TFY) , min=-1.0,max=1.0)
        initial_paramTY = initial_paramTYP
        row1 =torch.stack([torch.ones(self.num_items).float().cuda(),
                     torch.zeros(self.num_items).float().cuda(),
                     initial_paramTX.float().cuda()],dim=1)
        row2 =torch.stack([torch.zeros(self.num_items).float().cuda(),
                     torch.ones(self.num_items).float().cuda(),
                     initial_paramTY.float().cuda()],dim=1)
        row3 =torch.stack([torch.zeros(self.num_items).float().cuda(),
                     torch.zeros(self.num_items).float().cuda(),
                     torch.ones(self.num_items).float().cuda()],dim=1)
        rowcombineT=torch.stack([row1, row2,row3],dim=1).resize(self.num_items,3,3)
        trans_theta = torch.linalg.inv(torch.cat([rowcombineT], dim=0))[:,0:2,:]
        ######################################################################
        a=torch.ones((3, self.deformation_resolution, self.deformation_resolution, 2))*0.0
        b=torch.ones((self.num_items-3, self.deformation_resolution, self.deformation_resolution, 2))*0.05
        c=(torch.cat([a,b])).cuda()
        deform_theta= torch.tanh(self.deformation_offset)*c
        
        rowcombineX= torch.matmul(torch.matmul(rowcombineT,rowcombineS),rowcombineR )
        combine_theta = torch.linalg.inv(torch.cat([rowcombineX], dim=0))[:,0:2,:]
        
        
        rowcombineRT= torch.matmul(rowcombineT,rowcombineR )
        combineRT_theta = torch.linalg.inv(torch.cat([rowcombineRT], dim=0))[:,0:2,:]
        
        
        return angle_theta,scale_theta, trans_theta,deform_theta,combine_theta,combineRT_theta
    
    def doubly_stochastic_matrix_to_permutation(self, matrix):
        matrix_np = matrix.detach().cpu().numpy()
        row_indices, col_indices = linear_sum_assignment(-matrix_np)  
        permutation_matrix = torch.zeros_like(matrix, device="cuda")
        permutation_matrix[row_indices, col_indices] = 1
        
        return permutation_matrix
        
        
    def forward(self, x,x_distance,vfinetuneI=0,FinetuneStatus=False, inference=False):#False
        RF,SFX,TFX,TFY = self.source_parameters(x)#self.source_parameters_NoMLP(x)
        mr, ms, mt, md,mx,mxRT= self.theta_matrix(RF,SFX,TFX,TFY)
        soft_perms_inf, log_alpha_w_noise = my_sinkhorn_ops.my_gumbel_sinkhorn(
            self.log_alpha, 
            self.temperature, 
            1, 
            self.noise_factor,  
            self.n_iter_sinkhorn, squeeze=True)
        x = F.interpolate(x, size=(self.resolution, self.resolution), mode='bilinear')
        if inference:
            permutation_matrix = self.doubly_stochastic_matrix_to_permutation(self.log_alpha.squeeze())
            
            x = torch.matmul(permutation_matrix, 
                             x.reshape(self.num_items, -1)).squeeze().reshape(self.num_items, 
                                                                              x.shape[1], self.resolution, 
                                                                              self.resolution)
            ms = torch.matmul(permutation_matrix, 
                ms.reshape(self.num_items, -1)).squeeze().reshape(self.num_items, 2, 3)
            md = torch.matmul(permutation_matrix, 
                md.reshape(self.num_items, -1)).squeeze().reshape(self.num_items, self.deformation_resolution, 
                                                                  self.deformation_resolution,2)
        else:
            
            x = torch.matmul(soft_perms_inf, 
                             x.reshape(self.num_items, -1)).squeeze().reshape(self.num_items, 
                                                                              x.shape[1], self.resolution, 
                                                                              self.resolution)
            ms = torch.matmul(soft_perms_inf, 
                            ms.reshape(self.num_items, -1)).squeeze().reshape(self.num_items, 2, 3)
            md = torch.matmul(soft_perms_inf, 
                            md.reshape(self.num_items, -1)).squeeze().reshape(self.num_items, self.deformation_resolution, 
                                                                              self.deformation_resolution,2)
        gridSource_ms = F.affine_grid(ms, (self.num_items, 1, self.deformation_resolution, self.deformation_resolution), align_corners=True)
        gridSource_mxRT = F.affine_grid(mxRT, (self.num_items, 1, self.deformation_resolution, self.deformation_resolution), align_corners=True)
        gridD = gridSource_mxRT+md  
        gridSource_ms = F.interpolate(gridSource_ms.permute(3, 0, 1, 2), size=(self.resolution, self.resolution), mode='bilinear').permute(1, 2, 3, 0)
        gridSource_mxRT = F.interpolate(gridSource_mxRT.permute(3, 0, 1, 2), size=(self.resolution, self.resolution), mode='bilinear').permute(1, 2, 3, 0)
        
        
        affine_source_ms = F.grid_sample(x,  gridSource_ms, mode = "bilinear", align_corners=True)
        affine_source = F.grid_sample(affine_source_ms,  gridSource_mxRT, mode = "bilinear", align_corners=True)
        affine_distance_ms = F.grid_sample(x_distance,  gridSource_ms, mode = "bilinear", align_corners=True)
        affine_distance = F.grid_sample(affine_distance_ms,  gridSource_mxRT, mode = "bilinear", align_corners=True)
       
        
        gridD = F.interpolate(gridD.permute(3, 0, 1, 2), size=(self.resolution, self.resolution), mode='bilinear',align_corners = True).permute(1, 2, 3, 0)
    
        
        if vfinetuneI <= 400: #<=1000:
            final = affine_source
        else:
            affine_distance = F.grid_sample(affine_distance_ms,  gridD, mode = "bilinear", align_corners=True)
            affine = F.grid_sample(affine_source_ms,  gridD, mode = "bilinear", align_corners=True)
            final = affine
       
        return final,affine_source,affine_distance,gridD


# In[6]:


image_tensor = ObjBImage_list.reshape(objectnumber,1,resolutionF,resolutionF).to('cuda')
imagedistance_tensor = ObjBdistanceImage_list.reshape(objectnumber,1,resolutionF,resolutionF).to('cuda')
color_image_tensor = ObjImage_list.reshape(objectnumber,4,resolutionF,resolutionF).to('cuda')
Dilate_image_tensor = ObjDilateImage_list.reshape(objectnumber,1,resolutionF,resolutionF).to('cuda')
hole = torch.tensor(hole, requires_grad=False).float().to('cuda')#hole.to('cuda')


# In[7]:


def overlap_loss(object_sum,alpha,object_sum2):
    sigmoid = nn.Sigmoid()
    return (object_sum2*sigmoid((object_sum - alpha)*1000))

pn = PackNet(image_tensor.shape[0], recordvalue, resolutionF,100,image_tensor ).to('cuda')

learning_rate = 0.01
optimizer = torch.optim.Adam(pn.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[400], gamma=0.1)#torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000], gamma=0.1)
mseloss = nn.MSELoss(reduction='sum')


# In[8]:


cache = []
cache_color = []
lossalphaOverlap=0.01
lossalphaOutD=0.01
lossalphaOverlapD=0.01
lossalphaMSE=1.0
lossalphaMSE2=0.0

lossalphaOut=0.01

i= 1

stageshapevalue = 0 
while i<=600:#1500:
    
        
    if i%2!=0:
        model_h = pn(image_tensor,imagedistance_tensor,i)
        model_hDilate = pn(Dilate_image_tensor,imagedistance_tensor,i)
        h2 = torch.sum(model_h[1], dim=0).squeeze()
        h2distance = torch.sum(pn(imagedistance_tensor,imagedistance_tensor,i)[2], dim=0).squeeze()
        h2_D =  torch.sum(model_h[0], dim=0).squeeze()
        h2_Dilate =  torch.sum(model_hDilate[0], dim=0).squeeze()
        
        h2_D_color =  torch.sum(pn(color_image_tensor,imagedistance_tensor,i)[0], dim=0).squeeze()
    else:
        model_h = pn(image_tensor,imagedistance_tensor,i, inference=True)#!!!!!!!!!!!!!!!!!!!!!!!
        model_hDilate = pn(Dilate_image_tensor,imagedistance_tensor,i, inference=True)#!!!!!!!!!!!!!!!!!!!!!!!
        h2 = torch.sum(model_h[1], dim=0).squeeze()
        h2distance = torch.sum(pn(imagedistance_tensor,imagedistance_tensor,i, inference=True)[2], dim=0).squeeze()#!!!!!!!!!!!!!!!!!!!!!!!
        h2_D =  torch.sum(model_h[0], dim=0).squeeze()
        h2_Dilate =  torch.sum(model_hDilate[0], dim=0).squeeze()
        h2_D_color =  torch.sum(pn(color_image_tensor,imagedistance_tensor,i, inference=True)[0], dim=0).squeeze()#!!!!!!!!!!!!!!!!!!!!!!!
 
    
    if i == 200: #501:
        if i%50==0: #100 ==0 :
            lossalphaOverlap= lossalphaOverlap+0.1
            lossalphaOutD= lossalphaOutD+0.1
            lossalphaOverlapD= lossalphaOverlapD+0.1
            lossalphaOut = lossalphaOut+0.1
            lossalphaMSE=lossalphaMSE+0.0
            lossalphaMSE2=lossalphaMSE2+0.0
        
    if i == 400:#1000:
        stageshapevalue = ((totalshape-(totaloverlap+totalout))/torch.sum(hole))
    if i> 400:#1000 :    
       
        stageshapevalue = ((totalshape-(totaloverlap+totalout))/torch.sum(hole))
        lossalphaOverlap= lossalphaOverlap+0.01
        lossalphaOutD= lossalphaOutD+0.01
        lossalphaOverlapD= lossalphaOverlapD+0.01
        lossalphaOut = lossalphaOut+0.01
        lossalphaMSE=lossalphaMSE+0.0
        lossalphaMSE2=lossalphaMSE2+0.0
            
    
    grid = model_h[3]
    dx = (grid[:, 1:, :, :] - grid[:, :-1, :, :])  
    dy = (grid[:, :, 1:, :] - grid[:, :, :-1, :]) 
    smoothness_loss_x = torch.sum(torch.pow(torch.abs(dx),2))
    smoothness_loss_y = torch.sum(torch.pow(torch.abs(dy),2))
    smoothness_loss = (smoothness_loss_x+smoothness_loss_y)*100*100#/2*100 #/ 2*100000000
    
    
    if 200 < i < 400:#500< i <1001:
        MSElossF =   mseloss((h2_Dilate),(hole[0,0]))
        MSElossF2 =  (torch.abs(torch.sum((h2_Dilate*hole[0,0])-(overlap_loss(h2_Dilate*hole[0,0],1.1,h2_Dilate*hole[0,0])))-torch.sum(h2_Dilate)))
        overlaploss = torch.sum(overlap_loss(h2_Dilate*hole[0,0],1.1,h2_Dilate*hole[0,0]))
        overlapdistancetransformloss=torch.sum(overlap_loss(h2_Dilate*hole[0,0],1.1,h2distance))
        outloss = torch.sum(overlap_loss((h2_Dilate*(1.-hole[0,0])),0.1,h2_Dilate))
        outdistloss = torch.sum(overlap_loss((h2_Dilate*(1.-hole[0,0])),0.1,distouthole_distance))
        
        loss = MSElossF *lossalphaMSE+overlaploss*(lossalphaOverlap)+overlapdistancetransformloss*lossalphaOverlapD+outdistloss*(lossalphaOutD)+outloss*lossalphaOut           

        
    else:
        MSElossF =   mseloss((h2_D),(hole[0,0]))#mseloss(torch.sum(h2_D),torch.sum(hole[0,0]))  
        MSElossF2 =  (torch.abs(torch.sum((h2_D*hole[0,0])-(overlap_loss(h2_D*hole[0,0],1.1,h2_D*hole[0,0])))-torch.sum(h2_D))) #mseloss(((h2_D*hole[0,0])-(overlap_loss(h2_D*hole[0,0],1.1,h2_D*hole[0,0]))),(h2_D*hole[0,0]))  
        overlaploss = torch.sum(overlap_loss(h2_D*hole[0,0],1.1,h2_D*hole[0,0]))
        overlapdistancetransformloss=torch.sum(overlap_loss(h2_D*hole[0,0],1.1,h2distance))
        outloss = torch.sum(overlap_loss((h2_D*(1.-hole[0,0])),0.1,h2_D))
        outdistloss = torch.sum(overlap_loss((h2_D*(1.-hole[0,0])),0.1,distouthole_distance))
        MSElossDeform =   mseloss((h2_D),(h2))
        
        
        if i > 400:#1000:
            smoothness_loss = smoothness_loss *0
        else:
            smoothness_loss = smoothness_loss *0 
        
        loss = smoothness_loss+MSElossF *lossalphaMSE+overlaploss*(lossalphaOverlap)+overlapdistancetransformloss*lossalphaOverlapD+outdistloss*(lossalphaOutD)+outloss*lossalphaOut           
          
            
    if i >=0 :
        if (i) % 50 == 0:#100 == 0:
            cache.append((h2_D.reshape(1,resolutionF,resolutionF)+hole[0])/2)
            loss_string = f"Loss: {loss.item():.2f}, MSE smooth_loss: {smoothness_loss:.5f}, MSE Loss: {mseloss(h2_D,hole[0,0]).item():.2f}, OVERLAP: {(overlaploss).item():.2f}, Out: {outloss.item():.2f}, Outdistance: {(outdistloss*lossalphaOutD).item():.2f}, totaloverlap: {totaloverlap.item():.2f}, totalout: {totalout.item():.2f}, totalshape: {totalshape.item():.2f}, totalhole: {torch.sum(hole).item():.2f}, totaloverlap+out: {(totaloverlap.item()+totalout.item())/(torch.sum(hole)).item()}totalpercent: {(totalshape-(totaloverlap+totalout))/(torch.sum(hole))}"
            print(loss_string,":",i)
        
       
    optimizer.zero_grad()
    loss.backward()
   
            
    totaloverlap = torch.sum(torch.where(((h2_D*hole[0,0])*torch.sigmoid((h2_D*hole[0,0] - 1.1)*1000))>=0.1,1.0,0.0))
    totalout = torch.sum(torch.abs((torch.where(h2_D>=0.1,1.0,0.0)*(1.-hole[0,0]))))
    totalshape = torch.sum(torch.where(h2_D>=0.1,1.0,0.0) )    
    totalshape_ND = torch.sum(torch.where(h2>=0.1,1.0,0.0) )
    totalout_ND = torch.sum(torch.abs((torch.where(h2>=0.1,1.0,0.0)*(1.-hole[0,0]))))
    totaloverlap_ND = torch.sum(torch.where(((h2*hole[0,0])*torch.sigmoid((h2*hole[0,0] - 1.1)*1000))>=0.1,1.0,0.0))     
    optimizer.step()
    scheduler.step()
    
    i= i+1
      


# In[9]:


totaloverlap = torch.sum(torch.where(((h2_D*hole[0,0])*torch.sigmoid((h2_D*hole[0,0] - 1.01)*1000))>=0.01,1.0,0.0))
totalout = torch.sum(torch.abs((torch.where(h2_D>=0.01,1.0,0.0)*(1.-hole[0,0]))))


# In[10]:


print((totalshape.item())/(torch.sum(hole)).item())
print((totaloverlap.item()+totalout.item())/(torch.sum(hole)).item())
print((totalshape_ND.item())/(torch.sum(hole)).item())
print((totaloverlap_ND.item()+totalout_ND.item())/(torch.sum(hole)).item())


# In[11]:


training_result = torchvision.transforms.functional.to_pil_image(torchvision.utils.make_grid(cache))
training_result.save("affine_multiplication.png")
training_result


# In[12]:


resolutionF2=512


# In[13]:


pn2 = PackNet(image_tensor.shape[0], recordvalue, resolutionF2, 100 ,image_tensor).to('cuda')
pn2.load_state_dict(pn.state_dict())


# In[14]:


resized_img = F.interpolate(hole, size=(resolutionF2, resolutionF2), mode='bilinear', align_corners=False)


# In[15]:


torch.set_printoptions(profile='full')


# In[16]:


rgbpic = Image.new("RGBA", (resolutionF2, resolutionF2))
rgbpic_source = Image.new("RGBA", (resolutionF2, resolutionF2))
for i in range(pn2(color_image_tensor,image_tensor)[0].size()[0]-1, -1, -1):
        pic = torchvision.transforms.functional.to_pil_image(((pn2(color_image_tensor,image_tensor,5000, inference=True)[0][i] )).cpu())
        rgbpic.paste(pic, (0, 0),pic)
rgbpic.save(datasetnumber+"_TVCG_"+ordernumber+"_deform2.png")

for i in range(pn2(color_image_tensor,image_tensor)[1].size()[0]-1, -1, -1):
        pic = torchvision.transforms.functional.to_pil_image(((pn2(color_image_tensor,image_tensor,5000, inference=True)[1][i] )).cpu())
        rgbpic_source.paste(pic, (0, 0),pic)
rgbpic_source.save(datasetnumber+"_TVCG_"+ordernumber+"_source2.png")

