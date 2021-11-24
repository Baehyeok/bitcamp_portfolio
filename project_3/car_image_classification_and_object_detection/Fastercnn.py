import torch      #pytorch
import torch.nn as nn     #pytorch network
from torch.utils.data import Dataset, DataLoader      #pytorch dataset
from torch.utils.tensorboard import SummaryWriter     #tensorboard
import torchvision      #torchvision
import torch.optim as optim     #pytorch optimizer
import numpy as np      #numpy
import matplotlib.pyplot as plt     #matplotlib(이미지 표시를 위해 필요)
from collections import OrderedDict     #python라이브러리 (라벨 dictionary를 만들 때 필요)
import os     #os
import xml.etree.ElementTree as Et      #Pascal xml을 읽어올 때 필요
from xml.etree.ElementTree import Element, ElementTree
import cv2      #opencv (box 그리기를 할 때 필요)
from PIL import Image     #PILLOW (이미지 읽기)
import time     #time
import imgaug as ia     #imgaug
from imgaug import augmenters as iaa
from torchvision import transforms      #torchvision transform

#GPU연결
if torch.cuda.is_available():
  device = torch.device('cuda:0')
else:
  device = torch.device('cpu')
print(device)

def xml_parser(xml_path):
    xml_path = xml_path
    xml = open(xml_path, "r",encoding='UTF8')
    tree = Et.parse(xml)
    root = tree.getroot()
    size = root.find("size")
    file_name = root.find("filename").text
    object_name = []
    bbox = []
    objects = root.findall("object")
    for _object in objects:
        name = _object.find("name").text
        object_name.append(name)
        bndbox = _object.find("bndbox")
        one_bbox = []
        xmin = bndbox.find("xmin").text
        one_bbox.append(int(float(xmin)))
        ymin = bndbox.find("ymin").text
        one_bbox.append(int(float(ymin)))
        xmax = bndbox.find("xmax").text
        one_bbox.append(int(float(xmax)))
        ymax = bndbox.find("ymax").text
        one_bbox.append(int(float(ymax)))
        bbox.append(one_bbox)
    
    return file_name, object_name, bbox

def makeBox(voc_im,bbox,objects):
    image = voc_im.copy()
    for i in range(len(objects)):
        cv2.rectangle(image,(int(bbox[i][0]),int(bbox[i][1])),(int(bbox[i][2]),int(bbox[i][3])),color = (0,255,0),thickness = 1)
        cv2.putText(image, objects[i], (int(bbox[i][0]), int(bbox[i][1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2) # 크기, 색, 굵기
    return image

xml_list = os.listdir("C:/work/project/car/train_img/xml/")
xml_list.sort()

label_set = set()

for i in range(len(xml_list)):
    xml_path = "C:/work/project/car/train_img/xml/"+str(xml_list[i])
    file_name, object_name, bbox = xml_parser(xml_path)
    for name in object_name:
        label_set.add(name)

label_set = sorted(list(label_set))

label_dic = {}
for i, key in enumerate(label_set):
    label_dic[key] = (i+1)

print(label_dic)
  

class Car(Dataset):

    def __init__(self,xml_list,len_data):
        self.transform = transforms.ToTensor()
        self.xml_list = xml_list
        self.len_data = len_data
        self.to_tensor = transforms.ToTensor()
        self.flip = iaa.Fliplr(0.5)
        self.resize = iaa.Resize({"height": 600, "width": 1000})

    def __len__(self):
        return self.len_data

    def __getitem__(self, idx):

        xml_path = "C:/work/project/car/train_img/xml/"+str(xml_list[idx])
        file_name, object_name, bbox = xml_parser(xml_path)
        image_path = "C:/work/project/car/train_img/train_img/"+str(file_name)
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        image, bbox = self.flip(image = image, bounding_boxes = np.array([bbox]))
        image, bbox = self.resize(image = image,bounding_boxes = bbox)
        bbox = bbox.squeeze(0).tolist()
        image = self.to_tensor(image)
        targets = []
        d = {}
        d['boxes'] = torch.tensor(bbox,device=device)
        d['labels'] = torch.tensor([label_dic[x] for x in object_name],dtype=torch.int64,device = device)
        targets.append(d)

        return image, targets

backbone = torchvision.models.vgg16(pretrained=True).features[:-1]
backbone_out = 512

backbone.out_channels = backbone_out

anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(sizes=((128,256,512),),aspect_ratios=((0.5, 1.0, 2.0),))

resolution = 9

roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=resolution, sampling_ratio=2)

# 
# box_head = torchvision.models.detection.faster_rcnn.TwoMLPHead(in_channels= backbone_out*(resolution**2),representation_size=4096) 
box_head = torchvision.models.detection.faster_rcnn.TwoMLPHead(in_channels= backbone_out*(resolution**2),representation_size=1024) 
# box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(4096,2) #2개 class
box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(1024,3) #2개 class

model = torchvision.models.detection.FasterRCNN(backbone, num_classes=None,
                   min_size = 600, max_size = 1000,
                   rpn_anchor_generator=anchor_generator,
                   rpn_pre_nms_top_n_train = 6000, rpn_pre_nms_top_n_test = 5000,
                   rpn_post_nms_top_n_train=1000, rpn_post_nms_top_n_test=500,
                   rpn_nms_thresh=0.7,rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                   rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                   box_roi_pool=roi_pooler, box_head = box_head, box_predictor = box_predictor,
                   box_score_thresh=0.05, box_nms_thresh=0.7,box_detections_per_img=300,
                   box_fg_iou_thresh=0.7, box_bg_iou_thresh=0.5,
                   box_batch_size_per_image=128, box_positive_fraction=0.25
                 )
#roi head 있으면 num_class = None으로 함

for param in model.rpn.parameters():
    torch.nn.init.normal_(param,mean = 0.0, std=0.01)

for name, param in model.roi_heads.named_parameters():
    if "bbox_pred" in name:
        torch.nn.init.normal_(param,mean = 0.0, std=0.001)
    elif "weight" in name:
        torch.nn.init.normal_(param,mean = 0.0, std=0.01)
    if "bias" in name:
        torch.nn.init.zeros_(param)

def Total_Loss(loss):
    loss_objectness = loss['loss_objectness']
    loss_rpn_box_reg = loss['loss_rpn_box_reg']
    loss_classifier = loss['loss_classifier']
    loss_box_reg = loss['loss_box_reg']

    rpn_total = loss_objectness + 10*loss_rpn_box_reg
    fast_rcnn_total = loss_classifier + loss_box_reg

    total_loss = rpn_total + fast_rcnn_total

    return total_loss


len_data = 94


#Train

total_epoch = 600

loss_sum = 0
#model.load_state_dict(torch.load(,map_location=device))

model.to(device)

#model.to(torch.device('cuda'))

optimizer = torch.optim.Adam(params = model.parameters(),lr = 0.001, weight_decay=0.0005)
scheduler = optim.lr_scheduler.StepLR(optimizer = optimizer, step_size= 10, gamma= 0.9)

print("Training Start")
model.train()

start = time.time()
loss_list = []
epoch_list=[]
for epoch in range(total_epoch):

    dataset = Car(xml_list[:len_data],len_data)
    dataloader = DataLoader(dataset,shuffle=True)

    for i, (image,targets)in enumerate(dataloader):
        
        optimizer.zero_grad()

        targets[0]['boxes'].squeeze_(0)
        targets[0]['labels'].squeeze_(0)

        loss = model(image.to(device),targets)
        total_loss = Total_Loss(loss)
        loss_sum += total_loss
        
        total_loss.backward()
        torch.cuda.empty_cache()
        optimizer.step()
  
    end = time.time()
    print("Epoch {} | Loss: {} | Duration: {} sec".format(epoch,(loss_sum/i).item(),int((end-start))))
    loss_list.append((loss_sum/i).item())
    
    epoch_list.append(epoch)
    loss_sum = 0
    scheduler.step()
    start = time.time()

plt.plot(epoch_list, loss_list)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()



torch.save(model.state_dict(),"Car_600.pth")

model.to(device)
model.load_state_dict(torch.load("Car_600.pth",map_location=device))
resize = transforms.Resize((600,1000))
model.roi_heads.score_thresh = 0.1
model.roi_heads.nms_thresh = 0.001
model.roi_heads.detections_per_img = 5

image_list = os.listdir("C:/work/project/car/test_img")

#model.eval()
for i in range(len(image_list)):

    to_tensor = transforms.ToTensor()

    test_image = Image.open("C:/work/project/car/test_img/"+image_list[i])
    #test_image = Image.open("C:/work/project/car/test_img/Maliboo_2.jpg") #외부 이미지 사용
    test_image = resize(test_image)

    test_image = to_tensor(test_image).unsqueeze(0)

    targets = []
    d = {}
    d['boxes'] = torch.tensor(bbox,device=device)
    d['labels'] = torch.tensor([label_dic[x] for x in object_name],dtype=torch.int64,device = device)
    targets.append(d)
    model.eval()
    predictions = model(test_image.to(device))

    boxes = predictions[0]['boxes']
    labels = predictions[0]['labels']

    print("<Answer>")
    print(object_name, boxes)

    print("<Prediction>")
    objects = []
    for lb in labels:
        objects.append([k for k, v in label_dic.items() if v == lb][0])

    for a,b in zip(objects, boxes):
        print(a,": ",b.tolist())

    plot_image = test_image.squeeze().permute(1,2,0).numpy()
    answer = makeBox(plot_image,boxes,objects)
    plt.imshow(answer)
    plt.show()