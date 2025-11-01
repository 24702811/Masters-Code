import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
#from torchvision.models import ResNet50_Weights
import faiss
import numpy as np
import os
from PIL import Image
import pickle
from tqdm import tqdm 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# def calc_mean(images):
    
#     transform = transforms.Compose([     # resize and get tensor
#         transforms.Resize((224, 224)), 
#         transforms.ToTensor()  
#     ])

#     mean = torch.zeros(3) 
#     std = torch.zeros(3)
#     count = 0



#     for img_path in tqdm(images):  
#         image = Image.open(img_path).convert("RGB")    # get image from path - ensure it is rgb
#         image_tensor = transform(image)  # transform image

#         count += 1
#         mean += image_tensor.mean(dim=[1, 2])  #  find mean of colour channels
#         std += image_tensor.std(dim=[1, 2]) 

#     mean /= count
#     std /= count

#     mean = mean.tolist()
#     std = std.tolist()
#     return mean, std




def load_model(mean, std):
    
    
    model = models.resnet50(pretrained=False)
    model.fc = nn.Identity()  

    model.load_state_dict(torch.load('/home/alex/ros2_ws/mapping/SMALL_HOUSE_FINAL_MODEL.pth', map_location=device))
    model = model.to(device)
    model.eval()
    
    transform = transforms.Compose([     # same transform but normalise rgb channels also
    transforms.Resize((224, 224)),  
    transforms.ToTensor(), 
    transforms.Normalize(mean=mean, std=std)])  
    
    return model, transform

def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")    # open image
    return transform(image).unsqueeze(0).to(device)           #  add batch size 


def extract_descriptor(image_path, model, transform):
    
    image_tensor = preprocess_image(image_path, transform)  
    with torch.no_grad():  # dont need gradients - not training
        descriptor = model(image_tensor).squeeze(0)  #get descriptor
      #  descriptor = torch.nn.functional.normalize(descriptor, p=2, dim=0)   #l2 normalise (gives descriptors unit length)
        
    return descriptor.cpu() 



def extract_descriptors(images, model, transform):

    descs = []
    image_paths = []

    for img_path in tqdm(images):     # get descriptors for each image
        desc = extract_descriptor(img_path, model, transform)   
        descs.append(desc)
        image_paths.append(img_path)

   # descriptors = np.array(descs)
    descriptors = torch.stack(descs, dim=0).to(torch.float32).cpu().numpy()
    descriptors = np.ascontiguousarray(descriptors, dtype="float32")
    return descriptors, image_paths

def build_index(descriptors):
   
    d = descriptors.shape[1]  # get dimensions (2048 for resnet50)
    index = faiss.IndexFlatL2(d)  # create l2 index 
    index.add(descriptors)  # add descriptors
    
    return index


def save_index(index, images, mean, std):

    faiss.write_index(index, "SMALL_HOUSE_FINAL_INDEX.bin")  # save the index

    with open("SMALL_HOUSE_FINAL_IMAGE_PATHS.pkl", "wb") as f:
        pickle.dump(images, f)         #save the paths

    # with open("ESL_FINAL_MEAN.pkl", "wb") as f:
    #     pickle.dump({"mean": mean, "std": std}, f)     # save mean and std


def main(args=None):
    
    image_paths = [os.path.join("/home/alex/ros2_ws/map_data/small_house_FINAL/frames", f) for f in os.listdir("/home/alex/ros2_ws/map_data/small_house_FINAL/frames") if f.endswith((".jpg", ".png", ".webp"))]
    
   # mean, std = calc_mean(image_paths)   # calc mean and std for dataset
   
    with open("SMALL_HOUSE_FINAL_MEAN.pkl", "rb") as f:
        norm_values = pickle.load(f)       
    mean = norm_values["mean"]
    std = norm_values["std"]
    

    model, transform = load_model(mean, std)        #  load model and transform
    descriptors, paths = extract_descriptors(image_paths, model, transform)    # get image descriptors and respective paths
    index = build_index(descriptors)        #   create the index using descriptors
    save_index(index, paths, mean, std)     #   save the index
    
    print("mean: ", mean, " ", "std: ", std)
    

if __name__ == '__main__':
    main()
