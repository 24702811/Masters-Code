import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import pickle
from tqdm import tqdm
import torch.nn.functional as F 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(train_path):

    
    places = []
    for place in os.listdir(train_path):
        full_path = os.path.join(train_path, place)
        if os.path.isdir(full_path):
            places.append(place)   # get list of places
            
    class_images_dict = {}
    data = []
    
    for place in places:
        class_path = os.path.join(train_path, place)  #  get path for images of a place
        
        images = []
        for image in os.listdir(class_path):    
            if image.lower().endswith(('.png', 'jpg')):
                full_image_path = os.path.join(class_path, image)  
                images.append(full_image_path)      #  list of all the image paths
                
        if images:
            class_images_dict[place] = images  #  create dictionary of places and the image paths relating to place
            for image in images:
                data.append((place, image))    # do same but make list of place-paths pairs
                
    return data, places, class_images_dict

def get_triplet(index, data, transform, embeddings, margin):
  
    
    anchor_place, anchor_path = data[index]
   # anchor = Image.open(anchor_path).convert("RGB") # open the image
   # anchor_tensor = transform(anchor).to(device) # pre process for resnet50
    

    positives = []
    
    for place, path in data:
        if place == anchor_place and path != anchor_path:
            positives.append(path)
            
    positive_path = random.choice(positives)   # choose random positive image from the same place
    #positive = Image.open(positive_path).convert("RGB")
    #positive_tensor = transform(positive).to(device)

    # with torch.no_grad():
    #     anchor_embedding = model(anchor_tensor.unsqueeze(0)) # get anchor embedding
    #     positive_embedding = model(positive_tensor.unsqueeze(0)) # get positive embedding
    #     anchor_pos_distance = torch.norm(anchor_embedding - positive_embedding, p=2).item()    # get l2 distance between anchor and positive embedding
    
    anchor_embedding = embeddings[anchor_path]       
    positive_embedding = embeddings[positive_path]  
    anchor_pos_distance = torch.norm(anchor_embedding - positive_embedding, p=2).item()


    possible_negatives = []
    
    for negative_place, negative_path in (data):
        if negative_place != anchor_place:
            
           # negative = Image.open(negative_path).convert("RGB")
           # negative_tensor = transform(negative).unsqueeze(0).to(device)           # get current negative tensor
            
            
            negative_embedding = embeddings[negative_path]  
            neg_dist = torch.norm(anchor_embedding - negative_embedding, p=2).item() # get distance between anchor and negative embedding
            
            

            if anchor_pos_distance < neg_dist < anchor_pos_distance + margin:   # check if the negative is a valid negative for semi hard mining
                possible_negatives.append((neg_dist, negative_path))
                
    if possible_negatives:
        _, negative_path = min(possible_negatives, key=lambda x: x[0])   # get negative with the smallest distance to the anchor (hardest negative)
    else:
        negatives = [img_path for (place, img_path) in data if place != anchor_place] 
        negative_path = random.choice(negatives) # if no valid negatives choose randomly
        

    anchor = Image.open(anchor_path).convert("RGB") # open the image
    anchor_tensor = transform(anchor).to(device) # pre process for resnet50

    positive = Image.open(positive_path).convert("RGB")
    positive_tensor = transform(positive).to(device)

    negative = Image.open(negative_path).convert("RGB")
    negative_tensor = transform(negative).to(device)  # pre process for resnet50     
 
    # print("anc: ", anchor_path)   
    # print("pos: ", positive_path)
    # print("neg: ", negative_path)  
    # print("")

       


    return anchor_tensor, positive_tensor, negative_tensor


def calc_mean(folder_path):

    images = []
    for root, _, files in os.walk(folder_path):
        for f in files:
            if os.path.splitext(f)[1].lower() == ".png" or os.path.splitext(f)[1].lower() == ".jpg":
                images.append(os.path.join(root, f))

    transform = transforms.Compose([     # resize and get tensor
        transforms.Resize((224, 224)), 
        transforms.ToTensor()  
    ])

    mean = torch.zeros(3) 
    std = torch.zeros(3)
    count = 0

    for img_path in tqdm(images):  
        image = Image.open(img_path).convert("RGB")    # get image from path - ensure it is rgb
        image_tensor = transform(image)                # transform image

        count += 1
        mean += image_tensor.mean(dim=[1, 2])          # find mean of colour channels
        std  += image_tensor.std(dim=[1, 2])


    mean /= count
    std  /= count

    mean = mean.tolist()
    std = std.tolist()
    return mean, std



def compute_embeddings(data, transform, model):
    model.eval()
    embeddings = {}

    with torch.no_grad():
        for place, path in tqdm(data):
            image = Image.open(path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)
            embedding = model(image_tensor).squeeze(0).cpu()  
        #   embedding = F.normalize(embedding, p=2, dim=0)  # CHANGED: L2-normalize per-vector for mining/eval
            embeddings[path] = embedding

    return embeddings

def train_model(model, loss_function, optimizer, epochs, batch_size, data, transform, margin):
    for epoch in range(epochs):
        
        running_loss = 0.0
        
        num_samples = len(data)
        indices = list(range(num_samples))
        random.shuffle(indices)              #  shuffles the indices
        
        model.eval()
        embeddings = compute_embeddings(data, transform, model) 
        model.train()
        
        
        for i in tqdm(range(0, num_samples, batch_size)):   
            
            batch_indices = indices[i:i+batch_size]  #  gets the indices for the current batch
            anchor_tensors = []     
            positive_tensors = [] 
            negative_tensors = []

            for idx in (batch_indices):
                anchor_tensor, positive_tensor, negative_tensor = get_triplet(idx, data, transform, embeddings, margin)
                anchor_tensors.append(anchor_tensor)
                positive_tensors.append(positive_tensor)
                negative_tensors.append(negative_tensor)   #  makes a batch of triplets 

            anchor_tensors = torch.stack(anchor_tensors).to(device)
            positive_tensors = torch.stack(positive_tensors).to(device)
            negative_tensors = torch.stack(negative_tensors).to(device)  #  turn the batch into single tensors

            optimizer.zero_grad()   #  gradients from prev batch dont affect this one


            anchor_embeddings = model(anchor_tensors)
            positive_embeddings  = model(positive_tensors)
            negative_embeddings  = model(negative_tensors)    # get features

            # anchor_embeddings  = F.normalize(anchor_embeddings,  p=2, dim=1)
            # positive_embeddings = F.normalize(positive_embeddings, p=2, dim=1)
            # negative_embeddings = F.normalize(negative_embeddings, p=2, dim=1)

            loss = loss_function(anchor_embeddings, positive_embeddings, negative_embeddings)  #find loss
            loss.backward()  # calc gradients
            optimizer.step()  #   updates the models params and weights

            #running_loss += loss.item()
            
            curr_batch_size = anchor_tensors.size(0)            # actual batch size (last batch may be smaller)
            running_loss += loss.item() * curr_batch_size    # TripletMarginLoss default is reduction='mean'

        avg_loss = running_loss / num_samples 
        print("loss : ", avg_loss)
        
    return model
    

def main():

    data_path = 'small_house_FINAL_split_data/train/'  # path to training data
    data, places, class_images_dict = load_data(data_path)
    
    mean, std = calc_mean(data_path)   # calc mean and std for dataset
    print("mean: ", mean)
    with open("SMALL_HOUSE_FINAL_MEAN.pkl", "wb") as f:
        pickle.dump({"mean": mean, "std": std}, f)     # save mean and std

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])  

    batch_size = 64
    margin = 5.0 
    num_epochs = 8
    learning_rate = 1e-3
    

    model = models.resnet50(pretrained=True)
    model.fc = nn.Identity()  
    for p in model.conv1.parameters(): p.requires_grad = False
    for p in model.bn1.parameters():  p.requires_grad = False
    for p in model.layer1.parameters(): p.requires_grad = False
    for p in model.layer2.parameters(): p.requires_grad = False
    for p in model.layer3.parameters(): p.requires_grad = False
    model = model.to(device)   #  load pretrained mode to extract initial features
    
      
    
    loss_function = nn.TripletMarginLoss(margin, p=2)    # triplet loss ( play around with parameters )   
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)      # adam optimizer 
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True)


    

    trained_model = train_model(model, loss_function, optimizer, num_epochs, batch_size, data, transform, margin)

    torch.save(trained_model.state_dict(), 'models/SMALL_HOUSE_FINAL_MODEL.pth')

if __name__ == "__main__":
    main()
