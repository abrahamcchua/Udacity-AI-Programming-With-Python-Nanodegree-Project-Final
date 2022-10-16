# Built in modules
import random
import os
import json
import csv
# Python modules
import torch
from PIL import Image
# Own modules
from train import train

class predictor(train):
    def __init__(self, data_dir = "flowers", gpu = "default"):
        super().__init__(data_dir, gpu) 
        self.checkpoint_path = "checkpoint-densenet161.pth"
        
    def module(self, topk):
        test_data_loader = self.test_data_loader()
        loaded_model = self.load_checkpoint(self.checkpoint_path)
        loaded_model.to(self.device);
        self.tester(loaded_model, "imported_" + self.checkpoint_path.split("-")[1][:-4], test_data_loader, self.loss_func)
        cat_to_name = self.category_label_importer()
        test_folder, img_loc = self.random_file_selector()
        processed_image = self.image_processor(img_loc)
        top_classes, probabilities = self.predict(loaded_model, processed_image)
        model_classes = loaded_model.class_to_idx
        topk_class_list = self.cnumber_to_class(top_classes, model_classes, cat_to_name)
        self.output(cat_to_name, topk, test_folder, topk_class_list, probabilities)
        
        
    def load_checkpoint(self, file):
        model_state = torch.load(file)
        loaded_model = self.valid_pretrained_models.get(file.split("-")[1][:-4])(pretrained=True)
        loaded_model.classifier = model_state['classifier']
        loaded_model.load_state_dict(model_state['state_dict'])
        loaded_model.class_to_idx = model_state['class_to_idx']
        return loaded_model


    def image_processor(self, image):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''
        
        with Image.open(image) as im:
            image = self.eval_transforms(im)
        return image
    
    
    def random_file_selector(self):
        os.chdir("/data")
        # Choose a random file or folder path
        test_folder = random.choice(os.listdir(self.test_dir))
        # Go into that file path
        os.chdir(self.test_dir + "/" + test_folder)
        # File path to the random image
        img_loc = random.choice(os.listdir(os.getcwd()))

        return test_folder, img_loc
    
    
    def predict(self, model, img,  topk=5):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        # TODO: Implement the code to predict the class from an image file
        model.to(self.device)
        model.eval()
        image = img.to(self.device)
        image = image.unsqueeze(0)
        
        # Calculate the class probabilities (softmax) for img
        with torch.no_grad():
            output = model.forward(image)
            ps = torch.exp(output)
            top_p, top_class = ps.topk(topk, dim=1)
        ps = ps.cpu().data.numpy().squeeze()
        top_classes = top_class.cpu().numpy()
        probabilities = top_p.cpu().data.numpy().squeeze()
        return top_classes, probabilities
    
        
    def category_label_importer(self, file="cat_to_name.json"):
        os.chdir("/home/workspace/ImageClassifier")
        with open(file, 'r') as f:
            return json.load(f)
    

    def cnumber_to_class(self, model_predictions, model_classes, dictionary):
        output_list = []
        for prediction in model_predictions[0]:
            model_class = [k for k, v in model_classes.items() if v == prediction]
            output_list.append(dictionary.get(model_class[0]))
        return output_list
    

    def output(self, cat_to_name, topk, test_folder, topk_classes, probabilities, filename="prediction.csv"):
        os.chdir("/home/workspace/ImageClassifier")
        correct_class_name = cat_to_name.get(str(test_folder)).title()
        with open(filename, "w") as f:
            writer = csv.writer(f)
            print("Ground truth of Image : {}".format(correct_class_name))
            header  = ["Top"+str(topk)+"Classes", "Probabilites"]
            writer.writerow(header)
            for c, p in zip(topk_classes, probabilities):
                row = [c, p]
                writer.writerow(row)
                print(f"The class {c.title()} has a probability of {p*100}%")
            ground_truth = ["Ground truth of Image :", correct_class_name]
            writer.writerow(ground_truth)
        
    

        
classifier = predictor()
classifier.module(5)