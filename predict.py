# Built in modules
import random
import os
import json
import csv
from torchvision import models, transforms
# Python modules
import torch
from PIL import Image
# Own modules
from input_args import input_args_predict


'''
The prediction script for Project #2 for the AWS Udacity AI Programming With Python Nanodegree 
:Date: October 19, 2022
:Author/s: 
    - Chua, Abraham <abrahamcchua@gmail.com>
'''

class Predictor():
    def __init__(self, predict_arguments):
        self.checkpoint = predict_arguments.checkpoint
        self.save_dir = predict_arguments.save_dir
        self.checkpoint_path = predict_arguments.save_dir + "/" + predict_arguments.checkpoint
        if predict_arguments.image == "random":
            self.test_dir = "flowers/test/"
            self.test_folder, self.img_loc = self.random_file_selector()
        else:
            self.test_dir = "flowers/test/"
            self.test_folder = predict_arguments.image.split("/")[2]
            self.img_loc = predict_arguments.image.split("/")[3]
        self.topk = predict_arguments.topk
        self.category_names = predict_arguments.category_names
        gpu = predict_arguments.gpu
        if gpu == "default":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(gpu)
        self.valid_pretrained_models = {
        "densenet121": models.densenet121,
        "densenet161": models.densenet161,
        "densenet169": models.densenet169,
        "densenet201": models.densenet201
        } 
        self.eval_transforms = transforms.Compose([transforms.Resize(255),
                                                   transforms.CenterCrop(224),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                                        [0.229, 0.224, 0.225])
                                                  ])  
    
    
    def module(self):
        """Main highlight of function "module"
        Summary:
        The frame where in the different functions are used
        output of the function includes:
        None
        """
        loaded_model = self.load_checkpoint(self.save_dir, self.checkpoint)
        loaded_model.to(self.device);
        cat_to_name = self.category_label_importer(self.category_names)
        processed_image = self.image_processor(self.img_loc)
        top_classes, probabilities = self.predict(loaded_model, processed_image)
        model_classes = loaded_model.class_to_idx
        topk_class_list = self.cnumber_to_class(top_classes, model_classes, cat_to_name)
        self.output(cat_to_name, self.topk, self.test_folder, topk_class_list, probabilities)
        
        
    def load_checkpoint(self, save_dir, file):
        """Main highlight of function "load_checkpoint"
        Summary:
        loads a specified model
        :param save_dir: string; location of the checkpoint of the model to be loaded
        :param file: string; name of the checkpoint file to be loaded
        output of the function includes:
        The model to be from the checkpoint file
        """
        print(os.getcwd())
        print(os.listdir())
        os.chdir(save_dir)
        model_state = torch.load(file)
        loaded_model = self.valid_pretrained_models.get(file.split("-")[1][:11])(pretrained=True)
        loaded_model.classifier = model_state['classifier']
        loaded_model.load_state_dict(model_state['state_dict'])
        loaded_model.class_to_idx = model_state['class_to_idx']
        os.chdir("..")
        return loaded_model


    def image_processor(self, image):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''
        os.chdir(self.test_dir)
        os.chdir(self.test_folder)
        with Image.open(image) as im:
            image = self.eval_transforms(im)
        os.chdir("../../../..")
        return image
    
    
    def random_file_selector(self):
        """Main highlight of function "random_file_selector"
        Summary:
        Selects a random folder and then selects a random file/folder within that directory
        output of the function includes:
        test folder and image location
        """
        os.chdir("/data")
        # Choose a random file or folder path
        test_folder = random.choice(os.listdir(self.test_dir))
        # Go into that file path
        os.chdir(self.test_dir + "/" + test_folder)
        # File path to the random image
        img_loc = random.choice(os.listdir(os.getcwd()))
        os.chdir("../../..")
        return test_folder, img_loc
    
    
    def predict(self, model, img,  topk=5):
        """Main highlight of function "predict"
        Summary:
        Predict the class (or classes) of an image using a trained deep learning model.
        :param model; tensor; The model to be used for inference
        :param img; numpy array; The image to be classified
        :param topk; int; The number highest probabilities to be returned 
        output of the function includes:
        The topk probabilities and their corresponding classes
        """
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
    
        
    def category_label_importer(self, file):
        """Main highlight of function "category_label_importer"
        Summary:
        Reads a json object and return it as a dictionary
        :param file; string; The name of the json file where the class names are stored.
        output of the function includes:
        A dictionary containing the classes
        """
        with open(file, 'r') as f:
            return json.load(f)
    

    def cnumber_to_class(self, model_predictions, model_classes, dictionary):
        """Main highlight of function "cnumber_to_class"
        Summary:
        Converts the prediction of the model to a list
        :param model_predictions; list; The list containing the predictions of the model on the image
        :param model_classes; dictionary; The list containing the classes from the model
        :param dictionary; The dictionary containing the classes from the json file
        output of the function includes:
        A dictionary containing the classes
        """
        output_list = []
        for prediction in model_predictions[0]:
            model_class = [k for k, v in model_classes.items() if v == prediction]
            output_list.append(dictionary.get(model_class[0]))
        return output_list
    

    def output(self, cat_to_name, topk, test_folder, topk_classes, probabilities, filename="prediction.csv"):
        """Main highlight of function "output"
        Summary:
        Presents the results from the classifier
        :param cat_to_name; string; The name of the dictionary containing the class names
        :param topk; int; The top n classes
        :param test_folder; string; The test folder containing the image
        :param topk_classes; list;  The topk_classes predicted by the model
        :param probabilities; list; The probabilities of the topk classes
        :param filename: string; The prefix of the csv filename 
        output of the function includes:
        A dictionary containing the classes
        """
        correct_class_name = cat_to_name.get(str(test_folder)).title()
        with open(filename, "w") as f:
            writer = csv.writer(f)
            print("Ground truth of Image : {}".format(correct_class_name))
            header  = ["Top_"+str(topk)+"_Classes", "Probabilites"]
            writer.writerow(header)
            for c, p in zip(topk_classes, probabilities):
                row = [c, p]
                writer.writerow(row)
                print(f"The class {c.title()} has a probability of {p*100}%")
            ground_truth = ["Ground truth of Image :", correct_class_name]
            writer.writerow(ground_truth)
        
    

if __name__ == '__main__':
    arguments = input_args_predict()        
    classifier = Predictor(arguments)
    classifier.module()
    

# python predict.py --checkpoint checkpoint-densenet201_e_10_lr_0.01.pth 