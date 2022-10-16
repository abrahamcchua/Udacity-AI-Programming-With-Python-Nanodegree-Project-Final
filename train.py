# Built in modules
import time
import csv
# Third Party Modules
import torch
from torchvision import transforms, datasets, models
from torch import nn
from torch import optim
from torch.optim import lr_scheduler


class train():
    def __init__(self, data_dir = "flowers", gpu = "default"):
        self.train_dir = data_dir + '/train'
        self.valid_dir = data_dir + '/valid'
        self.test_dir = data_dir + '/test'
        if gpu == "default":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(gpu)
        self.train_transforms = transforms.Compose([transforms.RandomRotation(10),
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])
                                            ])
        self.eval_transforms = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])
                                            ])  
        self.loss_func = nn.CrossEntropyLoss()
        self.train_data = datasets.ImageFolder(self.train_dir, transform=self.train_transforms)
        self.validation_data = datasets.ImageFolder(self.valid_dir, transform=self.eval_transforms)
        self.test_data = datasets.ImageFolder(self.test_dir, transform=self.eval_transforms)
        self.valid_pretrained_models = {
        "densenet121": models.densenet121,
        "densenet161": models.densenet161,
        "densenet169": models.densenet169,
        "densenet201": models.densenet201
        }  
    
    def module(self, pretrained_model):
        train_data_loader = self.train_data_loader()
        validation_data_loader = self.validation_data_loader()
        test_data_loader = self.test_data_loader()
        model = self.model_architecture(pretrained_model)
        trained_model = self.trainer(model, train_data_loader, validation_data_loader, self.loss_func, pretrained_model)
        self.tester(trained_model, pretrained_model, test_data_loader, self.loss_func)
 
        
        
    
    # Separated from other data loaders if the user wants to change the batch size individually and for readability
    def train_data_loader(self, batch_size=64):
        return torch.utils.data.DataLoader(self.train_data, batch_size, shuffle=True)


    def validation_data_loader(self, batch_size=64):
        return torch.utils.data.DataLoader(self.validation_data, batch_size, shuffle=True)
    
    
    def test_data_loader(self, batch_size=64):
        return torch.utils.data.DataLoader(self.test_data, batch_size, shuffle=True)
   
        
    def model_architecture(self, pretrained_model, hidden_layer_one=810, hidden_layer_two=270):
        model = self.valid_pretrained_models.get(pretrained_model)(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        # Get the input features
        input_features = model.classifier.in_features
        model.classifier = nn.Sequential(nn.Linear(input_features, hidden_layer_one),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(hidden_layer_one, hidden_layer_two),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(hidden_layer_two, 102),
                                 nn.LogSoftmax(dim=1))
        model.to(self.device);
        return model
    
    
    def trainer(self, model, train_loader, validation_loader, loss_func, pretrained_model, filename="training_results", epochs=10, lr=0.05, momentum=0.9, step_size=7, gamma=0.5):
        with open(filename+"_"+pretrained_model+".csv", "w") as f:
            writer = csv.writer(f)
            parameters = ["Parameters:" ,epochs, lr, momentum, step_size, gamma]
            header = ["Epoch", "Train Loss", "Validation Loss", "Validation_Accuracy", "Time"]
            writer.writerow(parameters)
            writer.writerow(header)
            optimizer = optim.SGD(model.classifier.parameters(), lr, momentum)
            lr_change = lr_scheduler.StepLR(optimizer, step_size, gamma)
            for epoch in range(epochs):
                start_time = time.time()
                training_loss = 0
                validation_loss = 0
                accuracy = 0
                for image, label in train_loader:
                    # Move input and label tensors to the default device
                    image, label = image.to(self.device), label.to(self.device)
                    
                    result = model.forward(image)
                    loss = loss_func(result, label)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    training_loss += loss.item()
                lr_change.step()
                model.eval()
                with torch.no_grad():
                    for image, label in validation_loader:
                        image, label = image.to(self.device), label.to(self.device)
                        logps = model.forward(image)
                        batch_loss = loss_func(logps, label)
                        validation_loss += batch_loss.item()
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == label.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                values = [epoch+1, training_loss, validation_loss, accuracy/len(validation_loader), (time.time()-start_time)/60]
                writer.writerow(values)
                print(f"Epoch {epoch+1}/{epochs}.. "
                        f"Train loss: {training_loss:.4f}.. "
                        f"Validation loss: {validation_loss:.4f}.. "
                        f"Validation Accuracy: {accuracy/len(validation_loader):.4f}..")
                print(f"Time it took for this epoch: {time.time() - start_time: .4f}")
                model.train()
        model.class_to_idx = self.train_data.class_to_idx
        model_state = {
        'epoch': epochs,
        'state_dict': model.state_dict(),
        'optimizer_dict': optimizer.state_dict(),
        'classifier': model.classifier,
        'class_to_idx': model.class_to_idx,
        }

        torch.save(model_state, 'checkpoint-'+ pretrained_model +'.pth')
        return model
    
    
    def tester(self, model, pretrained_model, test_loader, loss_func, filename="testing_results"):
        with open(filename+"_"+pretrained_model+".csv", "w") as f:
            writer = csv.writer(f)
            header = ["Test Loss", "Test_Accuracy", "Time"]
            writer.writerow(header)
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                start_time = time.time()
                for image, label in test_loader:
                    image, label = image.to(self.device), label.to(self.device)
                    logps = model.forward(image)
                    batch_loss = loss_func(logps, label)
                    test_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == label.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Test loss: {test_loss/len(test_loader):.4f}.. "
                    f"Test Accuracy: {accuracy/len(test_loader):.4f}")
            values = [test_loss/len(test_loader), accuracy/len(test_loader), (time.time()-start_time)/60]
            writer.writerow(values)
    


# sample = train()
# sample.module("densenet161")
        


