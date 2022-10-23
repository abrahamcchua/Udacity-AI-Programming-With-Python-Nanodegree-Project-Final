# Built in modules
import time
import csv
# Third Party Modules
import torch
from torchvision import transforms, datasets, models
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
# Own modules
from input_args import input_args_train

'''
The training script for Project #2 for the AWS Udacity AI Programming With Python Nanodegree 
:Date: October 17, 2022
:Author/s: 
    - Chua, Abraham <abrahamcchua@gmail.com>
'''

class Train():
    def __init__(self, arguments):
        data_dir = arguments.dir
        self.save_dir = arguments.save_dir
        self.arch = arguments.arch
        self.lr = arguments.learning_rate
        self.hlayer_one = arguments.hlayer_one
        self.hlayer_two = arguments.hlayer_two
        self.epochs = arguments.epochs
        gpu = arguments.gpu
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
    
    
    def module(self):
        """Main highlight of function "module"
        Summary:
        The frame where in the different functions are used
        output of the function includes:
        None
        """
        train_data_loader = self.train_data_loader(self.train_data)
        validation_data_loader = self.validation_data_loader(self.validation_data)
        test_data_loader = self.test_data_loader(self.test_data)
        model = self.model_architecture(self.arch, self.hlayer_one, self.hlayer_two)
        trained_model = self.trainer(model, train_data_loader, validation_data_loader, self.loss_func, self.arch, epochs=self.epochs, lr=self.lr)
        self.tester(trained_model, self.arch, test_data_loader, self.loss_func)
 
        
    # Separated from other data loaders if the user wants to change the batch size individually and for readability
    def train_data_loader(self, train_data, batch_size=64):
        """Main highlight of function "train_data_loader"
        Summary:
        The loader function for the train data
        :param train_data: string; The dataset used for training
        :param batch_size: int; The number of data points used per training epoch
        output of the function includes:
        A wrapped iterable around the training data
        """
        return torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)


    def validation_data_loader(self, validation_data, batch_size=64):
        """Main highlight of function "validation_data_loader"
        Summary:
        The loader function for the validation data
        :param validation_data: string; The dataset used for validation
        :param batch_size: int; The number of data points used per training epoch
        output of the function includes:
        A wrapped iterable around the validation data
        """
        return torch.utils.data.DataLoader(validation_data, batch_size, shuffle=True)
    
    
    def test_data_loader(self, test_data, batch_size=64):
        """Main highlight of function "test_data_loader"
        Summary:
        The loader function for the test data
        :param test_data: string; The dataset used for testing
        :param batch_size: int; The number of data points used per training epoch
        output of the function includes:
        A wrapped iterable around the test data
        """
        print(type(test_data))
        return torch.utils.data.DataLoader(test_data, batch_size, shuffle=True)
   
        
    def model_architecture(self, pretrained_model, hidden_layer_one, hidden_layer_two):
        """Main highlight of function "model_architecture"
        Summary:
        The architecture to be used for the neural network
        :param pretrained_model: string; The name of the pretrained model to be used for transfer learning
        :param hidden_layer_one: int; The number of neurons of the first hidden layer
        :param hidden_layer_two: int; The number of neurons of the second hidden layer
        output of the function includes:
        The model architecture
        """
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
        """Main highlight of function "trainer"
        Summary:
        The training script for transfer learing
        :param model: Tensors; The model architecture to be used
        :param train_loader: iterable; A wrapped iterable around train data
        :param validation_loader: iterable; A wrapped iterable around validation data
        :param loss_func: loss function; loss function to be used in the model
        :param pretrained_model: string; The name of the pretrained model to be used for transfer learning
        :param filename: string; The prefix of the csv filename 
        :param epochs: int; The number of times the model learns from the data set
        :param lr; float; The learning rate of the model
        :param momentum; float; The momentum to be used for the optimizer
        :param step_size; int; number of steps for the learning rate to change
        :param gamma; float; rate at which the learning rate is decayed per step size
        output of the function includes:
        A trained model saved into checkpoint file, with its evalution metrics printed as well as saved in a csv file
        """
        with open(filename+"_"+pretrained_model+".csv", "w") as f:
            # Writing details into csv
            writer = csv.writer(f)
            parameters = ["Parameters:" ,epochs, lr, momentum, step_size, gamma]
            header = ["Epoch", "Train Loss", "Validation Loss", "Validation_Accuracy", "Time"]
            writer.writerow(parameters)
            writer.writerow(header)
            # Initializing the optimizer to be used and learning rate changes
            optimizer = optim.SGD(model.classifier.parameters(), lr, momentum)
            lr_change = lr_scheduler.StepLR(optimizer, step_size, gamma)
            for epoch in range(epochs):
                #Evaluation metric to be used later
                start_time = time.time()
                training_loss = 0
                validation_loss = 0
                accuracy = 0
                # Training Proper
                for image, label in train_loader:
                    # Move input and label tensors to the specified device
                    image, label = image.to(self.device), label.to(self.device)
                    
                    result = model.forward(image)
                    loss = loss_func(result, label)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    training_loss += loss.item()
                lr_change.step()
                # Validation
                model.eval()
                with torch.no_grad():
                    for image, label in validation_loader:
                        image, label = image.to(self.device), label.to(self.device)
                        logps = model.forward(image)
                        batch_loss = loss_func(logps, label)
                        validation_loss += batch_loss.item()
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        _, top_class = ps.topk(1, dim=1)
                        equals = top_class == label.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                # Evalution metrics
                values = [epoch+1, training_loss, validation_loss, accuracy/len(validation_loader), (time.time()-start_time)/60]
                writer.writerow(values)
                print(f"Epoch {epoch+1}/{epochs}.. "
                        f"Train loss: {training_loss:.4f}.. "
                        f"Validation loss: {validation_loss:.4f}.. "
                        f"Validation Accuracy: {accuracy/len(validation_loader):.4f}..")
                print(f"Time it took for this epoch: {time.time() - start_time: .4f}")
                # Change model to training mode
                model.train()
        self.save_checkpoint(model, epochs, optimizer, pretrained_model, lr)
        return model
    
    
    def save_checkpoint(self, model, epochs, optimizer, pretrained_model, lr):
        """Main highlight of function "save_checkpoint"
        Summary:
        Saves the current model to be used later
        :param model: tensor; The model to be saved
        :param epochs: int; The number of epochs used to train the model
        :param optimizer: tensor; The optimizer used to train the model
        :param lr; float; The learning rate of the model
        output of the function includes:
        A .pth file that contains the model parameters.
        """
        model.class_to_idx = self.train_data.class_to_idx
        model_state = {
        'epoch': epochs,
        'state_dict': model.state_dict(),
        'optimizer_dict': optimizer.state_dict(),
        'classifier': model.classifier,
        'class_to_idx': model.class_to_idx,
        }
        torch.save(model_state, self.save_dir+'checkpoint-'+ pretrained_model+"_e_"+str(epochs)+"_lr_"+str(lr)+'.pth')
    
    
    def tester(self, model, pretrained_model, test_loader, loss_func, filename="testing_results"):
        """Main highlight of function "tester"
        Summary:
        Testing the specified model on the test data set
        :param model: tensor; The model to be used to evaluate the training data set
        :param pretrained_model; string; The name of the pretrained_model
        :param test_loader; A wrapped iterable around the test data
        :param loss_func; loss function; The loss function to be used in testing
        :param filename: string; The prefix of the csv filename
        output of the function includes:
        The model architecture
        """
        with open(filename+"_"+pretrained_model+".csv", "w") as f:
            # Writing details into the csv
            writer = csv.writer(f)
            header = ["Test Loss", "Test_Accuracy", "Time"]
            writer.writerow(header)
            # Initializing metrics for evaluation
            test_loss = 0
            accuracy = 0
            # Evaluation proper
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
                    _, top_class = ps.topk(1, dim=1)
                    equals = top_class == label.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            # Evaluation metrics        
            print(f"Test loss: {test_loss/len(test_loader):.4f}.. "
                    f"Test Accuracy: {accuracy/len(test_loader):.4f}")
            values = [test_loss/len(test_loader), accuracy/len(test_loader), (time.time()-start_time)/60]
            writer.writerow(values)
            
            
            
if __name__ == "__main__":
    arguments = input_args_train()
    trainer = Train(arguments)
    trainer.module()
            


