import argparse


def input_args_train():
    parser = argparse.ArgumentParser()
    # Argument #1: Path to Data Folder
    parser.add_argument('--dir', type=str, help='path to the folder of pet images', required=True) 
    # Argument #2: Path to the save directory
    parser.add_argument('--save_dir', type=str, default="checkpoints", help ='path to the save directory')
    # Argument #3: Model Architechture to be used
    parser.add_argument('--arch', type=str, default='densenet161', help='type of densenet to be used')
    # Argument #4: Learning rate of the model
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate of the model')
    # Argument #5: Size of the first hidden layer in the model
    parser.add_argument('--hlayer_one', type=int, default=810, help='Size of the first hidden layer of the model')
    # Argument #6: Size of the second hidden layer in the model
    parser.add_argument('--hlayer_two', type=int, default=270, help='Size of the second hidden layer of the model')
    # Argument #7: Number of epochs the model will be trained
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs the model will train on')
    # Argument #8: Determines if a gpu will be used to rain the model
    parser.add_argument('--gpu', type=str, default="default", help='device that the model be trained on')
    return parser.parse_args()

def input_args_predict():
    parser = argparse.ArgumentParser()
    # Argument #1: Name of .pth file
    parser.add_argument('--checkpoint', type=str, help='name of pth file', required=True)
    # Argument #2: File Path of the image to be classified
    parser.add_argument('--image', type=str, default="random", help='name of pth file')
    # Argument #3: Number of top_k classes and their corresponding probabilities to be returned
    parser.add_argument('--topk', type=int, default=5, help ='top_k classes of the model')
    # Argument #4: Determines the file location of the classes 
    parser.add_argument('--category_names', type=str, default="cat_to_name.json", help ='File containing the mapping of the class names')
    # Argument #5: Determines if a gpu will be used for the inference
    parser.add_argument('--gpu', type=str, default="default", help='device that the inference will use')
    # Argument #6: Path to the save directory
    parser.add_argument('--save_dir', type=str, default="checkpoints", help ='path to the save directory')
    return parser.parse_args()
