
# Fruit Image Classification with PyTorch

This project is a simple fruit image classification task built using **Python** and **PyTorch**. The goal of this project is to train a Convolutional Neural Network (CNN) model to classify images of fruits into different categories, such as **Apple**, **Banana**, **Grape**, **Mango**, and **Strawberry**.

## Project Overview

The dataset consists of images of various fruits, and the model is trained to recognize and classify these images based on their respective labels. The project demonstrates how to use **PyTorch** for:

- Building a custom dataset and applying data transformations (e.g., resizing, tensor conversion)
- Training a CNN model for image classification
- Evaluating the model's performance on validation and test datasets
- Saving the trained model for future use

## Installation

To get started with this project, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/fruit-classification.git
   cd fruit-classification
2. **Install the necessary dependencies:**
You can install the required libraries using pip:
     ```bash
     pip install -r requirements.txt
     Here is the list of required packages:

   torch
   torchvision
   Pillow
   numpy
   matplotlib

If you don't have a requirements.txt file yet, you can create it by running pip freeze > requirements.txt.

Dataset
The dataset used in this project contains images of fruits stored in different folders, each corresponding to a fruit class. For example:

   ```bash
   data/
     train/
       Apple/
       Banana/
       Grape/
       Mango/
       Strawberry/
     valid/
       Apple/
       Banana/
       Grape/
       Mango/
       Strawberry/
     test/
       Apple/
       Banana/
       Grape/
       Mango/
       Strawberry/
Please make sure that you have the dataset organized in this format and update the paths in the code accordingly.

1. Usage
Training the Model:

To train the model, run the following Python script:

```bash
python train.py
This will start training the Convolutional Neural Network (CNN) model for 10 epochs. During training, the loss and validation accuracy will be printed for each epoch.

2. Model Evaluation:

After training, the model is evaluated on the test dataset to check its accuracy. You can also visualize the predictions if needed.

3. Saving the Model:

Once training is complete, the model's state_dict (parameters) will be saved in the fruit_classifier.pth file.

4. Loading the Model:

You can load the trained model using the following code:

python
model = SimpleCNN()
model.load_state_dict(torch.load('fruit_classifier.pth'))
model.eval()
Project Structure
graphql

fruit-classification/
├── data/                # Dataset folder containing train, valid, test data
├── train.py             # Script to train the model
├── model.py             # Define the CNN model (SimpleCNN)
├── requirements.txt     # List of dependencies
├── README.md            # Project documentation
└── fruit_classifier.pth # Trained model weights
License
This project is open-source and available under the MIT License.

Acknowledgments
Thanks to the contributors and open-source community for the resources and libraries that made this project possible.
Special thanks to the dataset providers for making the fruit image dataset available.
markdown
Copy code

### **Explanation:**

- **Overview**: Provides a brief introduction to the project, the tools and libraries used, and the objective.
- **Installation**: Explains how to set up the project locally by cloning the repository and installing dependencies.
- **Dataset**: Describes the expected structure of the dataset, which should be organized into subfolders for training, validation, and testing.
- **Usage**: Details the commands to run the training process, how to evaluate the model, and how to save/load the trained model.
- **Project Structure**: Lists the structure of the project folder.
- **License**: Indicates that the project is open-source and specifies the license (MIT in this case).
- **Acknowledgments**: Gives credit to contributors and external resources.
