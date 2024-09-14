#!/usr/bin/env python
# coding: utf-8

# # Date:14-09-2024.
# # Ritesh Kumar yadav- 22A91A4442.
# # kaushal kumar - 22A91A4426.
# 
# 
# 
# 
# # Project- Image Classification with Convolutional Neural Networks (CNNs):
# 
# 
# 
# 

# 
# # Description: Build a model to classify images into categories,
# # such as distinguishing between different types of animals or recognizing objects in images.
# # Tools: TensorFlow, Keras, PyTorch.
# # Datasets: CIFAR-10, MNIST, ImageNet etc.
# 
# # Overview:
# 
# # In this project, we'll build an image classification model using Convolutional Neural Networks (CNNs). We'll classify images into various categories, and the dataset used will be CIFAR-10. CNNs are the preferred model for image classification because they can efficiently capture spatial hierarchies and patterns in images
# 
# # Steps:
# # Understanding CNN Architecture: CNN consists of several key layers:
# 
# # Convolutional Layers: Extract features from input images using filters.
# # Pooling Layers:
# # Reduce dimensionality while preserving important features.
# # Fully Connected Layers: 
# # Final classification layers that output probabilities of each class.
# # Activation Functions:
# # Typically, ReLU is used to introduce non-linearity.
# # Softmax Layer:
# # For multi-class classification.
# # Dataset Selection:
# 
# # CIFAR-10:   
# # A dataset consisting of 60,000 32x32 color images in 10 classes, with 6,000 images per class.
# # Tools & Libraries:
# 
# # TensorFlow/Keras:    
# # For building and training the model.
# # PyTorch:   
# # An alternative deep learning library with dynamic computation graphs.
# 
# 
# # Input Layer (Image Data): CNNs take raw image data as input, typically represented as a matrix of pixel values (e.g., RGB images are 3D matrices with height, width, and color channels).
# 
# # Convolutional Layers: The core component of CNNs, convolutional layers apply filters (or kernels) that scan the image to detect specific features like edges, textures, or patterns. These filters move across the image, learning to identify various visual features.
# 
# # Activation Function (ReLU): After the convolution operation, the ReLU (Rectified Linear Unit) activation function is applied to introduce non-linearity, allowing CNNs to model complex patterns and relationships in the data.
# 
# # Pooling (Downsampling): Pooling layers, such as max pooling, reduce the spatial dimensions (height and width) of the feature maps. This helps in reducing computational complexity and extracting dominant features while maintaining spatial hierarchy.
# 
# # Feature Maps: Each convolutional layer generates multiple feature maps, each representing a specific feature detected in the image. As the network deepens, more abstract and complex features are captured.
# 
# # Fully Connected Layers: Towards the end of the CNN architecture, fully connected (dense) layers flatten the 2D feature maps into a 1D vector. This vector is used for classification based on the learned features from previous layers.
# 
# # Softmax Function: In multi-class classification tasks, the softmax function is used in the output layer to assign probabilities to each class. The class with the highest probability is considered the model’s prediction.
# 
# # End-to-End Learning: CNNs use backpropagation and gradient descent to adjust the weights of the filters during training. The network learns the optimal set of features for distinguishing between different classes by minimizing the loss function.
# 
# # Transfer Learning: Pre-trained CNNs, such as VGG, ResNet, or Inception, are often used for transfer learning. These models, trained on large datasets like ImageNet, can be fine-tuned on smaller datasets to improve performance on specific tasks.
# 
# # Robustness to Spatial Variations: CNNs are robust to variations in position, scale, and orientation of objects in images due to the shared weights and local receptive fields in convolutional layers. This makes CNNs well-suited for tasks like object recognition and classification.
# 
# 
# # applications 
# # Healthcare and Medical Imaging:
# # Disease Detection: CNNs can classify medical images (e.g., X-rays, MRIs, CT scans) to detect diseases such as cancer, pneumonia, or tumors. For example, a CNN can be used to identify malignant versus benign tumors in mammograms.
# # Organ Segmentation: Classifying different organs or tissues in medical scans for surgeries or diagnostics.
# #  Automotive Industry:
# # Autonomous Vehicles: CNNs are used in self-driving cars to classify objects like pedestrians, traffic signs, and other vehicles, enabling the car to make decisions based on its surroundings.
# # Vehicle Identification: Used for identifying different types of vehicles in images for security purposes or toll collection.
# #  Agriculture:
# # Plant Disease Detection: CNNs can classify images of plants to detect diseases or nutrient deficiencies, allowing farmers to take action early and increase crop yield.
# # Crop Monitoring: Classification of aerial or satellite images to monitor the health of crops or classify different types of plants.
# # Retail and E-Commerce:
# # Product Categorization: CNNs can automatically classify and tag images of products for e-commerce websites, enhancing search functionality and user experience.
# # Visual Search: Customers can upload photos to search for similar products in the store, using CNNs to match images with product inventory.
# # Security and Surveillance:
# # Facial Recognition: CNNs are widely used in security systems for recognizing faces in real-time, allowing access control or identifying individuals in crowds.
# # Intruder Detection: Used in surveillance systems to classify and identify suspicious activities, objects, or individuals in real-time video feeds.
# # Social Media and Content Moderation:
# # Image Tagging: Social media platforms use CNNs to automatically tag objects, people, or places in photos, making it easier for users to organize and search content.
# # Content Moderation: CNNs can automatically detect inappropriate or harmful content (e.g., violence, nudity) in uploaded images and flag it for removal.
# #  Manufacturing and Quality Control:
# # Defect Detection: CNNs are used to classify images of products to identify defects or faults in manufacturing processes, improving product quality and reducing waste.
# # Automation: Classifying objects for sorting, packaging, or quality assessment in automated production lines.
# #  Fashion and Clothing Industry:
# # Outfit Recommendation: CNNs can classify clothing images, allowing fashion apps or websites to recommend outfits based on user preferences or detected styles.
# # Virtual Try-Ons: CNNs can help classify clothing items to simulate how they would look on a person, enabling virtual try-on applications.
# # Environmental Monitoring:
# # Wildlife Conservation: CNNs can classify species of animals from camera trap images to monitor populations, track endangered species, or detect poaching activities.
# # Land Use and Land Cover Classification: Classifying satellite or drone images to analyze changes in land use (e.g., urbanization, deforestation, or flood zones).
# #  Education and Research:
# # Digital Learning: CNNs can be used to classify handwritten digits, objects, or drawings in educational tools, making it easier to develop digital learning platforms.
# # Research Projects: Image classification using CNNs can be applied in computer vision research, improving methods for understanding and interacting with visual data.
# #  Art and Culture:
# # Art Style Classification: CNNs can classify different styles of artwork (e.g., Impressionism, Baroque) based on patterns in the paintings, helping with art cataloging or identification.
# # Object Restoration: Classifying damaged parts of historical artifacts or artworks to guide restoration processes.
# # Financial and Legal Documents:
# # Signature Verification: Classifying handwritten or electronic signatures in documents to verify their authenticity for legal purposes.
# # Document Organization: Automatically classifying scanned legal or financial documents based on content or format.
# #  Robotics:
# # Object Detection and Classification: CNNs enable robots to detect and classify objects in their environment, making them useful for tasks like sorting, assembling, or navigating through spaces.
# #  Gaming and Augmented Reality (AR)/Virtual Reality (VR):
# # Gesture Recognition: Classifying hand or body gestures in real-time for use in gaming or AR/VR systems.
# # Object Interaction: In AR/VR, CNNs can classify objects in the environment, allowing for more immersive and interactive experiences.
# #  Remote Sensing and Geospatial Analysis:
# # Satellite Image Classification: CNNs are used to classify landforms, vegetation types, or urban areas in satellite or aerial imagery for environmental or urban planning purposes.
# 
# # Step-by-Step Implementation Using TensorFlow/Keras:

# In[2]:


# %pip install tensorflow numpy matplotlib

#Import Required Libraries
# TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np


# In[3]:


#Load and preprocess the CIFAR-10 dataset
# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# Normalize the pixel values between 0 and 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)


# In[4]:


x_train.shape, y_train.shape, x_test.shape, y_test.shape


# In[5]:


y_train[:5]


# In[6]:


classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# In[12]:


def plot_images(x, y, index):
    plt.figure(figsize=(15,2))
    plt.imshow(x[index])
    plt.xlabel(classes[np.argmax(y[index])])
    
        
plot_images(x_train, y_train, 1)
def plot_images(x, y, index):
    plt.figure(figsize=(15,2))
    plt.imshow(x[index])
    plt.xlabel(classes[np.argmax(y[index])])
    
        
plot_images(x_train, y_train, 0)
plot_images(x_train, y_train, 2)
plot_images(x_train, y_train, 3)
plot_images(x_train, y_train, 4)
plot_images(x_train, y_train, 5)
plot_images(x_train, y_train, 6)
plot_images(x_train, y_train, 7)
plot_images(x_train, y_train, 8)
plot_images(x_train, y_train, 9)
plot_images(x_train, y_train, 10)
plot_images(x_train, y_train, 11)
plot_images(x_train, y_train, 12)
plot_images(x_train, y_train, 13)
plot_images(x_train, y_train, 5999)


# # CNNs automatically learn relevant features from the images without requiring manual feature extraction. This leads to better accuracy in classification tasks compared to traditional methods.
# # Handling Complex Images:
# 
# # CNNs are capable of learning hierarchical patterns, detecting simple features (like edges) in the initial layers and more complex patterns (like faces or objects) in deeper layers. This makes CNNs highly effective in processing complex images.
# # Reduced Preprocessing Requirements:
# 
# # Unlike other machine learning algorithms, CNNs require minimal preprocessing of images. They can automatically learn features such as shape, texture, and object boundaries from raw pixel values.
# # Scalability and Efficiency:
# 
# # CNNs are scalable and work well with large image datasets. By using techniques like pooling, they reduce the computational cost while retaining important features from the images.
# # Robustness to Variations:
# 
# # CNNs are robust to variations in translation, scaling, rotation, and other spatial distortions in images. This makes them suitable for real-world image classification tasks where objects may appear in different orientations or lighting conditions.
# # Transfer Learning:
# 
# # CNN models pre-trained on large datasets like ImageNet can be fine-tuned for specific tasks, allowing developers to leverage powerful models without needing extensive datasets. This makes it easier to build applications with limited training data.
# # Real-Time Applications:
# 
# # CNNs are widely used in real-time applications such as face recognition, object detection, autonomous driving, and medical imaging. Their fast inference and accuracy are crucial for time-sensitive tasks.
# # Automated Systems:
# 
# # Image classification with CNNs is essential for building automated systems, such as sorting or defect detection in manufacturing, enabling machines to make decisions based on visual input.
# # Wide Application Areas:
# 
# # CNNs are versatile and are used in various domains like healthcare (medical image analysis), security (facial recognition), agriculture (crop monitoring), and more.
# # Improves Human Efficiency:
# 
# # In fields like medical diagnostics, CNNs assist professionals by accurately identifying patterns in medical images (e.g., MRI, X-ray). This leads to quicker and more reliable diagnoses, improving human efficiency.
# 
# 
# # Explanation: 
# # The CIFAR-10 dataset is divided into training and testing sets.
# # We normalize the pixel values to a range of [0, 1] to speed up convergence.
# # The labels are one-hot encoded (10 classes).

# # Build the CNN Model

# In[22]:


# Define the CNN architecture
model = models.Sequential()

# Input layer
model.add(layers.Input(shape=(32, 32, 3)))

# First Convolutional Layer
model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Second Convolutional Layer
model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Third Convolutional Layer
model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Flatten the feature maps before feeding into fully connected layers
model.add(layers.Flatten())

# Fully Connected Dense Layer
model.add(layers.Dense(128, activation='relu'))

# Output Layer (10 classes for CIFAR-10)
model.add(layers.Dense(10, activation='softmax'))



# # Explanation:
# # The CNN model has three convolutional layers, each followed by a pooling layer. After convolution and pooling, the feature maps are flattened, and the fully connected layers (dense layers) are added. The final output layer uses the softmax activation to classify the image into one of the 10 categories.

# # Compile the Model

# In[23]:


# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# # Explanation: 
# # The model is compiled using the Adam optimizer, which is efficient and widely used for deep learning models. The loss function used is categorical cross-entropy since this is a multi-class classification problem, and accuracy is the metric for performance evaluation.

# In[24]:


# Train the model
history = model.fit(x_train, y_train, epochs=10, 
                    validation_data=(x_test, y_test), batch_size=64)


# # Explanation: 
# # the model is trained on the CIFAR-10 training data for 10 epochs with a batch size of 64 We use the test data as validation data to monitor performance on unseen data during training.

# # Evaluate the Model

# In[25]:


# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc * 100:.2f}%")


# # Explanation:
# # After training, we evaluate the model’s accuracy on the test data to see how well it generalizes to new images.

# # Visualize Training History
# # # Plot the training and validation accuracy and loss over time

# In[26]:


# Plot the training and validation accuracy and loss over time
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# # Explanation: 
# # This visualizes the training and validation accuracy and loss over epochs, helping us understand the model's performance and whether it’s overfitting or underfitting

# # PyTorch Implementation:
# # Import Required Libraries

# In[30]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


# # Load and Preprocess Data

# In[32]:


# Transform: Normalize and convert to tensor
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)


# # Explanation
# # transforms.Compose:
# # This combines several transformations into a single pipeline. Each transformation is applied in the order they are passed to Compose.
# 
# # transforms.ToTensor(): 
# # This transformation converts a PIL image or NumPy array to a PyTorch tensor. The pixel values, originally in the range [0, 255], are scaled to [0, 1].
# 
# # transforms.Normalize((mean), (std)): 
# # This normalizes the image tensor. Each channel (Red, Green, Blue) is normalized independently using the formula:
# 
# # Normalized_value=Original_value−mean   /   std
# 
#  
# # Here, the mean and standard deviation for each channel are 0.5. This essentially scales the data from [0, 1] to approximately [-1, 1], which helps the model converge faster during training.

# # torchvision.datasets.CIFAR-10:
# # This function is used to load the CIFAR-10 dataset.
# # It has two parameters that control how data is loaded:
# 
# # train=True: 
# # This specifies that you're loading the training set. For the test set, it is set to False.
# # root='./data':
# # This is the directory where the dataset will be saved.
# # download=True:
# # This ensures that the dataset will be downloaded if it is not already present in the specified root directory.
# # transform=transform:
# # The data transformation pipeline (defined in step 1) is applied to every image as it’s loaded.
# # Training Data (trainset):
# # This contains the CIFAR-10 training images and labels.
# 
# # Test Data (testset): 
# # This contains the CIFAR-10 test images and labels.

#  # DataLoader for Batching:
#  # torch.utils.data.DataLoader: 
#  # This function is used to create an iterable (i.e., a loader) for the dataset, which allows for efficient batch loading and shuffling of the data during training. It takes a few important parameters:
# 
# # trainset/testset: 
# # This specifies the dataset (either the training or test set) to load.
# 
# # batch_size=64: 
# # This specifies the number of images in each batch. A batch size of 64 means that the model will process 64 images at a time before updating its weights.
# 
# # shuffle=True/False:
# 
# # For the training set (trainloader), shuffle=True means that the data is shuffled before each epoch. Shuffling ensures that the model doesn't learn the order of the images, improving its generalization.
# 

# # Define the CNN Model

# In[33]:


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# # Explanations
# # This CNN class defines a Convolutional Neural Network (CNN) using PyTorch. It has three convolutional layers (conv1, conv2, and conv3) and two fully connected (linear) layers (fc1 and fc2).
# 
# # Convolutional Layers (conv1, conv2, conv3): 
# # Each layer performs a 2D convolution, extracting features from the input image. The padding=1 ensures that the output spatial size remains consistent.
# # MaxPooling (pool):
# # After each convolution, max-pooling reduces the spatial dimensions of the feature maps by half.
# # Fully Connected Layers: 
# # After flattening the output of the third convolution (x.view(-1, 128 * 4 * 4)), two fully connected layers are used to produce the final 10-class output.
# # Activation Functions: 
# # ReLU is applied after each convolution and fully connected layer to introduce non-linearity.
# # Forward Method: 
# # This method defines the forward pass through the network, processing the input data through the layers sequentially

# 
#  # Training the Model:

# 
# # Initialize the model, loss function, and optimizer

# In[34]:


# Initialize the model, loss function, and optimizer
import torch.nn.functional as F
net = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Train the model
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    print(F"Epoch {epoch+1}, Loss: {running_loss / len(trainloader)}")


# # Explanations:
# # Model, Loss Function, and Optimizer Initialization:
# # net = CNN() initializes an instance of the CNN model defined earlier.
# # criterion = nn.CrossEntropyLoss() is a suitable loss function for classification tasks. It computes the difference between the predicted and true labels.
# # optimizer = optim.Adam(net.parameters(), lr=0.001): Adam optimizer is used to adjust model parameters (weights) during training based on the gradients computed from the loss. It’s popular because of its adaptive learning rate capabilities.
# # Training Loop:
# # for epoch in range(10):: The loop runs for 10 epochs, meaning the model will see the entire dataset 10 times.
# # running_loss = 0.0: Initializes the running loss for each epoch to keep track of the total loss for all batches processed during the epoch
# # Batch Processing:
# # for i, data in enumerate(trainloader, 0):: Loops through each batch of training data in the trainloader.
# # trainloader provides data in small batches to avoid memory overload.
# # inputs, labels = data: Each batch contains a set of images (inputs) and their corresponding true labels (labels).
# # Zero Gradients:
# # optimizer.zero_grad(): This clears any previously calculated gradients before computing new ones. Without this, gradients would accumulate across batches, leading to incorrect updates.
# # Forward Pass and Loss Calculation:
# # outputs = net(inputs): This performs a forward pass through the model, where the input images are processed, and predictions (outputs) are made.
# # loss = criterion(outputs, labels): The CrossEntropyLoss computes the difference between the predicted outputs and the true labels, returning a scalar loss value.
# 
# 
# # Backward Pass and Parameter Update:
# # loss.backward(): This computes the gradients of the loss with respect to each parameter (weight) in the model using backpropagation.
# # optimizer.step(): The optimizer adjusts the weights of the model based on the gradients calculated in the previous step.
# # Track Running Loss:
# # running_loss += loss.item(): loss.item() extracts the scalar value of the loss, and this is accumulated in running_loss to track the loss over all batches.
#  # Print Epoch Loss:
#  # At the end of each epoch, the average loss across all batches is printed. This gives an idea of how well the model is learning. Lower loss values indicate better performance.

# In[35]:


# Test the model
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total}%")


# # Explanations:
# # with torch.no_grad():: Disables gradient calculation during testing to save memory and computation.
# # for data in testloader:: Loops through batches of test data.
# # images, labels = data: Extracts images and their true labels from the batch.
# # outputs = net(images): Performs a forward pass through the model to get predictions.
# # predicted = torch.max(outputs.data, 1): Finds the class with the highest predicted score for each image.
# # total += labels.size(0): Keeps track of the total number of test samples.
# # correct += (predicted == labels).sum().item(): Counts how many predictions match the true labels.
# # print(f"Accuracy: {100 * correct / total}%"): Calculates and prints the overall accuracy of the model in percentage

# 
# # Project Summary: Image Classification Using CNN with PyTorch

# # In this project, a Convolutional Neural Network (CNN) was built to classify images from the CIFAR-10 dataset into 10 categories, such as airplanes, cars, birds, and cats. The project involved several key steps:

# # Data Loading and Preprocessing: The CIFAR-10 dataset was loaded using torchvision.datasets. The images were normalized, and data loaders were created for both training and testing using torch.utils.data.DataLoader.
# 
# # CNN Model Architecture: A custom CNN was designed with three convolutional layers followed by ReLU activations and max-pooling. These layers extract features from the images. After flattening the output, two fully connected layers were used to predict the class labels.
# 
# # Training the Model: The model was trained for 10 epochs. In each epoch, batches of images were passed through the network, the loss was calculated using CrossEntropyLoss, and the Adam optimizer was used to update the model's parameters. The loss was printed at the end of each epoch to monitor the training progress.
# 
# # Testing the Model: After training, the model was evaluated on the test set. The forward pass was performed on the test data without computing gradients. The predicted labels were compared to the true labels to calculate the accuracy.
# 
# # Results: The overall accuracy of the model was printed, providing a measure of its performance on unseen data.

# # This project demonstrated how to implement a CNN from scratch in PyTorch for an image classification task, highlighting key concepts like data preprocessing, model design, backpropagation, and evaluation
# 
# # Automation: By using CNNs for image classification, tasks that traditionally required human effort, such as sorting, tagging, or identifying objects, can be automated, saving time and costs.
# # Accuracy: CNNs achieve high accuracy in recognizing patterns and classifying objects, making them reliable for applications in critical sectors like healthcare, automotive, and security.
# # Scalability: CNN-based solutions can handle vast amounts of data, making them scalable for large applications like e-commerce platforms, social media, and satellite image analysis.
# # Real-Time Applications: CNNs can process and classify images in real-time, making them ideal for use in dynamic environments such as autonomous vehicles or security systems.

# # The End .
# # Thank You sir.
