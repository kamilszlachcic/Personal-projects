Face Recognition App

📌 Project Description

This project uses a deep learning model to recognize faces based on the LFW dataset and new user images. It utilizes OpenCV for capturing camera input and ResNet18 as a neural network for face classification.

📂 Project Structure

├── FaceRecognition_app.py    # Real-time face recognition application
├── train_model.py            # Code for training the model using LFW and new images
├── model.py                  # Code for training the ResNet18 model on the LFW dataset
├── model.pth                 # Saved trained model
├── requirements.txt          # List of required libraries
├── README.md                 # Project documentation

🛠 Installation

To run the project, follow these steps:

Clone the repository

git clone https://github.com/your-repository.git
cd your-repository

Install required dependencies

pip install -r requirements.txt

Train the model (optional, if you want to train a new model)

python train_model.py

Run the face recognition application

python FaceRecognition_app.py

📷 How the Application Works

The application uses OpenCV to capture video from the camera.

It detects faces in the video stream and classifies them using the ResNet18 model.

If a face is recognized, the corresponding label is displayed with the message "Access Granted", otherwise "Access Denied".

Press q to exit the application.

⚙️ System Requirements

Python 3.8+

Torch, torchvision

OpenCV

Scikit-learn

Pillow

📌 Author

Project created by Kamil Szlachcic.