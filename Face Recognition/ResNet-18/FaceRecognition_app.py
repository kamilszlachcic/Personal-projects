import cv2
import torch
from torchvision import models, transforms

# Load the pre-trained model
model = models.resnet18(num_classes=14)

# Load model weights and class mapping from the checkpoint
checkpoint = torch.load("model_updated.pth", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
class_mapping = checkpoint['class_mapping']

# Move the model to the appropriate device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()  # Set the model to evaluation mode

# Load Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Function to recognize a face
def recognize_face(image, model, class_mapping):
    """
    Recognizes a face using the trained model.

    Args:
        image (numpy array): The cropped face image.
        model (torch.nn.Module): The pre-trained model.
        class_mapping (dict): Mapping from class indices to class names.

    Returns:
        str: Recognition result ("Hello [Name]. Access Granted" or "Access Denied").
    """
    # Transform the image to match the model's input size
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((50, 37)),  # Resize images to 50x37 pixels
        transforms.ToTensor(),
    ])
    # Apply transformation and add batch dimension
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)  # Get the class with the highest score
        label_idx = pred.item()

    # Convert the class index to the corresponding name
    for name, idx in class_mapping.items():
        if idx == label_idx:
            return f"Hello {name}. Access Granted"
    return "Access Denied"

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Capture a frame from the webcam
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        # Crop the detected face
        face = frame[y:y + h, x:x + w]

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Recognize the face and print the result
        result = recognize_face(face, model, class_mapping)
        print(result)

    # Display the video feed with face detection
    cv2.imshow("Camera", frame)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
