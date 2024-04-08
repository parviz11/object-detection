import cv2
import torch
import urllib
import numpy as np

class ObjectDetectorYOLOv5:
    def __init__(self, model_name="yolov5s"):
        # Load YOLOv5 model from torch hub
        self.model = torch.hub.load('ultralytics/yolov5', model_name)
        # Initialize classes
        self.classes = self.model.names if hasattr(self.model, 'names') else []
        


    def detect_objects(self, image):
        results = self.model(image)

       
        # Ensure that 'results' is a list (as it might contain multiple outputs for different scales)
        if isinstance(results, list):
            results = results[0]

        # Extract bounding boxes, confidences, and class indices
        boxes = results[:, :4].cpu().numpy()
        scores = results[:, 4].cpu().numpy()
        class_ids = results[:, 5].cpu().numpy()


        # Combine into a single array
        detections = np.column_stack([boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], scores, class_ids])
        return detections


    def draw_boxes(self, frame, boxes):
        # Get the dimensions of the frame
        frame_height, frame_width, _ = frame.shape
        print("Frame dimensions:", frame_width, "x", frame_height)

        for box in boxes:
            x, y, w, h = box[:4]  # Ensure correct unpacking
            x, y, w, h = map(int, [x, y, w, h])  # Ensure coordinates are integers
            
            # Scale the coordinates to match the frame's dimensions
            x1 = int(max(0, min(x * frame_width, frame_width - 1)))
            y1 = int(max(0, min(y * frame_height, frame_height - 1)))
            x2 = int(max(0, min((x + w) * frame_width, frame_width - 1)))
            y2 = int(max(0, min((y + h) * frame_height, frame_height - 1)))
        
            
            # Draw the bounding box on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Extract class probabilities
            class_probabilities = box[5:]
            print("Class probabilities:", class_probabilities)
            # Normalize the class probabilities
            class_probabilities = np.exp(class_probabilities) / np.sum(np.exp(class_probabilities))
            print("Normalized probabilities:", class_probabilities)

            # Extract class_id and confidence
            class_id = np.argmax(class_probabilities)
            confidence = class_probabilities[class_id]

            print("Class ID:", class_id)
            print("Confidence:", confidence)

            # Check if class_id is within the valid range
            if 0 <= class_id < len(self.classes):
                label = f"{self.classes[class_id]}: {confidence:.2f}"
            else:
                print(f"Unknown class_id: {class_id}")
                label = f"Unknown Class: {confidence:.2f}"

            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    def process_frame(self, frame):
        # Get the dimensions of the frame
        frame_height, frame_width, _ = frame.shape
        print("Frame dimensions:", frame_width, "x", frame_height)

        # Resize the frame to a size compatible with YOLOv5 (adjust the size accordingly)
        target_size = (640, 480)  # Adjust as needed
        resized_frame = cv2.resize(frame, target_size)

        # Display the resized frame (optional)
        cv2.imshow('Resized Frame', resized_frame)
        cv2.waitKey(1)  # Add a short delay to display the frame (optional)

        # Convert resized frame to a format compatible with YOLOv5
        image = resized_frame[:, :, ::-1].transpose(2, 0, 1)
        image = torch.as_tensor(image.astype("float32") / 255.0)

        # Add a batch dimension (unsqueeze)
        image = image.unsqueeze(0)

        # Detect objects
        boxes = self.detect_objects(image)

        # Draw bounding boxes on the original frame
        self.draw_boxes(frame, boxes)

        return frame

if __name__ == "__main__":
    # Create an ObjectDetectorYOLOv5 instance
    object_detector = ObjectDetectorYOLOv5()

    # Open webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Process the frame
        processed_frame = object_detector.process_frame(frame)

        # Display the processed frame
        cv2.imshow('Object Detection', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()
