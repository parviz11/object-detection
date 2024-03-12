import cv2
import torch
import numpy as np

class ObjectDetectorYOLOv5:
    def __init__(self, model_name="yolov5s"):
        # Load YOLOv5 model from torch hub
        self.model = torch.hub.load('ultralytics/yolov5', model_name)

    def detect_objects(self, image, confidence_threshold=0.3):
        results = self.model(image)

        # Ensure that 'results' is a list (as it might contain multiple outputs for different scales)
        if isinstance(results, list):
            results = results[0]
        
        # Filter detections based on confidence
        mask = results[:, 4] > confidence_threshold
        results = results[mask]


        # Extract bounding boxes, confidences, and class indices
        boxes = results[:, :4].cpu().numpy()
        scores = results[:, 4].cpu().numpy()
        class_ids = results[:, 5].cpu().numpy()

        # Combine into a single array
        detections = np.column_stack([boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], scores, class_ids])
        return detections

    def draw_boxes(self, frame, boxes):
        for box in boxes:
            x, y, w, h, confidence, class_id = box[:6]  # Ensure correct unpacking
            x, y, w, h = map(int, [x, y, w, h])  # Ensure coordinates are integers

            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Check if class_id is within the valid range
            if 0 <= int(class_id) < len(self.model.names):
                label = f"{self.model.names[int(class_id)]}: {confidence:.2f}"
            else:
                print(f"Unknown class_id: {class_id}")
                label = f"Unknown Class: {confidence:.2f}"

            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def process_frame(self, frame):
        # Resize the frame while maintaining the aspect ratio
        target_size = 640
        height, width, _ = frame.shape
        if height > width:
            scale = target_size / height
        else:
            scale = target_size / width
        resized_frame = cv2.resize(frame, (int(width * scale), int(height * scale)))

        # Pad the resized frame to achieve the target size
        pad_h = target_size - resized_frame.shape[0]
        pad_w = target_size - resized_frame.shape[1]
        padded_frame = cv2.copyMakeBorder(resized_frame, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

        # Convert resized frame to a format compatible with YOLOv5
        image = padded_frame[:, :, ::-1].transpose(2, 0, 1)
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
