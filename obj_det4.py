import torch
import numpy as np
import cv2

from time import time


class ObjectDetectorYOLOv5:
    """
    Class implements Yolo5 model to make inferences on a youtube video using Opencv2.
    """

    def __init__(self):
        """
        Initializes the class with youtube url and output file.
        :param url: Has to be as youtube URL,on which prediction is made.
        :param out_file: A valid output file name.
        """

        self.model = self.load_model()
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def get_video_from_webcam(self):
        """
        Creates a new video streaming object to extract video frame by frame to make prediction on.
        :return: opencv2 video capture object, with lowest quality frame available for video.
        """
        cap = cv2.VideoCapture(0)
        assert cap is not None
        return cap

    def load_model(self):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def load_model_locally(self, model_path="models/yolov5s.pt"):
        """
        Loads YOLOv5 model from a locally saved file.
        :param model_path: Path to the locally saved YOLOv5 model file.
        :return: Trained PyTorch model.
        """
        model = torch.load(model_path, map_location=self.device)['model']
        model.to(self.device).eval()
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                class_label = self.class_to_label(labels[i])
                confidence = row[4]
                label_with_prob = f"{class_label}: {confidence:.2f}"
                cv2.putText(frame, label_with_prob, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame

    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        cap = self.get_video_from_webcam()  # Open webcam (number may vary depending on your system)
        if not cap.isOpened():
            print("Error: Couldn't open webcam.")
            return

        while True:
            ret, frame = cap.read()  # Read frame from webcam
            if not ret:
                print("Error: Couldn't read frame from webcam.")
                break

            start_time = time()  # Measure the FPS
            results = self.score_frame(frame)  # Score the frame
            frame = self.plot_boxes(results, frame)  # Plot the boxes
            end_time = time()
            fps = 1 / np.round(end_time - start_time, 3)  # Measure the FPS
            print(f"Frames Per Second: {fps}")

            cv2.imshow("Object Detection", frame)  # Show the frame with detections

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key press
                break

        cap.release()
        cv2.destroyAllWindows()


# Create a new object and execute.
if __name__ == "__main__":
    detector = ObjectDetectorYOLOv5()
    detector()
