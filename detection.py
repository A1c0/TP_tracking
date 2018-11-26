from imageai.Detection import VideoObjectDetection
import os
import cv2
from matplotlib import pyplot as plt


def forFrame(frame_number, output_array, output_count, returned_frame):
    for output in output_array:
        data = {'id': len(cameraDetector.detections), 'type': output.get('name'), 'box_points': output.get('box_points')}
        print(data)
        cameraDetector.detections.append(data)
        cv2.putText(returned_frame, str(data.get('id')), (data.get('box_points')[0], data.get('box_points')[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (72, 114, 255), thickness=4)
    cv2.imshow("Detection", returned_frame)
    cv2.waitKey(1)


class CameraDetector:
    detections = []

    def __init__(self, cam):
        self.execution_path = os.getcwd()

        self.color_index = {'bus': 'red', 'handbag': 'steelblue', 'giraffe': 'orange', 'spoon': 'gray', 'cup': 'yellow',
                            'chair': 'green', 'elephant': 'pink', 'truck': 'indigo', 'motorcycle': 'azure',
                            'refrigerator': 'gold', 'keyboard': 'violet', 'cow': 'magenta', 'mouse': 'crimson',
                            'sports ball': 'raspberry', 'horse': 'maroon', 'cat': 'orchid', 'boat': 'slateblue',
                            'hot dog': 'navy', 'apple': 'cobalt', 'parking meter': 'aliceblue', 'sandwich': 'skyblue',
                            'skis': 'deepskyblue', 'microwave': 'peacock', 'knife': 'cadetblue', 'baseball bat': 'cyan',
                            'oven': 'lightcyan', 'carrot': 'coldgrey', 'scissors': 'seagreen', 'sheep': 'deepgreen',
                            'toothbrush': 'cobaltgreen', 'fire hydrant': 'limegreen', 'remote': 'forestgreen',
                            'bicycle': 'olivedrab', 'toilet': 'ivory', 'tv': 'khaki', 'skateboard': 'palegoldenrod',
                            'train': 'cornsilk', 'zebra': 'wheat', 'tie': 'burlywood', 'orange': 'melon',
                            'bird': 'bisque', 'dining table': 'chocolate', 'hair drier': 'sandybrown',
                            'cell phone': 'sienna', 'sink': 'coral', 'bench': 'salmon', 'bottle': 'brown',
                            'car': 'silver', 'bowl': 'maroon', 'tennis racket': 'palevilotered',
                            'airplane': 'lavenderblush', 'pizza': 'hotpink', 'umbrella': 'deeppink', 'bear': 'plum',
                            'fork': 'purple', 'laptop': 'indigo', 'vase': 'mediumpurple', 'baseball glove': 'slateblue',
                            'traffic light': 'mediumblue', 'bed': 'navy', 'broccoli': 'royalblue',
                            'backpack': 'slategray', 'snowboard': 'skyblue', 'kite': 'cadetblue',
                            'teddy bear': 'peacock', 'clock': 'lightcyan', 'wine glass': 'teal',
                            'frisbee': 'aquamarine', 'donut': 'mincream', 'suitcase': 'seagreen', 'dog': 'springgreen',
                            'banana': 'emeraldgreen', 'person': 'honeydew', 'surfboard': 'palegreen',
                            'cake': 'sapgreen', 'book': 'lawngreen', 'potted plant': 'greenyellow', 'toaster': 'ivory',
                            'stop sign': 'beige', 'couch': 'khaki'}

        self.resized = False

        self.cap = cv2.VideoCapture(cam)

    def start(self):
        detector = VideoObjectDetection()
        detector.setModelTypeAsRetinaNet()
        detector.setModelPath(os.path.join(self.execution_path,
                                           "resnet50_coco_best_v2.0.1.h5"))
        detector.loadModel()
        detector.detectObjectsFromVideo(camera_input=self.cap,
                                        output_file_path=os.path.join(self.execution_path,
                                                                      "video_frame_analysis"),
                                        frames_per_second=30,
                                        per_frame_function=forFrame,
                                        minimum_percentage_probability=70,
                                        return_detected_frame=True)


if __name__ == "__main__":
    cameraDetector = CameraDetector(0)
    cameraDetector.start()
