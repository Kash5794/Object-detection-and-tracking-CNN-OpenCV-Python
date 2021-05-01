import numpy as np
import cv2
import matplotlib




#Graphical User Interface
import tkinter as tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
from tkinter import messagebox

window = tk.Tk()
window.geometry("500x400")

window.configure(bg='#856ff8')
Labelframe = tk.Frame(window)
label=tk.Label(Labelframe,text="Object Detection and Tracking System", bg="red",fg="white")
label.config(font=("Courier", 15))
label.pack()
Labelframe.pack()


def open_file_img():
    """Open a file for editing."""
    filepath = askopenfilename(
        filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
    )
    if filepath:
        fileFormat = str(filepath).split('.')[-1]
        while fileFormat.casefold() != "jpg" and fileFormat.casefold() != "png":
            messagebox.showwarning("showwarning", "File Not Supported")
            filepath = askopenfilename(
                filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
            )
            if not filepath:
                break

            fileFormat = str(filepath).split('.')[-1]

    img = cv2.imread(filepath)

    classLabels = []
    file = 'coco.names'
    with open(file, "rt") as f:
        classLabels = f.read().rstrip('\n').split('\n')

    configPath = '2ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = '2frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)


    classIds, confidences, bounding_boxes = net.detect(img, confThreshold=0.5)
    print(classIds, bounding_boxes)
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confidences.flatten(), bounding_boxes):
            if classId <= 80:
                cv2.rectangle(img, box, color=(0, 255, 255), thickness=2)
                cv2.putText(img, classLabels[classId - 1], (box[0] + 10, box[1] + 30),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Output", img)



def open_file_video():
    """Open a file for editing."""
    filepath = askopenfilename(
        filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
    )
    if filepath:
        fileFormat=str(filepath).split('.')[-1]
        print("fileFormat")
        while fileFormat.casefold()!="mp4":
            messagebox.showwarning("showwarning", "File Not Supported")
            filepath = askopenfilename(
                filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
            )
            if not filepath:
                break
            fileFormat = str(filepath).split('.')[-1]


    cap = cv2.VideoCapture(str(filepath))
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open video")

    cap.set(3, 640)
    cap.set(4, 480)

    classLabels = []
    file = 'coco.names'
    with open(file, "rt") as f:
        classLabels = f.read().rstrip('\n').split('\n')

    configPath = '2ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = '2frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    while (True):
        success, img = cap.read()
        classIds, confidences, bounding_boxes = net.detect(img, confThreshold=0.5)
        print(classIds, bounding_boxes)
        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confidences.flatten(), bounding_boxes):
                if classId <= 80:
                    cv2.rectangle(img, box, color=(0, 255, 255), thickness=2)
                    cv2.putText(img, classLabels[classId - 1], (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Output", img)
        cv2.waitKey(10)
        if cv2.getWindowProperty('Output', 4)!=1 :
            break

    cap.release()
    cv2.destroyAllWindows()


def open_webcam():

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open video")

    cap.set(3, 640)
    cap.set(4, 480)

    classLabels = []
    file = 'coco.names'
    with open(file, "rt") as f:
        classLabels = f.read().rstrip('\n').split('\n')

    configPath = '2ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = '2frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    while True:
        (success, img) = cap.read()
        classIds, confidences, bounding_boxes = net.detect(img, confThreshold=0.5)
        print(classIds, bounding_boxes)
        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confidences.flatten(), bounding_boxes):
                if classId <= 80:
                    cv2.rectangle(img, box, color=(0, 255, 255), thickness=2)
                    cv2.putText(img, classLabels[classId - 1], (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Output", img)
        cv2.waitKey(10)
        if cv2.getWindowProperty('Output', 4)!=1 :
            break

    cap.release()

    cv2.destroyAllWindows()



webCamFrame=tk.Frame(window,relief=tk.RAISED, bd=2)
webCamBtn = tk.Button(webCamFrame,text="Connect to WebCam", command=open_webcam,bg='black',fg='white')
webCamFrame.pack(pady=20)
webCamBtn.pack()

importImageFrame=tk.Frame(window,relief=tk.RAISED, bd=2)
importImageBtn = tk.Button(importImageFrame,text="Import Image", command=open_file_img)
importImageFrame.pack(pady=20)
importImageBtn.pack()

importVideoFrame=tk.Frame(window,relief=tk.RAISED, bd=2)
importVideoBtn = tk.Button(importVideoFrame,text="Import Video", command=open_file_video)
importVideoBtn.pack()
importVideoFrame.pack(pady=20)






window.title("Object Detection and Tracking")
window.rowconfigure(0, minsize=800, weight=1)
window.columnconfigure(1, minsize=800, weight=1)
window.resizable(False,False)


window.mainloop()