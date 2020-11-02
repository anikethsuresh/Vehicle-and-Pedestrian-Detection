import cvlib
import cv2
import numpy as np
SUPPORTED_YOLO_OBJECTS =("car","person","bicycle","motorcycle","truck")

class Old_Objects():
    def __init__(self):
        self.count_objs = {"car":0,
                     "person":0,
                     "bicycle":0,
                     "motorcycle":0,
                     "truck":0}
        self.old_objs = {"car":[],
                     "person":[],
                     "bicycle":[],
                     "motorcycle":[],
                     "truck":[]}
        self.new_objs = {"car":[],
                     "person":[],
                     "bicycle":[],
                     "motorcycle":[],
                     "truck":[]}
        self.distance_threshold = 100
        self.confs_threshold = 0.75

    def getObjs(self):
        bboxs, labels, confs  = [],[],[]
        for values in self.new_objs.values():
            if len(values) == 0:
                continue
            else:
                for ind_obj in values:
                    bboxs.append(ind_obj.bbox)
                    labels.append(ind_obj.label)
                    confs.append(ind_obj.conf)
        return bboxs, labels, confs
    
    def refresh(self, grouped_new_objs):
        for index in range(len(grouped_new_objs.labels)):
            currentBbox = grouped_new_objs.bboxs[index]
            currentLabel = grouped_new_objs.labels[index]
            currentConf = grouped_new_objs.confs[index]
            if currentLabel in SUPPORTED_YOLO_OBJECTS and currentConf > self.confs_threshold:
                bool_old_obj = False
                arr_ind_objs = self.old_objs[currentLabel]
                i = 0
                while bool_old_obj == False and i != len(arr_ind_objs) and len(self.old_objs[currentLabel]) != 0:
                    ind_obj = arr_ind_objs[i].bbox
                    ind_obj = np.array(ind_obj)
                    comparable = np.array(currentBbox)
                    output = np.sqrt(np.sum(pow(ind_obj - comparable,2)))
                    bool_old_obj = True if output< self.distance_threshold else False
                    i += 1
                if bool_old_obj:
                    # update bbox
                    new_individual_obj = Individual_Obj(currentBbox, self.old_objs[currentLabel][i-1].label,currentConf)
                    self.new_objs[currentLabel].append(new_individual_obj)
                else:
                    # New yolo object to be added
                    label_count = self.count_objs[currentLabel]
                    self.count_objs[currentLabel] +=1
                    new_individual_obj = Individual_Obj(currentBbox,currentLabel + str(label_count),currentConf)
                    self.new_objs[currentLabel].append(new_individual_obj)
    
    def reset(self):
        self.old_objs = self.new_objs
        self.new_objs = {"car":[],
                     "person":[],
                     "bicycle":[],
                     "motorcycle":[],
                     "truck":[]}

class Grouped_new_objs():
    def __init__(self, bboxs, labels, confs):
        self.bboxs = bboxs
        self.labels = labels
        self.confs = confs

class Individual_Obj():
    def __init__(self, bbox, label, conf):
        self.bbox = bbox
        self.label = label
        self.conf = conf

    def __str__(self):
        return self.label

def drawBoundingBoxes(frame, bbox, label, conf):
    for i in range(len(label)):
        frame = cv2.rectangle(frame, (bbox[i][0],bbox[i][1]),(bbox[i][2],bbox[i][3]),(0,0,255))
        frame = cv2.putText(frame, label[i],(bbox[i][0],bbox[i][1] - 10),cv2.FONT_HERSHEY_COMPLEX,0.5,(225,0,0), thickness=2)
    return frame


i = 0
old_objs = Old_Objects()
cap = cv2.VideoCapture('Sherbrooke/sherbrooke_video.avi')
result = cv2.VideoWriter('output.avi',  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         30, (800,600))
while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    bbox, label, conf = cvlib.detect_common_objects(frame)
    grouped_new_objs = Grouped_new_objs(bbox, label, conf)
    old_objs.refresh(grouped_new_objs)
    bbox, label, conf = old_objs.getObjs()
    old_objs.reset()
    output_image = drawBoundingBoxes(frame, bbox, label, conf)
    result.write(output_image) 
    print(i)
    # output_image = cvlib.object_detection.draw_bbox(frame, bbox, label, conf)
    # cv2.imshow('frame', output_image)
    if cv2.waitKey(1) == ord('q'):
        break
    i += 1
cap.release()
result.release()
cv2.destroyAllWindows()