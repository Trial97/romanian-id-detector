import os
import cv2
import numpy as np
import tensorflow as tf
import sys
# Import utilites
import imutils
from imutils import paths
from utils import label_map_util
from utils import visualization_utils as vis_util
from scipy.spatial import distance as dist

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

def plotThis(img):
    # global i
    # cv2.imshow(str(i),img)
    # i+=1
    # return      
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()    

def simplify_contour(contour, n_corners=4):
    '''
    Binary searches best `epsilon` value to force contour 
        approximation contain exactly `n_corners` points.

    :param contour: OpenCV2 contour.
    :param n_corners: Number of corners (points) the contour must contain.

    :returns: Simplified contour in successful case. Otherwise returns initial contour.
    '''
    n_iter, max_iter = 0, 100
    lb, ub = 0., 1.

    while True:
        n_iter += 1
        if n_iter > max_iter:
            return contour

        k = (lb + ub)/2.
        eps = k*cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, eps, True)

        if len(approx) > n_corners:
            lb = (lb + ub)/2.
        elif len(approx) < n_corners:
            ub = (lb + ub)/2.
        else:
            return approx

 
def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]
 
    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
 
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
 
    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
 
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")
def four_point_transform(image, pts):
    pts=np.array(pts, dtype = "float32")
    rect=pts.reshape((4,2))
    rect=order_points(rect)
    (tl, tr, br, bl) = rect
 
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
 
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
 
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
 
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
    # return the warped image
    return warped

class IDector:
    def __init__(self):
        # Name of the directory containing the object detection module we're using
        self.model_name = 'model'
        # Grab path to current working directory
        self.cwd_path = os.getcwd()
        # Path to frozen detection graph .pb file, which contains the model that is used
        # for object detection.
        self.path_to_ckpt = os.path.join(self.cwd_path, self.model_name, 'frozen_inference_graph.pb')
        # Path to label map file
        self.path_to_labels = os.path.join(self.cwd_path,'data','labelmap.pbtxt')
        # Number of classes the object detector can identify
        self.num_classes = 1
        self.load_label_map()
        self.load_tf_model()
        self.load_cascade()

    def load_cascade(self):
        # Load the cascades
        self.face_cascade_name = 'data/haarcascades/haarcascade_frontalface_alt.xml'
        self.eyes_cascade_name = 'data/haarcascades/haarcascade_eye_tree_eyeglasses.xml'
        self.face_cascade = cv2.CascadeClassifier()
        self.eyes_cascade = cv2.CascadeClassifier()
        if not self.face_cascade.load(cv2.samples.findFile(self.face_cascade_name)):
            print('--(!)Error loading face cascade')
            exit(0)
        if not self.eyes_cascade.load(cv2.samples.findFile(self.eyes_cascade_name)):
            print('--(!)Error loading eyes cascade')
            exit(0)

    def load_label_map(self):
        # Load the label map.
        # Label maps map indices to category names, so that when our convolution
        # network predicts `5`, we know that this corresponds to `king`.
        # Here we use internal utility functions, but anything that returns a
        # dictionary mapping integers to appropriate string labels would be fine
        self.label_map = label_map_util.load_labelmap(self.path_to_labels)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=self.num_classes, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)

    def load_tf_model(self):
        # Load the Tensorflow model into memory.
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            self.od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                self.od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(self.od_graph_def, name='')
            self.sess = tf.Session(graph=self.detection_graph)
        # Define input and output tensors (i.e. data) for the object detection classifier

        # Input tensor is the image
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Output tensors are the detection boxes, scores, and classes
        # Each box represents a part of the image where a particular object was detected
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represents level of confidence for each of the objects.
        # The score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')

        # Number of objects detected
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def detect_tf(self,img):
        image=img.copy()
        image_expanded = np.expand_dims(image, axis=0)
        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_expanded})
        return vis_util.visualize_boxes_and_labels_on_image_array2(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.60)

    def get_buletin_from_box(self,img,box,addp=0.125):
        ymin, xmin, ymax, xmax = box
        im_height, im_width = img.shape[:2]
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                        ymin * im_height, ymax * im_height)
        w=np.abs(right-left)
        h=np.abs(bottom-top)
        toadd=min(w*addp,h*addp)
        left=max(left-toadd,0)
        top=max(top-toadd,0)
        right=min(right+toadd,im_width)
        bottom=min(bottom+toadd,im_height)
        return img[int(top):int(bottom),int(left):int(right),:]

    def get_bounding_rect(self,img):
        buletin = img.copy()
        buletin_grey= cv2.cvtColor(buletin,cv2.COLOR_BGR2GRAY)
        kernel = np.ones((5,5),np.float32)/25
        gray = cv2.filter2D(buletin_grey,-1,kernel)
        # gray = cv2.blur(gray,(3,3))  
        edges = cv2.Canny(gray,600,700,apertureSize = 5)
        dilatation_size = 4
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (2*dilatation_size + 1, 2*dilatation_size+1), (dilatation_size, dilatation_size))
        edges = cv2.dilate(edges, element)

        contours, hier = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hull =None
        maxCA=0
        for c in contours:
            ca=cv2.contourArea(c)
            if ca>maxCA:
                hull=cv2.convexHull(c, False)
                maxCA=ca
        return simplify_contour(hull)

    def get_masked(self,img):
        lower_gray = np.array([0, 5, 50], np.uint8)
        upper_gray = np.array([179, 50, 255], np.uint8)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
        mask = cv2.inRange(hsv, lower_gray, upper_gray)
        dilatation_size = 2
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (2*dilatation_size + 1, 2*dilatation_size+1), (dilatation_size, dilatation_size))
        mask = cv2.dilate(mask, element)
        img_copy=img.copy()
        img_copy=cv2.bitwise_and(img_copy, img_copy, mask = mask)
        return img_copy

    def get_minimal_area(self,buletin):
        rect1 = self.get_bounding_rect(buletin)
        ca1=cv2.contourArea(rect1)
        buletin_masked=self.get_masked(buletin) 
        rect2 = self.get_bounding_rect(buletin_masked)
        ca2=cv2.contourArea(rect2)
        if ca1 < ca2:
            return rect1
        return rect2

    def detect_face(self, img):
        img = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        #-- Detect faces
        faces = self.face_cascade.detectMultiScale(gray)
        for (x,y,w,h) in faces:
            img = cv2.rectangle(img, (x,y), (x+w,y+h), (255, 0, 255), 4)
            faceROI = gray[y:y+h,x:x+w]
            #-- In each face, detect eyes
            eyes = self.eyes_cascade.detectMultiScale(faceROI)
            for (x2,y2,w2,h2) in eyes:
                eye_center = (x + x2 + w2//2, y + y2 + h2//2)
                radius = int(round((w2 + h2)*0.25))
                img = cv2.circle(img, eye_center, radius, (255, 0, 0 ), 4)
        if len(faces)!=0:
            plotThis(img)
        return len(faces)!=0

    def detect_buletin(self,img):
        if type(img) is str:
            img = cv2.imread(img)
        if img is None:
            return None
        img=imutils.resize(img, width=800)
        tf_result,boxes = self.detect_tf(img)
        boxes = list(boxes)
        if len(boxes)==0:
            return None
        plotThis(tf_result)
        box=boxes[0]
        buletin = self.get_buletin_from_box(img,box)
        rect = self.get_minimal_area(buletin)
        buletin_copy = buletin.copy()
        cv2.drawContours(buletin_copy,[rect],0,(0,191,255),20)
        plotThis(buletin_copy)
        unwraped = four_point_transform(buletin,rect)
        plotThis(unwraped)
        return unwraped
        
    def get_buletin(self,img):
        buletin = self.detect_buletin(img)
        if buletin is None:
            return None
        for angle in [0,90,180,270]:
            img = buletin.copy()
            if angle != 0:
                img = imutils.rotate(img,angle)
            if self.detect_face(img):
                return img
        print("No face")
        return buletin


rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
def detectROI(image):
    image = imutils.resize(image, height=600)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dilated_img = cv2.dilate(image[:,:,1], np.ones((7, 7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    
    #--- finding absolute difference to preserve edges ---
    diff_img = 255 - cv2.absdiff(image[:,:,1], bg_img)
    
    #--- normalizing between 0 to 255 ---
    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    
    gray = cv2.threshold(norm_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # smooth the image using a 3x3 Gaussian, then apply the blackhat
    # morphological operator to find dark regions on a light background
    # gray = cv2.GaussianBlur(gray, (3, 3), 0)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

    # compute the Scharr gradient of the blackhat image and scale the
    # result into the range [0, 255]
    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

    # apply a closing operation using the rectangular kernel to close
    # gaps in between letters -- then apply Otsu's thresholding method
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # perform another closing operation, this time using the square
    # kernel to close gaps between lines of the MRZ, then perform a
    # serieso of erosions to break apart connected components
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
    thresh = cv2.erode(thresh, None, iterations=4)

    # during thresholding, it's possible that border pixels were
    # included in the thresholding, so let's set 5% of the left and
    # right borders to zero
    p = int(image.shape[1] * 0.05)
    thresh[:, 0:p] = 0
    thresh[:, image.shape[1] - p:] = 0

    # find contours in the thresholded image and sort them by their
    # size
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    roi=None
    # loop over the contours
    for c in cnts:
        # compute the bounding box of the contour and use the contour to
        # compute the aspect ratio and coverage ratio of the bounding box
        # width to the width of the image
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        crWidth = w / float(gray.shape[1])

        # check to see if the aspect ratio and coverage width are within
        # acceptable criteria
        if ar > 5 and crWidth > 0.75:
            # pad the bounding box since we applied erosions and now need
            # to re-grow it
            pX = int((x + w) * 0.03)
            pY = int((y + h) * 0.03)
            (x, y) = (x - pX, y - pY)
            (w, h) = (w + (pX * 2), h + (pY * 2))

            # extract the ROI from the image and draw a bounding box
            # surrounding the MRZ
            roi = image[y:y + h, x:x + w].copy()
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            break
    return roi




id = IDector()
for p in paths.list_images("./test_images/"):
    b = id.get_buletin(p)
    plotThis(b)
    roi=detectROI(b)
    if roi is None:
        print("No MZR detected")
        continue
    plotThis(roi)


