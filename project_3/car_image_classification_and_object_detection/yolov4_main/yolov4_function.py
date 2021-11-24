import os
os.getcwd()
import cv2
import numpy as np
import random
import colorsys
import pytesseract
import re

CLASS_NAME = "./data/classes/coco.names"


def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

def recognize_plate(img, coords):
    # separate coordinates from box
    xmin, ymin, xmax, ymax = coords
    # get the subimage that makes up the bounded region and take an additional 5 pixels on each side
    box = img[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]
    # grayscale region within bounding box
    gray = cv2.cvtColor(box, cv2.COLOR_RGB2GRAY)
    # resize image to three times as large as original for better readability
    gray = cv2.resize(gray, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
    # perform gaussian blur to smoothen image
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    #cv2.imshow("Gray", gray)
    #cv2.waitKey(0)
    # threshold the image using Otsus method to preprocess for tesseract
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    #cv2.imshow("Otsu Threshold", thresh)
    #cv2.waitKey(0)
    # create rectangular kernel for dilation
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    # apply dilation to make regions more clear
    dilation = cv2.dilate(thresh, rect_kern, iterations = 1)
    #cv2.imshow("Dilation", dilation)
    #cv2.waitKey(0)
    # find contours of regions of interest within license plate
    try:
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        ret_img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # sort contours left-to-right
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    # create copy of gray image
    im2 = gray.copy()
    # create blank string to hold license plate number
    plate_num = ""
    # loop through contours and find individual letters and numbers in license plate
    for cnt in sorted_contours:
        x,y,w,h = cv2.boundingRect(cnt)
        height, width = im2.shape
        # if height of box is not tall enough relative to total height then skip
        if height / float(h) > 6: continue

        ratio = h / float(w)
        # if height to width ratio is less than 1.5 skip
        if ratio < 1.5: continue

        # if width is not wide enough relative to total width then skip
        if width / float(w) > 15: continue

        area = h * w
        # if area is less than 100 pixels skip
        if area < 100: continue

        # draw the rectangle
        rect = cv2.rectangle(im2, (x,y), (x+w, y+h), (0,255,0),2)
        # grab character region of image
        roi = thresh[y-5:y+h+5, x-5:x+w+5]
        # perfrom bitwise not to flip image to black text on white background
        roi = cv2.bitwise_not(roi)
        # perform another blur on character region
        roi = cv2.medianBlur(roi, 5)
        try:
            text = pytesseract.image_to_string(roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
            # clean tesseract text by removing any unwanted blank spaces
            clean_text = re.sub('[\W_]+', '', text)
            plate_num += clean_text
        except: 
            text = None
    if plate_num != None:
        print("License Plate #: ", plate_num)
    #cv2.imshow("Character's Segmented", im2)
    #cv2.waitKey(0)
    return plate_num

def draw_bbox(image, bboxes, info = False, counted_classes = None, show_label=True, allowed_classes=list(read_class_names(CLASS_NAME).values()), read_plate = False):
    classes = read_class_names(CLASS_NAME)
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    out_boxes, out_scores, out_classes, num_boxes = bboxes
    for i in range(num_boxes[0]):
        if int(out_classes[0][i]) < 0 or int(out_classes[0][i]) > num_classes: continue
        coor = out_boxes[0][i]
        coor[0] = int(coor[0] * image_h)
        coor[2] = int(coor[2] * image_h)
        coor[1] = int(coor[1] * image_w)
        coor[3] = int(coor[3] * image_w)
        
        fontScale = 0.5
        score = out_scores[0][i]
        class_ind = int(out_classes[0][i])
        class_name = classes[class_ind]
        if class_name not in allowed_classes:
            continue
        else:
            if read_plate:
                height_ratio = int(image_h / 25)
                plate_number = recognize_plate(image, coor)
                if plate_number != None:
                    cv2.putText(image, plate_number, (int(coor[0]), int(coor[1]-height_ratio)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255,255,0), 2)

        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (int(coor[1]), int(coor[0])), (int(coor[3]), int(coor[2]))
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if info:
            print("Object found: {}, Confidence: {:.2f}, BBox Coords (xmin, ymin, xmax, ymax): {}, {}, {}, {} ".format(class_name, score, coor[0], coor[1], coor[2], coor[3]))

        if show_label:
            bbox_mess = '%s: %.2f' % (class_name, score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
            c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
            cv2.rectangle(image, c1, (int(np.float32(c3[0])), int(np.float32(c3[1]))), bbox_color, -1) #filled

            cv2.putText(image, bbox_mess, (c1[0], int(np.float32(c1[1] - 2))), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)

        if counted_classes != None:
            height_ratio = int(image_h / 25)
            offset = 15
            for key, value in counted_classes.items():
                cv2.putText(image, "{}s detected: {}".format(key, value), (5, offset),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
                offset += height_ratio
    return image

# function to count objects, can return total classes or count per class
def count_objects(data, by_class = False, allowed_classes = list(read_class_names(CLASS_NAME).values())):
    boxes, scores, classes, num_objects = data

    #create dictionary to hold count of objects
    counts = dict()

    # if by_class = True then count objects per class
    if by_class:
        class_names = read_class_names(CLASS_NAME)

        # loop through total number of objects found
        for i in range(num_objects[0]):
            # grab class index and convert into corresponding class name
            class_index = int(classes[0][i])
            class_name = class_names[class_index]
            if class_name in allowed_classes:
                counts[class_name] = counts.get(class_name, 0) + 1
            else:
                continue

    # else count total objects found
    else:
        counts['total object'] = num_objects
    
    return counts