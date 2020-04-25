import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from PIL import Image

def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''
    #iou = np.random.random()
    intersect = 0

    for box1row in range(box_1[0],box_1[2] + 1):
        for box1col in range(box_1[1], box_1[3] + 1):
            if box_2[2] >= box1row >= box_2[0] and box_2[3] >= box1col >= box_2[1]:
                intersect = intersect + 1

    box1height = box_1[2] - box_1[0]
    box1length = box_1[3] - box_1[1]
    box2height = box_2[2] - box_2[0]
    box2length = box_2[3] - box_2[1]

    union = box1height*box1length + box2height*box2length - intersect
    iou = intersect/union

    assert (iou >= 0) and (iou <= 1.0)

    #print(iou)
    return iou


def compute_counts(preds, gts, iou_thr, conf_thr):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.)
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives.
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0
    FP = 0
    FN = 0


    # loop over the images
    for pred_file, pred in preds.items():
        valid_boxes = []
        gt = gts[pred_file]

        # pred is a list of boxes for the current image

        for box in pred:
            if box[4] > conf_thr:
                valid_boxes.append(box)

        TP_image = 0
        FP_image = len(valid_boxes)
        FN_image = len(gt)
        #print(len(valid_boxes))

        if len(valid_boxes) > 0:
            for i in range(len(gt)):  #loops over bouding boxes in image
                ious = []
                for j in range(len(valid_boxes)):
                    ious.append(compute_iou(valid_boxes[j][:4], gt[i]))

                if max(ious) > iou_thr:
                    # match found for the current ground truth stoplight
                    # there bay be more than one prediction near the ground truth box.
                    # But the ground truth box can only correspond to one prediction.
                    # so any other close predictions remain false positives
                    TP_image = TP_image + 1
                    FP_image = FP_image - 1
                    FN_image = FN_image - 1

        #what if gt is empty

        TP = TP + TP_image
        FP = FP + FP_image
        FN = FN + FN_image

    return TP, FP, FN

def view(image_name):
    predict_boxes = preds_train[image_name]
    gts_boxes = gts_train[image_name]


    data_path = '../data/RedLights2011_Medium'
    I = Image.open(os.path.join(data_path, image_name))

    # convert to numpy array:
    #I = np.asarray(I)


    fig, ax = plt.subplots(1)
    ax.imshow(I)


    for i, box in enumerate(predict_boxes):
        if i > 2:
            break
        tl_row = box[0]
        tl_col = box[1]
        br_row = box[2]
        br_col = box[3]
        size_y = br_row - tl_row
        size_x = br_col - tl_col
        rect = patches.Rectangle((tl_col, tl_row), size_x, size_y, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    for box in gts_boxes:
        tl_row = box[0]
        tl_col = box[1]
        br_row = box[2]
        br_col = box[3]
        size_y = br_row - tl_row
        size_x = br_col - tl_col
        rect = patches.Rectangle((tl_col, tl_row), size_x, size_y, linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(rect)

    plt.show()
    fig2, ax2 = plt.subplots(1)
    ax2.plot([x for x in range(20)], [predict_boxes[i][4] for i in range(20)])
    plt.show()

############################################

# set a path for predictions and annotations:
preds_path = '../data/hw02_preds'
gts_path = '../data/hw02_annotations'

# load splits:
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Load training data. 
'''



with open(os.path.join(preds_path,'preds_train.json'),'r') as f:
    preds_train = json.load(f)
    
with open(os.path.join(gts_path, 'formatted_annotations_students.json'),'r') as f:
    gts_train = json.load(f)

if done_tweaking:
    
    '''
    Load test data.
    '''
    
    with open(os.path.join(preds_path,'preds_test.json'),'r') as f:
        preds_test = json.load(f)
        
    with open(os.path.join(gts_path, 'formatted_annotations_students.json'),'r') as f:
        gts_test = json.load(f)


# For a fixed IoU threshold, vary the confidence thresholds.
# The code below gives an example on the training set for one IoU threshold. 


# code here for visualizing train


#for i in range(3):
#    print(preds_train[file_names_train[i]])
#    print("#####################")
 #   print(gts_train[file_names_train[i]])

#view(file_names_train[0])
#view(file_names_train[1])
#view(file_names_train[2])
#view(file_names_train[3])


'''
    for i in range(20):
        tl_row = output[i][0]
        tl_col = output[i][1]
        br_row = output[i][2]
        br_col = output[i][3]
        size_y = br_row - tl_row
        size_x = br_col - tl_col
        rect = patches.Rectangle((tl_col, tl_row), size_x, size_y, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()

    #print(output[:8])

    fig2, ax2 = plt.subplots(1)
    ax2.plot([x for x in range(20)], [output[i][4] for i in range(20)])
    plt.show()
'''







'''
confidence_thrs = np.sort(np.array([preds_train[fname][4] for fname in preds_train],dtype=float)) # using (ascending) list of confidence scores as thresholds
# sort it by trhe
tp_train = np.zeros(len(confidence_thrs))
fp_train = np.zeros(len(confidence_thrs))
fn_train = np.zeros(len(confidence_thrs))
    
    
for i, conf_thr in enumerate(confidence_thrs):
    tp_train[i], fp_train[i], fn_train[i] = compute_counts(preds_train, gts_train, iou_thr=0.5, conf_thr=conf_thr)

# Plot training set PR curves

if done_tweaking:
    print('Code for plotting test set PR curves.')
'''




confs = []
for name in preds_train:
    for box in preds_train[name]:
        confs.append(box[4])

confs.sort()
min_confidence = confs[0]
max_confidence = confs[-1]


confidences = np.linspace(min_confidence,max_confidence, 400)

tp_train_1 = np.zeros(len(confidences))
tp_train_2 = np.zeros(len(confidences))
tp_train_3 = np.zeros(len(confidences))
fp_train_1 = np.zeros(len(confidences))
fp_train_2 = np.zeros(len(confidences))
fp_train_3 = np.zeros(len(confidences))
fn_train_1 = np.zeros(len(confidences))
fn_train_2 = np.zeros(len(confidences))
fn_train_3 = np.zeros(len(confidences))

P_train_1 = np.zeros(len(confidences))
R_train_1 = np.zeros(len(confidences))
P_train_2 = np.zeros(len(confidences))
R_train_2 = np.zeros(len(confidences))
P_train_3 = np.zeros(len(confidences))
R_train_3 = np.zeros(len(confidences))

for i, conf_thr in enumerate(confidences):
    if i > 40:
        break
    print(i)
    tp_train_1[i], fp_train_1[i], fn_train_1[i] = compute_counts(preds_train, gts_train, iou_thr=0.5, conf_thr=conf_thr)
    tp_train_2[i], fp_train_2[i], fn_train_2[i] = compute_counts(preds_train, gts_train, iou_thr=0.25, conf_thr=conf_thr)
    tp_train_3[i], fp_train_3[i], fn_train_3[i] = compute_counts(preds_train, gts_train, iou_thr=0.75, conf_thr=conf_thr)


    P_train_1[i] = tp_train_1[i] / (tp_train_1[i] + fp_train_1[i])
    R_train_1[i] = tp_train_1[i] / (tp_train_1[i] + fn_train_1[i])

    P_train_2[i] = tp_train_2[i] / (tp_train_2[i] + fp_train_2[i])
    R_train_2[i] = tp_train_2[i] / (tp_train_2[i] + fn_train_2[i])

    P_train_3[i] = tp_train_3[i] / (tp_train_3[i] + fp_train_3[i])
    R_train_3[i] = tp_train_3[i] / (tp_train_3[i] + fn_train_3[i])

    print(P_train_2[i])

fig2, ax2 = plt.subplots(1)
ax2.plot(P_train_1,R_train_1, 'ro', label = 'iou_thr = 0.5', markersize=3)
ax2.plot(P_train_2, R_train_2, 'go', label = 'iou_thr = 0.25', markersize=3)
ax2.plot(P_train_3, R_train_3, 'bo', label = 'iou_thr = 0.75', markersize=3)
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.legend()
plt.title('PR Curves Train Algorithm ver. 2')
plt.show()






if done_tweaking:
    confs = []
    for name in preds_test:
        for box in preds_test[name]:
            confs.append(box[4])

    confs.sort()
    min_confidence = confs[0]
    max_confidence = confs[-1]

    confidences = np.linspace(min_confidence, max_confidence, 100)

    tp_test_1 = np.zeros(len(confidences))
    tp_test_2 = np.zeros(len(confidences))
    tp_test_3 = np.zeros(len(confidences))
    fp_test_1 = np.zeros(len(confidences))
    fp_test_2 = np.zeros(len(confidences))
    fp_test_3 = np.zeros(len(confidences))
    fn_test_1 = np.zeros(len(confidences))
    fn_test_2 = np.zeros(len(confidences))
    fn_test_3 = np.zeros(len(confidences))

    P_test_1 = np.zeros(len(confidences))
    R_test_1 = np.zeros(len(confidences))
    P_test_2 = np.zeros(len(confidences))
    R_test_2 = np.zeros(len(confidences))
    P_test_3 = np.zeros(len(confidences))
    R_test_3 = np.zeros(len(confidences))



    for i, conf_thr in enumerate(confidences):
        if i > 40:
            break
        print(i)
        tp_test_1[i], fp_test_1[i], fn_test_1[i] = compute_counts(preds_test, gts_test, iou_thr=0.5,
                                                                     conf_thr=conf_thr)
        tp_test_2[i], fp_test_2[i], fn_test_2[i] = compute_counts(preds_test, gts_test, iou_thr=0.25,
                                                                     conf_thr=conf_thr)
        tp_test_3[i], fp_test_3[i], fn_test_3[i] = compute_counts(preds_test, gts_test, iou_thr=0.75,
                                                                     conf_thr=conf_thr)

        P_test_1[i] = tp_test_1[i] / (tp_test_1[i] + fp_test_1[i])
        R_test_1[i] = tp_test_1[i] / (tp_test_1[i] + fn_test_1[i])

        P_test_2[i] = tp_test_2[i] / (tp_test_2[i] + fp_test_2[i])
        R_test_2[i] = tp_test_2[i] / (tp_test_2[i] + fn_test_2[i])

        P_test_3[i] = tp_test_3[i] / (tp_test_3[i] + fp_test_3[i])
        R_test_3[i] = tp_test_3[i] / (tp_test_3[i] + fn_test_3[i])

        print(P_test_2[i])

    fig3, ax3 = plt.subplots(1)
    ax3.plot(P_test_1, R_test_1, 'ro', label='iou_thr = 0.5', markersize=3)
    ax3.plot(P_test_2, R_test_2, 'go', label='iou_thr = 0.25', markersize=3)
    ax3.plot(P_test_3, R_test_3, 'bo', label='iou_thr = 0.75', markersize=3)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.legend()
    plt.title('PR Curves Test Algorithm ver. 2')
    plt.show()
