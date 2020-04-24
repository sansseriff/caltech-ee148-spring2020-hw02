import os
import json
import numpy as np

def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''
    #iou = np.random.random()
    intersect = 0

    for box1row in range(box_1[0],box_1[2] + 1):
        for box1col in range(box_1[1]. box_1[3] + 1):
            if box_2[2] >= box1row >= box_2[0] and box_2[3] >= box1col >= box_2[1]:
                intersect = intersect + 1

    box1height = box_1[2] - box_1[0]
    box1length = box_1[3] - box_1[1]
    box2height = box_2[2] - box_2[0]
    box2length = box_2[3] - box_2[1]

    union = box1height*box1length + box2height*box2length - intersect
    iou = intersect/inion

    assert (iou >= 0) and (iou <= 1.0)

    return iou


def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5):
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

    '''
    BEGIN YOUR CODE
    '''


    # loop over the images
    for pred_file, pred in preds.iteritems():
        gt = gts[pred_file]
        for i in range(len(gt)):  #loops over bouding boxes in image
            for j in range(len(pred)):
                iou = compute_iou(pred[j][:4], gt[i])


    '''
    END YOUR CODE
    '''

    return TP, FP, FN

# set a path for predictions and annotations:
preds_path = '../data/hw02_preds'
gts_path = '../data/hw02_annotations'

# load splits:
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = False

'''
Load training data. 
'''


'''when I have predictions'''
#with open(os.path.join(preds_path,'preds_train.json'),'r') as f:
#    preds_train = json.load(f)
    
with open(os.path.join(gts_path, 'formatted_annotations_students.json'),'r') as f:
    gts_train = json.load(f)

if done_tweaking:
    
    '''
    Load test data.
    '''
    
    with open(os.path.join(preds_path,'preds_test.json'),'r') as f:
        preds_test = json.load(f)
        
    with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
        gts_test = json.load(f)


# For a fixed IoU threshold, vary the confidence thresholds.
# The code below gives an example on the training set for one IoU threshold. 


# code here for visualizing train

finish = False

for idx, item in gts_train:
    print(idx)
    print(item)


if finish:
    confidence_thrs = np.sort(np.array([preds_train[fname][4] for fname in preds_train],dtype=float)) # using (ascending) list of confidence scores as thresholds
    tp_train = np.zeros(len(confidence_thrs))
    fp_train = np.zeros(len(confidence_thrs))
    fn_train = np.zeros(len(confidence_thrs))
    for i, conf_thr in enumerate(confidence_thrs):
        tp_train[i], fp_train[i], fn_train[i] = compute_counts(preds_train, gts_train, iou_thr=0.5, conf_thr=conf_thr)

    # Plot training set PR curves

    if done_tweaking:
        print('Code for plotting test set PR curves.')


