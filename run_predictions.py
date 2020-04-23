import os
import numpy as np
import json
from PIL import Image

def compute_convolution(I, T, stride=None):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays) 
    and returns a heatmap where each grid represents the output produced by 
    convolution at each location. You can add optional parameters (e.g. stride, 
    window_size, padding) to create additional functionality. 
    '''

    normed = False
    (n_rows,n_cols,n_channels) = np.shape(I)
    (k_rows,k_cols, k_channels) = np.shape(T)

    heatmap = np.zeros(np.shape(I))

    kc = (round(k_rows/2),round(k_cols/2))
    heat = 0
    norm = 0
    for row in range(n_rows - k_rows):
        print('new row')
        for col in range(n_cols - k_cols):
            for krow in range(k_rows):
                for kcol in range(k_cols):
                    if normed:
                        heat = heat + np.inner(I[row + krow][col + kcol],T[krow][kcol])
                        norm = norm + np.inner(np.square(I[row + krow][col + kcol]),np.square(T[krow][kcol]))
                    else:
                        heat = heat + np.inner(I[row + krow][col + kcol], T[krow][kcol])
                        #print(heat)
            if normed:
                heatmap[row + kc[0]][col + kc[1]] = heat/norm
            else:
                heatmap[row + kc[0]][col + kc[1]] = heat
            heat = 0
            norm = 0

    #print(heatmap)
    return heatmap/np.max(heatmap)


def predict_boxes(heatmap):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''

    output = []

    '''
    BEGIN YOUR CODE
    '''
    
    '''
    As an example, here's code that generates between 1 and 5 random boxes
    of fixed size and returns the results in the proper format.
    '''

    box_height = 8
    box_width = 6

    num_boxes = np.random.randint(1,5)

    for i in range(num_boxes):
        (n_rows,n_cols,n_channels) = np.shape(I)

        tl_row = np.random.randint(n_rows - box_height)
        tl_col = np.random.randint(n_cols - box_width)
        br_row = tl_row + box_height
        br_col = tl_col + box_width

        score = np.random.random()

        output.append([tl_row,tl_col,br_row,br_col, score])

    '''
    END YOUR CODE
    '''

    return output


def detect_red_light_mf(I):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>. 
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>. 
    The first four entries are four integers specifying a bounding box 
    (the row and column index of the top left corner and the row and column 
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1. 

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''

    '''
    BEGIN YOUR CODE
    '''

    shrink_factor = 4
    #template_height = 8
    #template_width = 6


    Kernals = [0]*7
    npKernals = [0]*7
    croppedKernals = [0]*7
    Small_Kernals = [0]*7
    for i in range(7):
        Kernals[i] = (Image.open(os.path.join(kernal_path,'K' + str(i+1) + '.jpg')))
        npKernals[i] = np.asarray(Kernals[i])

        # need to crop the kernals so that I can divide them by 4
        cropped_row = (np.shape(npKernals[i])[0]//shrink_factor)*shrink_factor
        cropped_col = (np.shape(npKernals[i])[1]//shrink_factor)*shrink_factor
        croppedKernals[i] = npKernals[i][:cropped_row,:cropped_col]

        #print(np.shape(croppedKernals[i]))

        bin_size = shrink_factor
        Small_Kernals[i] = croppedKernals[i].reshape((np.shape(croppedKernals[i])[0] // shrink_factor, bin_size,
                                               np.shape(croppedKernals[i])[1] // shrink_factor, bin_size, 3)).max(3).max(1)

        print(np.shape(Small_Kernals[i]))
        #print(npKernals[i].shape)


    I_cropped_row = (np.shape(I)[0] // shrink_factor) * shrink_factor
    I_cropped_col = (np.shape(I)[1] // shrink_factor) * shrink_factor
    Icropped = I[:I_cropped_row, :I_cropped_col]


    Ismall = Icropped.reshape((np.shape(Icropped)[0] // shrink_factor, bin_size,
                                               np.shape(Icropped)[1] // shrink_factor, bin_size, 3)).max(3).max(1)

    #print(np.shape(Ismall))
    #img = Image.fromarray(Ismall, 'RGB')
    #img.show()


    # You may use multiple stages and combine the results
    #T = np.random.random((template_height, template_width))

    heatmap = compute_convolution(Ismall, Small_Kernals[0])
    print(heatmap)

    #print(np.rint(heatmap * 256))
    #img = Image.fromarray(np.rint(heatmap*256),'L')
    #img.show()

    #output = predict_boxes(heatmap)

    '''
    END YOUR CODE
    '''
    '''
    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output
    '''

#global npKernals
# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = '../data/RedLights2011_Medium'
kernal_path = '../data/hw02_kernals'

# load splits: 
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# set a path for saving predictions:
preds_path = '../data/hw02_preds'
os.makedirs(preds_path, exist_ok=True) # create directory if needed

# Set this parameter to True when you're done with algorithm development:
done_tweaking = False

'''
Make predictions on the training set.
'''
preds_train = {}


'''limit images used here ######################################################################
len(file_names_train)

'''
for i in range(1):

    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names_train[i]))

    # convert to numpy array:
    I = np.asarray(I)

    preds_train[file_names_train[i]] = detect_red_light_mf(I)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds_train.json'),'w') as f:
    json.dump(preds_train,f)

################################################################################################

if done_tweaking:
    '''
    Make predictions on the test set. 
    '''
    preds_test = {}
    for i in range(len(file_names_test)):

        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_test[i]))

        # convert to numpy array:
        I = np.asarray(I)

        preds_test[file_names_test[i]] = detect_red_light_mf(I)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_test.json'),'w') as f:
        json.dump(preds_test,f)
