import os
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg

def compute_convolution(I, T, stride=None):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays) 
    and returns a heatmap where each grid represents the output produced by 
    convolution at each location. You can add optional parameters (e.g. stride, 
    window_size, padding) to create additional functionality. 
    '''

    normed = True
    (n_rows,n_cols,n_channels) = np.shape(I)
    (k_rows,k_cols, k_channels) = np.shape(T)
    print(np.shape(T))

    dimRGB = np.shape(I)
    dimRGBlist = list(dimRGB)

    dimLlist = dimRGBlist[:2]
    heatmap = np.zeros(tuple(dimLlist))

    kc = (round(k_rows/2),round(k_cols/2))


    for row in range(n_rows - k_rows):
        #print('new row')
        for col in range(n_cols - k_cols):
            heat = 0
            norm1 = 0
            norm2 = 0
            for krow in range(k_rows):
                for kcol in range(k_cols):
                    if normed:
                        heat = heat + int(I[row + krow, col + kcol, 0]) * int(T[krow, kcol, 0])
                        + int(I[row + krow, col + kcol, 1]) * int(T[krow, kcol, 1])
                        + int(I[row + krow, col + kcol, 2]) * int(T[krow, kcol, 2])

                        norm1 = norm1 + (int(I[row + krow, col + kcol, 0])**2) + (int(I[row + krow, col + kcol, 1])**2)
                        + (int(I[row + krow, col + kcol, 2])**2)

                        norm2 = norm2 + (int(T[krow, kcol, 0])**2) + (int(T[krow, kcol, 1])**2) + (int(T[krow, kcol, 2])**2)

                    else:

                        #IMPORTANT! cast the array entries to a larger datatype before multiplication to avoid overflow
                        heat = heat + int(I[row + krow,col + kcol,0])*int(T[krow,kcol,0])
                        + int(I[row + krow,col + kcol,1])*int(T[krow,kcol,1])
                        + int(I[row + krow, col + kcol, 2]) * int(T[krow, kcol, 2])
                        #print(heat)
            if normed:
                norm = np.sqrt(norm1*norm2)
                if norm > 0:
                    heatmap[row + kc[0],col + kc[1]] = heat/norm
                else:
                    heatmap[row + kc[0], col + kc[1]] = 0
            else:
                #print(heat)
                heatmap[row + kc[0],col + kc[1]] = heat

    #print("max of meatmap", np.max(heatmap))

    return heatmap/np.max(heatmap)
    #return heatmap/np.sum(heatmap)
    #return heatmap

global idx
global fh

def predict_boxes(heatmaps, kernals):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''


    out = []
    fullres = True

    for heatmap, kernal in zip(heatmaps, kernals):

        fh = heatmap.flatten()
        #print(fh[17000:17100])

        idx = np.argsort(fh)
        idx = np.flip(idx)
        #print(idx[0:100])

        sh = np.shape(heatmap)
        rows = sh[0]
        #print(rows)
        cols = sh[1]
        #print(cols)
        locations = []

        for i in range(100):
            flat_index = idx[i]
            #print("flat index is:", flat_index, "and i is ", i)
            column = flat_index % cols
            #print("column is:", column)
            row = (flat_index - column)//cols
            #print("flat_index - column: ", flat_index - column)
            #print("rows times row: ", cols*row)
            assert (flat_index - column) == cols*row
            intensity = fh[flat_index]
            rank = i
            column = column - 1
            row = row - 1
            locations.append([row,column,intensity,rank])

        '''
        valid is a list of hot pixels that are specially seperated from one another. 
        This assumes any two stoplights are more than the seperation number of pixels apart
        '''
        seperation = 10
        valid = []
        valid.append(locations[0])
        for pixel in locations:
            if pixel == valid[0]:
                continue
            row = pixel[0]
            col = pixel[1]
            u_rank = 0
            for npixel in valid:
                nrow = npixel[0]
                ncol = npixel[1]
                distance = (abs(row-nrow))**2 + (abs(col - ncol))**2
                if distance > seperation:
                    u_rank = u_rank + 1
                else:
                    break
                if u_rank >= len(valid):
                    valid.append(pixel)



        kernal_dims = np.shape(kernal)
        krows = kernal_dims[0]
        kcols = kernal_dims[1]

        krow_offsett = krows//2
        kcol_offsett = kcols//2

        for valid_pixel in valid:
            tl_row = valid_pixel[0] - krow_offsett
            tl_col = valid_pixel[1] - kcol_offsett
            br_row = tl_row + krows
            br_col = tl_col + kcols
            if fullres:
                out.append([tl_row*shrink_factor,tl_col*shrink_factor,br_row*shrink_factor,br_col*shrink_factor, valid_pixel[2]])
            else:
                out.append([tl_row, tl_col, br_row, br_col, valid_pixel[2]])

    output = sorted(out, key = lambda x: -x[4])
    print("length of output:", len(output))

    #json doesn't play well with numpy types
    output = [[int(entry[0]) + 0, int(entry[1]) + 0,int(entry[2]) - 3,int(entry[3]) + 6, float(entry[4])] for entry in output]


    max_conf = output[0][4]
    sum_conf = 0
    for item in output:
        sum_conf = sum_conf + item[4]

    #for item in output:
    #    item[4] = item[4]/max_conf

    for item in output:
        item[4] = item[4]/sum_conf

    return output


#output.append([tl_row,tl_col,br_row,br_col, score])


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


    Kernals = [0]*8
    npKernals = [0]*8
    croppedKernals = [0]*8
    Small_Kernals = [0]*8
    for i in range(8):
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



    I_cropped_row = (np.shape(I)[0] // shrink_factor) * shrink_factor
    I_cropped_col = (np.shape(I)[1] // shrink_factor) * shrink_factor
    Icropped = I[:I_cropped_row, :I_cropped_col]


    Ismall = Icropped.reshape((np.shape(Icropped)[0] // shrink_factor, bin_size,
                                               np.shape(Icropped)[1] // shrink_factor, bin_size, 3)).max(3).max(1)




    heatmap1 = compute_convolution(Ismall, Small_Kernals[1])
    heatmap2 = compute_convolution(Ismall, Small_Kernals[2])
    heatmap3 = compute_convolution(Ismall, Small_Kernals[3])
    #Heatmap = np.stack((heatmap1, heatmap2, heatmap3), axis=-1)
    #Heatmap.sort() #put most probable points for each heatmap at index 0 for the innermost array
    #heatmap = Heatmap[:,:,0]    #slice that first index

    heatmaps = [heatmap1, heatmap2, heatmap3]
    kernals = Small_Kernals[1:4]


    output = predict_boxes(heatmaps,kernals)



    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output


shrink_factor = 4
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

sub_path = '../submission'
os.makedirs(preds_path, exist_ok=True) # create directory if needed

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Make predictions on the training set.
'''
preds_train = {}


for i in range(len(file_names_train)):
    print(i)
    # read image using PIL:
    print(file_names_train[i])
    I = Image.open(os.path.join(data_path,file_names_train[i]))

    # convert to numpy array:
    I = np.asarray(I)


    preds_train[file_names_train[i]] = detect_red_light_mf(I)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds_train.json'),'w') as f:
    json.dump(preds_train,f)



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
