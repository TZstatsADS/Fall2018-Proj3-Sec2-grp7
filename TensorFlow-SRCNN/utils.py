import cv2
import numpy as np
import tensorflow as tf
import os 
import glob
import h5py
import random



def load_data1(is_train):
    if is_train:
        data_dir1 = os.path.join(os.getcwd(), 'train_set', "LR0") # Join the Train dir to current directory
        data1 = glob.glob(os.path.join(data_dir1, "*.jpg")) # make set of all dataset file path
        
        data_dir2 = os.path.join(os.getcwd(), 'train_set', "HR0") # Join the Train dir to current directory
        data2 = glob.glob(os.path.join(data_dir2, "*.jpg")) # make set of all dataset file path
    return data1,data2


def make_sub_data1(data1,data2, padding, config):
    """
        Make the sub_data set
        Args:
            data : the set of all file path 
            padding : the image padding of input to label
            config : the all flags
    """
    sub_input_sequence = []
    sub_label_sequence = []
    for i in range(len(data1)):
        if config.is_train:
            input_=data1[i]
            label_=data2[i]
            
            
            input_=cv2.imread(input_)
            input_=cv2.resize(input_,None,fx = 2 ,fy = 2, interpolation = cv2.INTER_CUBIC)
            label_=cv2.imread(label_)
            
        if len(input_.shape) == 3: # is color
            h, w, c = input_.shape
        else:
            h, w = input_.shape # is grayscale
 
        nx, ny = 0, 0
        for x in range(0, h - config.image_size + 1, config.stride):
            nx += 1; ny = 0
            for y in range(0, w - config.image_size + 1, config.stride):
                ny += 1

                sub_input = input_[x: x + config.image_size, y: y + config.image_size] # 33 * 33
                sub_label = label_[x + padding: x + padding + config.label_size, y + padding: y + padding + config.label_size] # 21 * 21


                # Reshape the subinput and sublabel
                sub_input = sub_input.reshape([config.image_size, config.image_size, config.c_dim])
                sub_label = sub_label.reshape([config.label_size, config.label_size, config.c_dim])
                # Normialize
                sub_input =  sub_input / 255.0
                sub_label =  sub_label / 255.0
                
                #cv2.imshow("im1",sub_input)
                #cv2.imshow("im2",sub_label)
                #cv2.waitKey(0)

                # Add to sequence
                sub_input_sequence.append(sub_input)
                sub_label_sequence.append(sub_label)

        
    # NOTE: The nx, ny can be ignore in train
    return sub_input_sequence, sub_label_sequence, nx, ny

def make_data_hf1(input_, label_, config):
    """
        Make input data as h5 file format
        Depending on "is_train" (flag value), savepath would be change.
    """
    # Check the check dir, if not, create one
    if not os.path.isdir(os.path.join(os.getcwd(),config.checkpoint_dir)):
        os.makedirs(os.path.join(os.getcwd(),config.checkpoint_dir))

    if config.is_train:
        savepath = os.path.join(os.getcwd(), config.checkpoint_dir + '/train.h5')

    with h5py.File(savepath, 'w') as hf:
        hf.create_dataset('input', data=input_)
        hf.create_dataset('label', data=label_)

def input_setup1(config):
    """
        Read image files and make their sub-images and saved them as a h5 file format
    """

    # Load data path, if is_train False, get test data
    data1,data2 = load_data1(config.is_train)

    padding = abs(config.image_size - config.label_size) // 2

    # Make sub_input and sub_label, if is_train false more return nx, ny
    sub_input_sequence, sub_label_sequence, nx, ny = make_sub_data1(data1,data2, padding, config)


    # Make list to numpy array. With this transform
    arrinput = np.asarray(sub_input_sequence) # [?, 33, 33, 3]
    arrlabel = np.asarray(sub_label_sequence) # [?, 21, 21, 3]

    make_data_hf1(arrinput, arrlabel, config)
    
    return nx, ny



##############################################################################



def checkpoint_dir1(config):
    if config.is_train:
        return os.path.join('./{}'.format(config.checkpoint_dir), "train.h5")
    #else:
        #return os.path.join('./{}'.format(config.checkpoint_dir), "test.h5")
    
    



#########################################################################################

def read_data1(path):
    """
        Read h5 format data file

        Args:
            path: file path of desired file
            data: '.h5' file format that contains  input values
            label: '.h5' file format that contains label values 
    """
    with h5py.File(path, 'r') as hf:
        input_ = np.array(hf.get('input'))
        label_ = np.array(hf.get('label'))
        return input_, label_

    
######################################
#                                    #
#          FOR TESTING ONLY          #
#                                    #
######################################    
    
    

    # Get the Image
    def imread(path):
        img = cv2.imread(path)
        return img

    def imsave(image, path, config):
        #checkimage(image)
        # Check the check dir, if not, create one
        if not os.path.isdir(os.path.join(os.getcwd(),config.result_dir)):
            os.makedirs(os.path.join(os.getcwd(),config.result_dir))

        # NOTE: because normial, we need mutlify 255 back    
        cv2.imwrite(os.path.join(os.getcwd(),path),image * 255.)

    def checkimage(image):
        cv2.imshow('test', image)
        cv2.waitKey(0)

    def modcrop(img, scale = 3):
        """
            To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
        """
        # Check the image is grayscale
        if len(img.shape) ==3:
            h, w, _ = img.shape
            h = (h // scale) * scale
            w = (w // scale) * scale
            img = img[0:h, 0:w, :]
        else:
            h, w = img.shape
            h = (h // scale) * scale
            w = (w // scale) * scale
            img = img[0:h, 0:w]
        return img

    def checkpoint_dir(config):
        if config.is_train:
            return os.path.join('./{}'.format(config.checkpoint_dir), 'train.h5')
        else:
            return os.path.join('./{}'.format(config.checkpoint_dir), 'test.h5')

    # def preprocess(path, scale = 3):
    #     img = imread(path)
    #     #img=cv2.resize(img,None,fx = 2 ,fy = 2, interpolation = cv2.INTER_CUBIC)

    #     label_ = modcrop(img, scale)

    #     bicbuic_img = cv2.resize(label_, None, fx = 1.0/scale, fy = 1.0/scale, interpolation = cv2.INTER_CUBIC)# Resize by scaling factor
    #     input_ = cv2.resize(bicbuic_img, None, fx = scale, fy = scale, interpolation = cv2.INTER_CUBIC)# Resize by scaling factor
    #     return input_, label_
    def prepare_data(dataset='Train',Input_img=''):
        """
            Args:
                dataset: choose train dataset or test dataset
                For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp',..., 't99.bmp']
        """
        if dataset == 'Train':
            data_dir = os.path.join(os.getcwd(), dataset) # Join the Train dir to current directory
            data = glob.glob(os.path.join(data_dir, '*.*')) # make set of all dataset file path
        else:
            if Input_img !='':
                data = [os.path.join(os.getcwd(),Input_img)]
            else:
                data_dir = os.path.join(os.path.join(os.getcwd(), dataset), 'Set5')
                data = glob.glob(os.path.join(data_dir, '*.*')) # make set of all dataset file path
        print(data)
        return data

    def load_data(is_train, test_img):
        """
            Args:
                is_train: decides if we choose train dataset or test dataset
                For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp',..., 't99.bmp']
        """
        if is_train:
            data_dir = os.path.join(os.getcwd(), 'Train') # Join the Train dir to current directory
            data = glob.glob(os.path.join(data_dir, '*.*')) # make set of all dataset file path
        else:
            if test_img != '':
                return [os.path.join(os.getcwd(), test_img)]
            data_dir = os.path.join(os.path.join(os.getcwd(), 'Test'), 'Set5')
            data = glob.glob(os.path.join(data_dir, '*.*')) # make set of all dataset file path
        return data

    def make_sub_data2(data, padding, config):
        """
            Make the sub_data set
            Args:
                data : the set of all file path 
                padding : the image padding of input to label
                config : the all flags
        """
        sub_input_sequence = []
    #     sub_label_sequence = []
        for i in range(len(data)):
            input_ = cv2.imread(data[i])
            input_=cv2.resize(input_,None,fx = 2 ,fy = 2, interpolation = cv2.INTER_CUBIC)

            if len(input_.shape) == 3: # is color
                h, w, c = input_.shape
            else:
                h, w = input_.shape # is grayscale

            nx, ny = 0, 0
            for x in range(0, h - config.image_size + 1, config.stride):
                nx += 1; ny = 0
                for y in range(0, w - config.image_size + 1, config.stride):
                    ny += 1

                    sub_input = input_[x: x + config.image_size, y: y + config.image_size] # 33 * 33
    #                 sub_label = label_[x + padding: x + padding + config.label_size, y + padding: y + padding + config.label_size] # 21 * 21

                    # Reshape the subinput and sublabel
                    sub_input = sub_input.reshape([config.image_size, config.image_size, config.c_dim])
    #                 sub_label = sub_label.reshape([config.label_size, config.label_size, config.c_dim])
                    # Normialize
                    sub_input =  sub_input / 255.0
    #                 sub_label =  sub_label / 255.0

                    #cv2.imshow('im1',sub_input)
                    #cv2.imshow('im2',sub_label)
                    #cv2.waitKey(0)

                    # Add to sequence
                    sub_input_sequence.append(sub_input)
    #                 sub_label_sequence.append(sub_label)

    #     return sub_input_sequence, sub_label_sequence, nx, ny
        return sub_input_sequence, nx, ny

    def read_data(path):
        """
            Read h5 format data file

            Args:
                path: file path of desired file
                data: '.h5' file format that contains  input values
                label: '.h5' file format that contains label values 
        """
        with h5py.File(path, 'r') as hf:
            input_ = np.array(hf.get('input'))
            label_ = np.array(hf.get('label'))
            return input_, label_

    def make_data_hf2(input_, config):
        """
            Make input data as h5 file format
            Depending on 'is_train' (flag value), savepath would be change.
        """
        # Check the check dir, if not, create one
        if not os.path.isdir(os.path.join(os.getcwd(),config.checkpoint_dir)):
            os.makedirs(os.path.join(os.getcwd(),config.checkpoint_dir))

        if config.is_train:
            savepath = os.path.join(os.getcwd(), config.checkpoint_dir + '/train.h5')
        else:
            savepath = os.path.join(os.getcwd(), config.checkpoint_dir + '/test.h5')

        with h5py.File(savepath, 'w') as hf:
            hf.create_dataset('input', data=input_)

    def merge(images, size, c_dim):
        """
            images is the sub image set, merge it
        """
        h, w = images.shape[1], images.shape[2]
        print(h, w)
        print(size[0], size[1])
        img = np.zeros((h*size[0], w*size[1], c_dim))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h : j * h + h, i * w : i * w + w, :] = image
            #cv2.imshow('srimg',img)
            #cv2.waitKey(0)

        return img

    def input_setup2(config):
        """
            Read image files and make their sub-images and saved them as a h5 file format
        """

        # Load data path, if is_train False, get test data
        data = load_data(config.is_train, config.test_img)

        padding = abs(config.image_size - config.label_size) // 2

        # Make sub_input and sub_label, if is_train false more return nx, ny
    #     sub_input_sequence, sub_label_sequence, nx, ny = make_sub_data2(data, padding, config)
        sub_input_sequence, nx, ny = make_sub_data2(data, padding, config)

        # Make list to numpy array. With this transform
        arrinput = np.asarray(sub_input_sequence) # [?, 33, 33, 3]
    #     arrlabel = np.asarray(sub_label_sequence) # [?, 21, 21, 3]

        make_data_hf2(arrinput, config)

        return nx, ny