import numpy as np
import os
import cv2

def number_of_frames(video_directory):
    vs = cv2.VideoCapture(video_directory)
        
    # Try calculating the total number of frames in the video file
    try:
        prop = cv2.CAP_PROP_FRAME_COUNT
        total = int(vs.get(prop))
        print("[INFO] {} total frames in video".format(total))

    # Raise an error if the number of frames couldn't be calculated
    except:
        print("[INFO] could not determine # of frames in video")
        print("[INFO] no approx. completion time can be provided")

def cartoon(cartoon_video_directory):
    try:
        vs = cv2.VideoCapture(cartoon_video_directory)
    except:
        print("Did not find video in the given directory.")
        return 
    frame_count = 0
    c = 1963
    while True:
        (grabbed, frame) = vs.read()
            
        if not grabbed:
            break

        if frame_count % 30 == 0 and not frame.all() == 0 and frame_count > 10000:
            frame = cv2.resize(frame ,(500, 500))
            cv2.imwrite('CartoonData/' + str(c) + '.jpg', frame)
            c+=1
        frame_count+=1



def original(directory):

    original_imgs = []
    for dirs in os.listdir(directory):
        img = cv2.imread(directory + '/' + dirs)
        img = cv2.resize(img, (500, 500))
        original_imgs.append(img)
        
    original_imgs = np.array(original_imgs)
    return original_imgs

def smooth(images):
    kernel_size = 5
    kernel = np.ones((kernel_size,kernel_size), np.float32)
    gauss = cv2.getGaussianKernel(kernel_size, 0)
    gauss = gauss * gauss.transpose(1,0)
    #dst = cv2.filter2D(image, -1, kernel)
    smooth_images = []
    for img in images:
        pad_img = np.pad(img, ((2,2), (2,2), (0,0)), mode = "reflect")
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_img, 100, 200)
        dilation = cv2.dilate(edges, kernel)
            
        gauss_img = np.copy(img)
        idx = np.where(dilation != 0)

        for i in range(np.sum(dilation != 0)):
            gauss_img[idx[0][i], idx[1][i], 0] = np.sum(np.multiply(pad_img[idx[0][i] : idx[0][i]+kernel_size, idx[1][i] : idx[1][i]+kernel_size, 0], gauss))
            gauss_img[idx[0][i], idx[1][i], 1] = np.sum(np.multiply(pad_img[idx[0][i] : idx[0][i]+kernel_size, idx[1][i] : idx[1][i]+kernel_size, 1], gauss))
            gauss_img[idx[0][i], idx[1][i], 2] = np.sum(np.multiply(pad_img[idx[0][i] : idx[0][i]+kernel_size, idx[1][i] : idx[1][i]+kernel_size, 2], gauss))
            
        smooth_images.append(gauss_img)

    return np.array(smooth_images)

def main(**kwargs):
    directory = kwargs["cartoon_directory"]
    cartoon(directory)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cartoon_directory", type=float)

    args = parser.parse_args()
    kwargs = vars(args)

    main(**kwargs)