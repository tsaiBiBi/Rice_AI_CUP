import  cv2, os, random
import numpy as np

source_root = "./data_32/"
dst_root = "./augmentation/"

def img_arg():
    
    for file in os.listdir(source_root + '0/'):
        # load data
        img=cv2.imread(source_root + '0/' + file)
        
        # change coler
        bright = random.randrange(0, int(np.sum(img[0,0,:])/3*0.5)+1)
        contrast = random.randrange(8,12)*0.1
        result = change_color(img, contrast, bright)
        # store
        store_path = dst_root + '0/' + file.split('.')[0] + '_1.jpg'
        cv2.imwrite(store_path, result)
        
        # change coler
        bright = random.randrange(-int(np.sum(img[0,0,:])/3*0.5)-1, 0)
        contrast = random.randrange(8,15)*0.1
        result = change_color(img, contrast, bright)
        # store
        store_path = dst_root + '0/' + file.split('.')[0] + '_2.jpg'
        cv2.imwrite(store_path, result)

        # move
        cv2.imwrite(dst_root + '0/' + file, img)

    for file in os.listdir(source_root + '1/'):
        # load data
        img=cv2.imread(source_root + '1/' + file)
        
        # change coler
        bright = random.randrange(0, int(np.sum(img[0,0,:])/3*0.5)+1)
        contrast = random.randrange(8,15)*0.1
        result = change_color(img, contrast, bright)
        # store
        store_path = dst_root + '1/' + file.split('.')[0] + '_1.jpg'
        cv2.imwrite(store_path, result)
        
        # change coler
        bright = random.randrange(-int(np.sum(img[0,0,:])/3*0.5)-1, 0)
        contrast = random.randrange(8,15)*0.1
        result = change_color(img, contrast, bright)
        # store
        store_path = dst_root + '1/' + file.split('.')[0] + '_2.jpg'
        cv2.imwrite(store_path, result)
        
        cv2.imwrite(dst_root + '1/' + file, img)

def change_color(img, contrast, bright):
    rows,cols,channels = img.shape
    dst=img.copy()
    for i in range(rows):
        for j in range(cols):
            for c in range(channels):
                color=img[i,j][c]*contrast+bright
                if color>255:           # 防止像素值越界（0~255）
                    dst[i,j][c]=255
                elif color<0:           # 防止像素值越界（0~255）
                    dst[i,j][c]=0
                else:
                    dst[i,j][c]=color
    return dst

if __name__ == '__main__':
    os.makedirs(dst_root + "0")
    os.makedirs(dst_root + "1")
    img_arg()