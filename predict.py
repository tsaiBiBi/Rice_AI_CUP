import numpy as np
import os, cv2, re, time
from rice_classifier import classify, cluster, find_center

WINDOW_SIZE = 32
STEP = WINDOW_SIZE//4

# input img \
# return cut_img_array, xy, isCrop(01)
def sliding_window(img):
    num_cut_img = (img.shape[0]//STEP +1) * (img.shape[1]//STEP +1)
    center_xy = np.empty((num_cut_img, 2))
    cut_img = np.empty((num_cut_img, WINDOW_SIZE, WINDOW_SIZE, 3), dtype="uint8")
    pad_img = np.pad(img, ((0,WINDOW_SIZE),(0,WINDOW_SIZE),(0,0)))
    isCrop = np.empty((num_cut_img,))
    print('pad', pad_img.shape)
    count = 0
    for y in range(0, img.shape[0], STEP):
        for x in range(0, img.shape[1], STEP):
            cut_img[count] = pad_img[
                y:y+WINDOW_SIZE,
                x:x+WINDOW_SIZE,
                :
            ]
            center_xy[count] = [
                (x + x + WINDOW_SIZE) // 2,
                (y + y + WINDOW_SIZE) // 2,
            ]
            count += 1
    isCrop = classify(cut_img) # predict
    return (cut_img, center_xy, isCrop)

def main():
    os.makedirs("./submit_result/csv")
    os.makedirs("./submit_result/image")
    os.makedirs("./submit_result/scan")

    for dataset in ['__test_private__', '__test_public__']:
        for file in os.listdir(dataset):
            name='.'.join(re.split('[.]', file)[:-1])
            img=cv2.imread(dataset+'/'+file)
            cut_img, center_xy, isCrop = sliding_window(img)
            isCrop_idx = np.where(isCrop == 1)[0]

            # scan
            img=cv2.imread(dataset+'/'+file)
            center_xy = center_xy.astype('int')
            for xy in center_xy[isCrop_idx]:
                cv2.circle(img, tuple(xy), 4, (255,0,0), -1)
            cv2.imwrite('submit_result/scan/'+name+'.jpg', img)

            # 取得中心位置
            # labels = cluster(center_xy[isCrop_idx])
            # rice_xy = find_center(center_xy[isCrop_idx], labels)
            img=cv2.imread(dataset+'/'+file)
            mask=np.zeros(img.shape[:2], dtype='uint8')
            # cv2.imshow('Mask', mask)
            for xy in center_xy[isCrop_idx]:
                cv2.circle(mask, tuple(xy), 4, (255,), -1)
            # cv2.imshow('Org', mask)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            rate=STEP*2
            mask=cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (rate, rate)))
            mask=cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (rate, rate)))
            # cv2.imshow('Dilate', mask)
            # cv2.imshow('erode', mask)
            cc_count, cc_map, cc_box, rice_xy = cv2.connectedComponentsWithStats(mask)
            # 儲存結果
            rice_xy = rice_xy.astype('int')
            for center_xy in rice_xy:
                cv2.circle(img, tuple(center_xy), 6, (0,0,255), -1)
            cv2.imwrite('submit_result/image/'+name+'.jpg', img)

            csv_str='\n'.join([','.join(i) for i in rice_xy.astype('str')])
            open('submit_result/csv/'+name+'.csv', 'w').write(csv_str)
            img=cv2.imread(dataset+'/'+file)

if __name__ == '__main__':
    main()
    # name = '0154_new'
    # img=cv2.imread('./img/0154_new.jpg')
    # cut_img, center_xy, isCrop = sliding_window(img)
    # isCrop_idx = np.where(isCrop == 1)[0]
    # labels = cluster(center_xy[isCrop_idx])
    # rice_xy = find_center(center_xy[isCrop_idx], labels)
    # csv_str='\n'.join([','.join(i) for i in rice_xy.astype('str')])
    # open('submit_result/csv/'+name+'.csv', 'w').write(csv_str)