def show_boundingbox(image_path, label_path):
    #e.g: image_path = 'E:\\VIP CUP\\dataset\\01_fisheye_day_000028.jpg'
    #label_path = 'E:\\VIP CUP\\dataset\\01_fisheye_day_000028.txt'
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    img = cv2.imread(image_path)
    w = img.shape[0]
    label = np.ceil(np.loadtxt(label_path)[:, 1:]*w).astype(int)
    for i in range(label.shape[0]):
        start_point = (label[i, 0]-label[i, 2]//2, label[i, 1]-label[i, 3]//3)
        end_point = (label[i, 0]+label[i, 2]//2, label[i, 1]+label[i, 3]//2)
        img = cv2.rectangle(img, start_point, end_point, (0, 255, 0), 3)
    plt.imshow(img)
