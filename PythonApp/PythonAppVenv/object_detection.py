import cv2
import numpy as np
import imutils
from skimage.morphology import opening
import matplotlib.pyplot as plt


class CountObject:
  def __init__(self, image_path, equal_hist=None):
    self.image_path = image_path
    
    if equal_hist is None:
      self.equal_hist = False
      self.new_image_path= image_path.replace('.jpg', "").replace('.png', '').replace('.jpeg','') + "_new.png"
    else: 
      self.new_image_path= image_path.replace('.jpg', "").replace('.png', '').replace('.jpeg','') + "_new_with_equal_hist.png"
    
    self.kernel =  np.ones((3,3),np.uint8)

  def read_image(self):
    img = cv2.imread(self.image_path)
    return img

  # create a funtion which will display the image
  def display(self, img, count, cmap="gray"):
    f, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].imshow(img, cmap="gray")
    axs[1].imshow(img, cmap="gray")
    axs[1].set_title("Total Count = {}".format(count))
    plt.show()

  
  def show(self, image, name, waitKeyTime=2000):
    cv2.imshow(name, image)
    cv2.waitKey(waitKeyTime)
  
  def show_plt(self, img, dst):
    print("The diff after denoise")
    plt.subplot(121),plt.imshow(img)
    plt.subplot(122),plt.imshow(dst)
    plt.show()

  def process(self, origin_image):

    # remove noise 
    image = cv2.fastNlMeansDenoising(origin_image, None, 30.0, 7, 21)

    self.show_plt(origin_image, image)

    # smoothing
    image_blur = cv2.medianBlur(image, 5)
    # transform to gray image
    image_gray = cv2.cvtColor(image_blur, cv2.COLOR_BGR2GRAY)
    image_blur_gray = cv2.cvtColor(image_blur, cv2.COLOR_BGR2GRAY)


    # balanced histogram image
    if self.equal_hist == True:
      image_gray =   cv2.equalizeHist(image_gray)

    # self.show(image_gray, "image_gray")
    # self.show( image_blur_gray, "image_Blur_gray")

    init = 128
    kernel = np.ones((3, 3), np.uint8)

    # use threshold to separate image into 2 parts 
    image_res, image_thresh = cv2.threshold(image_gray, init, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    image_res_blur, image_thresh_blur = cv2.threshold(image_blur_gray, init, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)

    # self.show(image_thresh, "image_thresh")
    # self.show(image_thresh_blur, "image_thresh_blur")

    # filter noise
    opening = cv2.morphologyEx(image_thresh, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(image_thresh, cv2.MORPH_CLOSE, kernel)

    # self.show(opening, "opening")
    # self.show(closing, "closing")


    img_dilation = cv2.dilate(image_thresh, kernel, iterations=1)
    img_erosion_dilation = cv2.erode(img_dilation, kernel, iterations=1)

    # self.show(img_dilation, "img_dilation")
    # self.show(img_erosion_dilation, "img_erosion_dilation")


    # clean noise after dilation and erosion
    img_erode = cv2.medianBlur(img_dilation, 3)

    self.show(img_erode, "img_erode")
    return img_erode

  def process_with_fourier_transform(self, origin_image):
    img =  cv2.imread('test_3.png', 0)

    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    mask = np.zeros((rows, cols, 2), np.uint8)
    r = 100
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 1
    # Band Pass Filter - Concentric circle mask, only the points living in concentric circle are ones
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    mask = np.zeros((rows, cols, 2), np.uint8)
    r_out = 85
    r_in = 25
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = np.logical_and(((x - center[0]) ** 2 + (y - center[1]) ** 2 >= r_in ** 2),
                              ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= r_out ** 2))
    mask[mask_area] = 1

    fshift = dft_shift * mask

    fshift_mask_mag = 2000 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1] + 1))

    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(2,2,1)
    ax1.imshow(img, cmap='gray')
    ax1.title.set_text('Input Image')
    ax2 = fig.add_subplot(2,2,2)
    ax2.imshow(magnitude_spectrum, cmap='gray')
    ax2.title.set_text('FFT of image')
    ax3 = fig.add_subplot(2,2,3)
    ax3.imshow(fshift_mask_mag, cmap='gray')
    ax3.title.set_text('FFT + Mask')
    ax4 = fig.add_subplot(2,2,4)
    ax4.imshow(img_back, cmap='gray')
    ax4.title.set_text('After inverse FFT')
    plt.show()

    img_back= cv2.normalize(img_back, None, alpha=1.2, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img_erode = cv2.medianBlur(img_back, 3)

    init = 25
    kernel = np.ones((3, 3), np.uint8)

    # use threshold to separate image into 2 parts 
    image_res, image_thresh = cv2.threshold(img_erode, init, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    # clean noise after dilation and erosion
    img_erode_ = cv2.medianBlur(image_thresh, 3)
    dilation = cv2.dilate(img_erode_,kernel,iterations = 2)
    mg_erosion_dilation = cv2.erode(dilation,kernel,iterations = 5)

    # mg_erosion_dilation = cv2.erode(img_erode_, kernel, iterations=1)
    thresh = cv2.morphologyEx(mg_erosion_dilation, cv2.MORPH_OPEN, kernel)
    
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, None, None, None, 8, cv2.CV_32S)

    #get CC_STAT_AREA component as stats[label, COLUMN] 
    areas = stats[1:,cv2.CC_STAT_AREA]

    result = np.zeros((labels.shape), np.uint8)

    for i in range(0, nlabels - 1):
        if areas[i] >= 150:   #keep
            result[labels == i + 1] = 255

    cv2.imshow("Result", result)

    self.show(thresh, "thresh")
    self.show(mg_erosion_dilation, "mg_erosion_dilation")

    return result
    # return img_erode
    
  def count(self, image, last_image):
    cnts = cv2.findContours(last_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for (i, c) in enumerate(cnts):
      ((x, y), _) = cv2.minEnclosingCircle(c)
      cv2.drawContours(image, [c], -1, (0, 255, 0), 2)

    self.display(image, len(cnts))
    cv2.imwrite(self.new_image_path, image)

    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    return 

  def count_obj(self):
    img = self.read_image()
    last_image = self.process(img)
    re = self.count(img, last_image)


  def count_obj_with_fourier_transform(self):
    img = self.read_image()
    last_image = self.process_with_fourier_transform(img)
    re = self.count(img, last_image)


# count_object_1 = CountObject('test_1.png')
# count_object_1.count_obj()

# count_object_2 = CountObject('test_2.png')
# count_object_2.count_obj()


count_object_3 = CountObject('test_3.png', 0)
count_object_3.count_obj_with_fourier_transform()

# count_object_4 = CountObject('test_4.png')
# count_object_4.count_obj()











