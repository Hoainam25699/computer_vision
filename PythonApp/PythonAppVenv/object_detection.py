import cv2
import numpy as np
import imutils
from skimage.morphology import opening
import matplotlib.pyplot as plt


class CountObject:
  def __init__(self, image_path):
    self.image_path = image_path
    

    self.new_image_path= image_path.replace('.jpg', "").replace('.png', '').replace('.jpeg','') + "_new_with_equal_hist.png"    
    self.kernel =  np.ones((3,3),np.uint8)

  def read_image(self):
    img = cv2.imread(self.image_path)
    return img

  # create a function which will display the image
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


    image_gray =   cv2.equalizeHist(image_gray)

    self.show(image_gray, "image_gray")
    self.show( image_blur_gray, "image_Blur_gray")

    init = 128
    kernel = np.ones((3, 3), np.uint8)

    # use threshold to separate image into 2 parts 
    image_res, image_thresh = cv2.threshold(image_gray, init, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    image_res_blur, image_thresh_blur = cv2.threshold(image_blur_gray, init, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)

    self.show(image_thresh, "image_thresh")
    self.show(image_thresh_blur, "image_thresh_blur")

    # filter noise
    opening = cv2.morphologyEx(image_thresh, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(image_thresh, cv2.MORPH_CLOSE, kernel)

    self.show(opening, "opening")
    self.show(closing, "closing")


    img_dilation = cv2.dilate(image_thresh, kernel, iterations=1)
    img_erosion_dilation = cv2.erode(img_dilation, kernel, iterations=1)

    self.show(img_dilation, "img_dilation")
    self.show(img_erosion_dilation, "img_erosion_dilation")


    # clean noise after dilation and erosion
    img_erode = cv2.medianBlur(img_dilation, 3)

    self.show(img_erode, "img_erode")
    return img_erode
  

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


count_object_1 = CountObject('test_1.png')
count_object_1.count_obj()

# count_object_2 = CountObject('test_2.png')
# count_object_2.count_obj()

# count_object_4 = CountObject('test_4.png')
# count_object_4.count_obj()











