import cv2
import numpy as np
import imutils
from skimage.morphology import opening
import matplotlib.pyplot as plt


class CountObject:
  def __init__(self, image_path):
    self.image_path = image_path
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
    image_blur_gray = cv2.cvtColor(image_blur, cv2.COLOR_BGR2GRAY)

    init = 108
    kernel = np.ones((3, 3), np.uint8)

    # use threshold to separate image into 2 parts 
    image_res, image_thresh = cv2.threshold(image_blur_gray, init, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)

    # filter noise
    opening = cv2.morphologyEx(image_thresh, cv2.MORPH_OPEN, kernel)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, last_image = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
    last_image = np.uint8(last_image)
    return last_image

  def count(self, image, last_image):
    cnts = cv2.findContours(last_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)


    for (i, c) in enumerate(cnts):
      ((x, y), _) = cv2.minEnclosingCircle(c)
      cv2.putText(image, "#{}".format(i + 1), (int(x) - 45, int(y) + 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)
      cv2.drawContours(image, [c], -1, (0, 255, 0), 2)

    # tis funtion disply the image which i have already describe above
    self.display(image, len(cnts))
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    return 
  
  def count_obj(self):
    img = self.read_image()
    last_image = self.process(img)
    re = self.count(img, last_image)

count_object_1 = CountObject('test_1.png')
count_object_1.count_obj()

count_object_2 = CountObject('test_2.png')
count_object_2.count_obj()

# count_object_3 = CountObject('test_3.png')
# count_object_3.count_obj()

# count_object_4 = CountObject('test_4.png')
# count_object_4.count_obj()


# count_object_5 = CountObject('photo2.jpg')
# count_object_5.count_obj()





