import cv2
import numpy as np
import imutils
from skimage.morphology import opening
import matplotlib.pyplot as plt


class CountObject:
  def __init__(self, image_path):
    self.image_path = image_path
    self.new_image_path= image_path.replace('.jpg', "").replace('.png', '').replace('.jpeg','') + "_new.png"
     

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
    gray_im = cv2.cvtColor(origin_image, cv2.COLOR_BGR2GRAY)
    plt.subplot(221)
    plt.title('Grayscale image')
    plt.imshow(gray_im, cmap="gray", vmin=0, vmax=255)

    # Contrast adjusting with gamma correction y = 1.2

    gray_correct = np.array(255 * (gray_im / 255) ** 1.2 , dtype='uint8')
    plt.subplot(222)
    plt.title('Gamma Correction y= 1.2')
    plt.imshow(gray_correct, cmap="gray", vmin=0, vmax=255)
    # Contrast adjusting with histogramm equalization
    gray_equ = cv2.equalizeHist(gray_im)
    plt.subplot(223)
    plt.title('Histogram equilization')
    plt.imshow(gray_correct, cmap="gray", vmin=0, vmax=255)\
    

    thresh = cv2.adaptiveThreshold(gray_correct, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 255, 19)
    thresh = cv2.bitwise_not(thresh)
    plt.subplot(221)
    plt.title('Local adapatative Threshold')
    plt.imshow(thresh, cmap="gray", vmin=0, vmax=255)


    # Dilatation et erosion
    kernel = np.ones((15,15), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    img_erode = cv2.erode(img_dilation,kernel, iterations=1)
    # clean all noise after dilatation and erosion
    img_erode = cv2.medianBlur(img_erode, 7)
    plt.subplot(221)
    plt.title('Dilatation + erosion')
    plt.imshow(img_erode, cmap="gray", vmin=0, vmax=255)


    ret, labels = cv2.connectedComponents(img_erode)
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue == 0] = 0

    plt.subplot(222)
    plt.title('Objects counted:'+ str(ret-1))
    plt.imshow(labeled_img)
    print('objects number is:', ret-1)
    plt.show()


      
    cv2.imwrite(self.new_image_path, labeled_img)

    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    return 
  
  def count_obj(self):
    img = self.read_image()
    last_image = self.process(img)

# count_object_1 = CountObject('objets1.jpg')
# count_object_1.count_obj()

# count_object_2 = CountObject('objets2.jpg')
# count_object_2.count_obj()


# count_object_3 = CountObject('objets3.jpg')
# count_object_3.count_obj()

# count_object_4 = CountObject('objets4.jpg')
# count_object_4.count_obj()


count_object_5 = CountObject('test_3.png')
count_object_5.count_obj()








