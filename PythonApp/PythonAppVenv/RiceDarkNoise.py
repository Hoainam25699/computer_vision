import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
import imutils

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
    plt.title('Grayscale image'), plt.xticks([]), plt.yticks([])
    plt.imshow(gray_im, cmap="gray", vmin=0, vmax=255)

    # Contrast adjusting with gamma correction y = 0.1

    gray_correct = np.array(255 * (gray_im / 255) ** 0.1 , dtype='uint8')
    plt.subplot(222)
    plt.title('Gamma Correction y= 0.1'), plt.xticks([]), plt.yticks([])
    plt.imshow(gray_correct, cmap="gray", vmin=0, vmax=255)

    #thresshole
    res,thresh = cv2.threshold(gray_correct,155,255,cv2.THRESH_BINARY)
    plt.subplot(223)
    plt.title('global Threshold'), plt.xticks([]), plt.yticks([])
    plt.imshow(thresh, cmap="gray", vmin=0, vmax=255)

    #opening
    kernel = np.ones((5,5),np.uint8)
    opening1 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    opening = cv2.erode(opening1,(15,15), iterations=7)
    plt.subplot(224)
    plt.title('opening'), plt.xticks([]), plt.yticks([])
    plt.imshow(opening, cmap="gray", vmin=0, vmax=255)

    blur = cv2.GaussianBlur(opening, (9,9), 2)
    canny = cv2.Canny(blur, 50, 250, 3)

    dilate = cv2.dilate(canny, (1,1), iterations = 1)
    (cnt, heirachy) = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # plt.title(len(cnt))
    # plt.imshow(dilate, cmap='gray')
    # print(len(cnt))

    cv2.imwrite(self.new_image_path, dilate)
    self.display(dilate, len(cnt))
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    return 
  

  def count_obj(self):
    img = self.read_image()
    last_image = self.process(img)


count_object_5 = CountObject('test_4.png')
count_object_5.count_obj()
