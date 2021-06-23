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

    # GaussianBlur
    # blur = cv2.medianBlur(gray_im, 5)
    blur = cv2.GaussianBlur(gray_im, (5,5), 3)
    plt.subplot(222)
    plt.title('GaussianBlur'), plt.xticks([]), plt.yticks([])
    plt.imshow(blur, cmap="gray", vmin=0, vmax=255)

    
    canny = cv2.Canny(blur, 50, 150, 3)

    dilate = cv2.dilate(canny, (3,3), iterations = 1)
    (cnt, herachy) = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rbg = cv2.cvtColor(dilate, cv2.COLOR_BGR2RGB)
    cv2.drawContours(rbg, cnt, -1, (0,255, 0), 2)


    cv2.imwrite(self.new_image_path, dilate)
    self.display(rbg, len(cnt))
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    return 
  

  def count_obj(self):
    img = self.read_image()
    last_image = self.process(img)


count_object_5 = CountObject('C:/Users/Admin/Documents/GitHub/computer_vision/PythonApp/PythonAppVenv/test_1.png')
count_object_5.count_obj()