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
    plt.title('Grayscale image')
    plt.imshow(gray_im, cmap="gray", vmin=0, vmax=255)

    # Contrast adjusting with gamma correction y = 1.2
    gray_correct = np.array(255 * (gray_im / 255) ** 1.2 , dtype='uint8')
    plt.subplot(222)
    plt.title('Gamma Correction y= 1.2')
    plt.imshow(gray_correct, cmap="gray", vmin=0, vmax=255)

    # Fourier Transform
    dft = cv2.dft(np.float32(gray_im),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
    plt.subplot(223)
    plt.title('Magnitude Spectrum')
    plt.imshow(gray_correct, cmap="gray", vmin=0, vmax=255)

    # remove sin
    dft_shift[230][238] = 0
    dft_shift[230][222] = 0

    # inverse and return the result
    f_ishift = np.fft.ifftshift(dft_shift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
    plt.subplot(224)
    plt.title('after remove sin noise')
    plt.imshow(gray_correct, cmap="gray", vmin=0, vmax=255)
    imageio.imwrite('img.png', img_back)
    image1 = cv2.imread('img.png')
    gray_im1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    # Contrast adjusting with gamma correction y = 1.2
    gray_correct = np.array(255 * (gray_im1 / 255) ** 1.2 , dtype='uint8')
    plt.subplot(221)
    plt.title('Gamma Correction y= 1.2'), plt.xticks([]), plt.yticks([])
    plt.imshow(gray_correct, cmap="gray", vmin=0, vmax=255)

    #Contrast adjusting with histogramm equalization
    gray_equ = cv2.equalizeHist(gray_im1)
    plt.subplot(222)
    plt.title('Histogram equilization'), plt.xticks([]), plt.yticks([])
    plt.imshow(gray_correct, cmap="gray", vmin=0, vmax=255)

    # thresshold
    res,thresh = cv2.threshold(gray_correct,174,255,cv2.THRESH_BINARY)
    plt.subplot(223)
    plt.title('global Threshold'), plt.xticks([]), plt.yticks([])
    plt.imshow(thresh, cmap="gray", vmin=0, vmax=255)

    #opening
    kernel = np.ones((7,7),np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    plt.subplot(224)
    plt.title('opening'), plt.xticks([]), plt.yticks([])
    plt.imshow(opening, cmap="gray", vmin=0, vmax=255)
    erode = cv2.erode(opening,(19,19), iterations=5)
    opening2 = cv2.morphologyEx(erode, cv2.MORPH_OPEN, kernel)
    canny = cv2.Canny(opening2, 50, 100, 3)


    dilate = cv2.dilate(canny, (1,1), iterations = 1)
    (cnt, heirachy) = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
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


count_object_5 = CountObject('test_3.png')
count_object_5.count_obj()
