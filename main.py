import pickle
import pywt
from skimage import feature
import customtkinter as ctk
import numpy as np
from PIL import Image,ImageTk
from tkinter import filedialog
import tkinter as tk
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class MyGUI:

    def __init__(self):
        ctk.set_default_color_theme("dark-blue")

        # Configure Main Window Design
        self.root = ctk.CTk()
        self.root.title("Sign Language Segmentation and Classificaiton")

        # Set the size of window to size of screen
        self.width = self.root.winfo_screenwidth()
        self.height = self.root.winfo_screenheight()
        self.root.geometry("%dx%d" % (self.width, self.height))


        # Creating upload button and putting it in the main frame (root)
        self.uploadButton = ctk.CTkButton(self.root, text="Upload Image", command=self.uploadImage)
        self.uploadButton.pack(pady=10)

        # Main frame in window and setting it to the root (main window)
        self.frame = ctk.CTkFrame(master=self.root)

        self.root.columnconfigure(0,weight=1)

        # Setting our main frame (the whole grid), 5 columns x 5 rows, uniform 'a' just makes the column or row size unresizable
        self.frame.columnconfigure(0,weight=1)
        self.frame.columnconfigure(1,weight=1,uniform = 'a')
        self.frame.columnconfigure(2,weight=1)
        self.frame.columnconfigure(3,weight=1)
        self.frame.columnconfigure(4,weight=1)
        self.frame.columnconfigure(5,weight=1)
        self.frame.rowconfigure(0,weight=1,uniform = 'a')
        self.frame.rowconfigure(1,weight=1,uniform = 'a')
        self.frame.rowconfigure(2,weight=1,uniform = 'a')
        self.frame.rowconfigure(3,weight=1,uniform = 'a')
        self.frame.rowconfigure(4,weight=1,uniform = 'a')

        # Our input image (yes images here are stored in labels) (image place holder)
        self.inputImage1 = tk.Label(self.frame,bg="#212121")
        self.inputImage1.grid(row = 0,column = 0,sticky='nesw')

        # Cleaning button on clicked gos to function self.clean
        self.cleaningButton = ctk.CTkButton(master=self.frame, text="Image Cleaning", font=("Roboto", 24), command=self.clean)
        self.cleaningButton.grid(row=0, column=1)

        # Makes a sub frame to store the image and a label describing it above it
        self.subFrame6 = ctk.CTkFrame(master=self.frame)
        self.subFrame6.grid(row=0,column=2)

        # Place label describing image inside sub frame
        self.subFrame6Label = ctk.CTkLabel(master=self.subFrame6,text="Sharpening with Unmask", font=("Roboto", 24))
        self.subFrame6Label.pack()

        # Sharpened Image
        # Place image inside sub frame
        self.outputImage11 = tk.Label(self.subFrame6,bg="#212121")
        self.outputImage11.pack()

        # Same concept of sub framing above
        self.subFrame7 = ctk.CTkFrame(master=self.frame)
        self.subFrame7.grid(row=0,column=3)

        self.subFrame7Label = ctk.CTkLabel(master=self.subFrame7,text="Gaussian Blurr",bg_color="green", font=("Roboto", 24))
        self.subFrame7Label.pack()

        # Gaussian Blurred Image
        self.outputImage12 = tk.Label(self.subFrame7, bg="#212121")
        self.outputImage12.pack()

        # The histogram placeholder
        self.histogram1 = tk.Label(self.frame,bg="#212121")
        self.histogram1.grid(row=0, column=4, sticky='nesw')

        self.inputImage3 = tk.Label(self.frame,bg="#212121")
        self.inputImage3.grid(row=1,column=0,sticky='nesw')

        self.segmentationButton = ctk.CTkButton(master=self.frame, text="Segmentation", font=("Roboto", 24), command=self.segment)
        self.segmentationButton.grid(row=1, column=1)

        self.subFrame3 = ctk.CTkFrame(master=self.frame)
        self.subFrame3.grid(row=1,column=2)

        self.subFrame3Label = ctk.CTkLabel(master=self.subFrame3,text="Canny's Edge Segmentation", font=("Roboto", 24))
        self.subFrame3Label.pack()

        # Canny's Edge Segmentation
        self.outputImage31 = tk.Label(master=self.subFrame3,bg="#212121")
        self.outputImage31.pack()

        # Otsu's Segmentation
        self.subFrame4 = ctk.CTkFrame(master=self.frame)
        self.subFrame4.grid(row=1,column=3)

        self.subFrame4Label = ctk.CTkLabel(master=self.subFrame4,text="Otsu's Segmentation",bg_color="green", font=("Roboto", 24))
        self.subFrame4Label.pack()

        self.outputImage32 = tk.Label(self.subFrame4, bg="#212121")
        self.outputImage32.pack()

        self.histogram3 = tk.Label(self.frame,bg="#212121")
        self.histogram3.grid(row=1, column=4, sticky='nesw')

        self.inputImage2 = tk.Label(self.frame,bg="#212121")
        self.inputImage2.grid(row=2,column=0,sticky='nesw')

        self.boundingButton = ctk.CTkButton(master=self.frame, text="Bounding Box", font=("Roboto", 24), command=self.boundingBox)
        self.boundingButton.grid(row=2, column=1)

        self.subFrame8 = ctk.CTkFrame(master=self.frame)
        self.subFrame8.grid(row=2,column=2)

        self.subFrame8Label = ctk.CTkLabel(master=self.subFrame8,text="Area Contour", font=("Roboto", 24))
        self.subFrame8Label.pack()

        # Area Contour
        self.outputImage21 = tk.Label(self.subFrame8,bg="#212121")
        self.outputImage21.pack()

        self.subFrame9 = ctk.CTkFrame(master=self.frame)
        self.subFrame9.grid(row=2, column=3)

        self.subFrame9Label = ctk.CTkLabel(master=self.subFrame9, text="Bounding Box Contour",bg_color="green", font=("Roboto", 18))
        self.subFrame9Label.pack()

        # Bouding Box
        self.outputImage22 = tk.Label(self.subFrame9, bg="#212121")
        self.outputImage22.pack()

        self.histogram2 = tk.Label(self.frame,bg="#212121")
        self.histogram2.grid(row=2, column=4, sticky='nesw')

        self.inputImage4 = tk.Label(self.frame,bg="#212121")
        self.inputImage4.grid(row=3,column=0,sticky='nesw')

        self.morphologoicalButton = ctk.CTkButton(master=self.frame, text="Morphological", font=("Roboto", 24), command=self.morpho)
        self.morphologoicalButton.grid(row=3, column=1)

        self.frame5 = ctk.CTkFrame(master=self.frame)
        self.frame5.grid(row=3, column=2)

        self.frame5Label = ctk.CTkLabel(master=self.frame5,text="Dilation",bg_color="green", font=("Roboto", 24))
        self.frame5Label.pack()

        # Dilated Image
        self.outputImage4 = tk.Label(self.frame5,bg="#212121")
        self.outputImage4.pack()

        self.inputImage5 = tk.Label(self.frame, bg="#212121")
        self.inputImage5.grid(row=4, column=0, sticky='nesw')

        self.classifyButton = ctk.CTkButton(master=self.frame, text="Classify", font=("Roboto", 24), command=self.classify)
        self.classifyButton.grid(row=4, column=1)

        self.histogram4 = tk.Label(self.frame,bg="#212121")
        self.histogram4.grid(row=3, column=4, sticky='nesw')

        self.subFrame1 = ctk.CTkFrame(master=self.frame)
        self.subFrame1.grid(row=4,column=2)

        self.classifySVM = tk.Label(self.subFrame1,bg="#212121",font=("Roboto",24),fg="white")
        self.classifySVM.grid(row=1, column=0, sticky='nesw')

        self.classifySVMTitle = ctk.CTkLabel(master=self.subFrame1, text="Classification using SVM",bg_color="green", font=("Roboto", 24))
        self.classifySVMTitle.grid(row=0, column=0, sticky='nesw')


        self.frame.pack(fill="both",expand=True)

        self.root.mainloop()

    def uploadImage(self):
        # Allowed image inputs, jpg, jpeg and png and opens a window to take in an image
        filename = filedialog.askopenfilename(initialdir="/images", title="Select Image",filetypes=(("jpg images","*.jpg"),("png images","*.png"),("jpeg images","*.jpeg")))
        self.path = filename
        # Reads input image from above path
        imageCv = cv2.imread(filename)
        imageCv = cv2.cvtColor(imageCv,cv2.COLOR_BGR2GRAY)
        self.image_from_file = imageCv.copy()
        # 270x270 just for view in gui
        imageCv = cv2.resize(imageCv,(240,240))

        # Darkens an image (needed for bright handed people)
        imageCv = cv2.multiply(imageCv, np.array([0.5]))

        # Copys this image for later use in other functions
        self.original_image = imageCv.copy()

        # Display the image in gui
        # Image is a function in Pillow library, as gui cant take the image directly,
        # it needs to be converted to Pillow then to gui
        image = Image.fromarray(imageCv)
        self.photoImage1 = ImageTk.PhotoImage(image)
        self.inputImage1['image'] = self.photoImage1

    def clean(self):
        if self.original_image is not None:
            # Gets coppied image from above to numpy
            original_image_np = np.array(self.original_image)

            # Smooth image with gaussian blur with kernel of size (5,5) (represents a wider area for blurring) and
            # standard deviation of 2, (higher number means more blurred)
            cleaned_image_np = cv2.GaussianBlur(original_image_np, (5,5), 2)

            # Copies cleaned image for later uses
            self.cleaned_image = cleaned_image_np.copy()


            cv2.imwrite("C:/Users/coola/Desktop/pics/cleaned.jpg", cleaned_image_np)

            # 2nd cleaning technique, Sharpening using Unmask method as lecture
            # Calculate the high-pass filter by subtracting the cleaned image from the original image
            high_pass = cv2.subtract(original_image_np, cleaned_image_np)
            # Define a scaling factor to control the strength of the high-pass filter
            scaling_factor = 5.0
            # Adjust the high-pass filter by multiplying it with the scaling factor
            adjusted_high_pass = cv2.multiply(high_pass, scaling_factor)
            # Combine the adjusted high-pass filter with the original image to create a sharpened image
            sharpened = cv2.add(original_image_np, adjusted_high_pass)


            # Plot the histogram using matplotlib
            # Explanation:
            # [cleaned_image_np]: The image(s) for which the histogram is calculated.
            # [0]: The channel (0 for grayscale). For color images, multiple channels can be specified.
            # None: The mask, which is not used here (set to None).
            # [256]: The number of bins in the histogram (256 for pixel intensities ranging from 0 to 255).
            # [0, 256]: The range of pixel values to consider for the histogram.
            hist = cv2.calcHist([cleaned_image_np], [0], None, [256], [0, 256])
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(hist, color='black')
            ax.set_title('Grayscale Histogram')
            ax.set_xlabel('Pixel Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, linestyle='--', alpha=0.5)


            # Display Histogram
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            pil_image = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb()).resize((700,240))
            self.photoHisto1 = ImageTk.PhotoImage(pil_image)
            self.histogram1['image'] = self.photoHisto1

            # Display Gaussian blurred image
            cleaned_image_pil = Image.fromarray(cleaned_image_np)
            self.photoImage22 = ImageTk.PhotoImage(cleaned_image_pil)
            self.outputImage12['image'] = self.photoImage22
            self.inputImage3['image'] = self.photoImage22

            # Display Sharpened Image
            sharpened_image = Image.fromarray(sharpened)
            # Convert the PIL Image to a PhotoImage for displaying in a Tkinter GUI
            self.photoImage21 = ImageTk.PhotoImage(sharpened_image)
            self.outputImage11['image'] = self.photoImage21

    def segment(self):
        if self.cleaned_image is not None:
            # Gets bounded image coppied above to numpy
            cleaned_image = np.array(self.cleaned_image)

            # Performs Otsu's Threshholding on cleaned image
            # Note that in otsu's method, 0 and 255 (thresholding numbers) doesn't matter, as it finds its optimal value
            _, thresh = cv2.threshold(cleaned_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            self.thresh = thresh.copy()
            cv2.imwrite("C:/Users/coola/Desktop/pics/segmented.jpg", thresh)

            # Cannys thresholding
            # low_threshold: The lower threshold for edge detection. Pixels with gradient values
            # below this threshold are considered non-edges.

            # high_threshold: The higher threshold for edge detection. Pixels with gradient values
            # above this threshold are considered strong edges. Pixels between the low and high
            # thresholds are considered weak edges and are included if they are connected to strong edges.
            low_threshold = 50
            high_threshold = 70
            edges = cv2.Canny(cleaned_image, low_threshold, high_threshold)

            # Plot the histogram using matplotlib
            # Explanation:
            # [cleaned_image_np]: The image(s) for which the histogram is calculated.
            # [0]: The channel (0 for grayscale). For color images, multiple channels can be specified.
            # None: The mask, which is not used here (set to None).
            # [256]: The number of bins in the histogram (256 for pixel intensities ranging from 0 to 255).
            # [0, 256]: The range of pixel values to consider for the histogram.
            hist = cv2.calcHist([thresh], [0], None, [256], [0, 256])
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(hist, color='black')
            ax.set_title('Grayscale Histogram')
            ax.set_xlabel('Pixel Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, linestyle='--', alpha=0.5)

            # Displays the segmented image
            # Convert Matplotlib plot to Pillow Image
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            # 'RGB': Specifies the color mode of the resulting PIL Image. Assumes the canvas contains RGB data.
            # canvas.get_width_height(): Retrieves the width and height of the canvas in pixels.
            # canvas.tostring_rgb(): Converts the RGB pixel data of the canvas to a byte string.
            pil_image = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb()).resize((700,240))
            self.photoHisto3 = ImageTk.PhotoImage(pil_image)
            self.histogram3['image'] = self.photoHisto3

            # Display the otsu's segmented image
            segmented = Image.fromarray(thresh)
            self.photoImage41 = ImageTk.PhotoImage(segmented)
            self.outputImage32['image'] = self.photoImage41
            self.inputImage2['image'] = self.photoImage41

            # Display the canny's edge image
            edged = Image.fromarray(edges)
            self.photoImage42 = ImageTk.PhotoImage(edged)
            self.outputImage31['image'] = self.photoImage42

    def boundingBox(self):
        if self.cleaned_image is not None:
            # Gets threshed_image image coppied above to numpy
            threshed_image = np.array(self.thresh)

            # Explanation:
            # thresh: input image from above
            # cv2.RETR_CCOMP: Retrieval mode specifying a two-level hierarchy. The outer contours
            # are considered as components (top-level), and holes are considered as components (second-level).

            # cv2.CHAIN_APPROX_NONE: Contour approximation method specifying that all the boundary points
            # of the contours should be stored. No compression or approximation is applied.

            # The resulting 'contours' variable contains a list of contours, where each contour is
            # represented as a list of points (coordinates) forming the contour boundary.
            contours, _ = cv2.findContours(threshed_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

            # Finding the largest contour area (the hand)
            c = max(contours, key=cv2.contourArea)

            # Gets the corners of the rectangle to draw
            x, y, w, h = cv2.boundingRect(c)

            # We changed it to bgr just to show contour box and line color (pink)
            threshed_image = cv2.cvtColor(threshed_image,cv2.COLOR_GRAY2BGR)

            # Draws the rectangle on the cleaned image
            cv2.rectangle(threshed_image, (x, y), (x + w, y + h), (255, 0, 255), 5)

            # Crops the image based on the bouding box corners (location)
            cropped_image = threshed_image[y:y + h, x:x + w]

            # Switch it back to gray scale
            cropped_image = cv2.cvtColor(cropped_image,cv2.COLOR_BGR2GRAY)

            self.cropped_image = cropped_image.copy()
            cv2.imwrite("C:/Users/coola/Desktop/pics/box.jpg", cropped_image)

            # Plot the histogram using matplotlib
            # Explanation:
            # [cleaned_image_np]: The image(s) for which the histogram is calculated.
            # [0]: The channel (0 for grayscale). For color images, multiple channels can be specified.
            # None: The mask, which is not used here (set to None).
            # [256]: The number of bins in the histogram (256 for pixel intensities ranging from 0 to 255).
            # [0, 256]: The range of pixel values to consider for the histogram.
            hist = cv2.calcHist([threshed_image], [0], None, [256], [0, 256])
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(hist, color='black')
            ax.set_title('Grayscale Histogram')
            ax.set_xlabel('Pixel Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, linestyle='--', alpha=0.5)

            # Display the bouding box histogram
            # Convert Matplotlib plot to Pillow Image
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            # 'RGB': Specifies the color mode of the resulting PIL Image. Assumes the canvas contains RGB data.
            # canvas.get_width_height(): Retrieves the width and height of the canvas in pixels.
            # canvas.tostring_rgb(): Converts the RGB pixel data of the canvas to a byte string.
            pil_image = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb()).resize((700,240))
            self.photoHisto2 = ImageTk.PhotoImage(pil_image)
            self.histogram2['image'] = self.photoHisto2

            # Display the bouding box image
            boundingbox_image = Image.fromarray(threshed_image)
            cropped_image = Image.fromarray(cropped_image)
            self.photoImage3 = ImageTk.PhotoImage(boundingbox_image)
            self.photoImageCropped = ImageTk.PhotoImage(cropped_image)
            self.outputImage22['image'] = self.photoImage3
            self.inputImage4['image'] = self.photoImageCropped

            # Finds the line contour around the image
            threshed_image = np.array(self.thresh)
            threshed_image = cv2.cvtColor(threshed_image,cv2.COLOR_GRAY2BGR)
            for contour in contours:
                area = cv2.contourArea(contour)
                print(area)
                cv2.drawContours(threshed_image,[contour],-1,(255,0,255),2)

            # Displays the line contour
            bounding_line_image = Image.fromarray(threshed_image)
            self.photoImage32 = ImageTk.PhotoImage(bounding_line_image)
            self.outputImage21['image'] = self.photoImage32



    def morpho(self):
        if self.cropped_image is not None:
            # Gets cropped_image image coppied above to numpy
            cropped_image = np.array(self.cropped_image)

            # The kernel size for dilation of integers
            kernel = np.ones((5, 5), np.uint8)  # Adjust the kernel size as needed
            dilated_image = cv2.dilate(cropped_image, kernel, iterations=1)

            self.dilated_image = dilated_image.copy()
            cv2.imwrite("C:/Users/coola/Desktop/pics/dilated.jpg", dilated_image)

            # Plot the histogram using matplotlib
            # Explanation:
            # [cleaned_image_np]: The image(s) for which the histogram is calculated.
            # [0]: The channel (0 for grayscale). For color images, multiple channels can be specified.
            # None: The mask, which is not used here (set to None).
            # [256]: The number of bins in the histogram (256 for pixel intensities ranging from 0 to 255).
            # [0, 256]: The range of pixel values to consider for the histogram.
            hist = cv2.calcHist([dilated_image], [0], None, [256], [0, 256])
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(hist, color='black')
            ax.set_title('Grayscale Histogram')
            ax.set_xlabel('Pixel Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, linestyle='--', alpha=0.5)

            # Displays the dilation histogram
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            # 'RGB': Specifies the color mode of the resulting PIL Image. Assumes the canvas contains RGB data.
            # canvas.get_width_height(): Retrieves the width and height of the canvas in pixels.
            # canvas.tostring_rgb(): Converts the RGB pixel data of the canvas to a byte string.
            pil_image = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb()).resize((700,240))
            self.photoHisto4 = ImageTk.PhotoImage(pil_image)
            self.histogram4['image'] = self.photoHisto4

            # Displays the dilated image in gui
            dilated = Image.fromarray(dilated_image)
            self.photoImage5 = ImageTk.PhotoImage(dilated)
            self.outputImage4['image'] = self.photoImage5
            self.inputImage5['image'] = self.photoImage5

    def classify(self):
        # THIS IS KMEANS CLUSTERING PART

        # path = self.path
        # img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        #
        # X=img.reshape(-1,3)
        # kmeans = KMeans(n_clusters=2, n_init=10)
        # kmeans.fit(X)
        # segmented_img = kmeans.cluster_centers_[kmeans.labels_]
        # segmented_img = segmented_img.reshape(img.shape)
        # # plt.imshow(segmented_img/255)
        # #segmented_img = segmented_img / 255
        # segmented_img_GRAY = cv2.cvtColor(segmented_img.astype('uint8'), cv2.COLOR_BGR2GRAY)
        #
        # max = np.max(segmented_img_GRAY)
        #
        # _, thresholded_img = cv2.threshold(segmented_img_GRAY, max-5, 255, cv2.THRESH_BINARY_INV)
        # pil = Image.fromarray(thresholded_img)
        # self.k = ImageTk(pil)
        # self.kclust['image'] = self.k

        categories = {0: "A",
                      1: "B",
                      2: "C",
                      3: "D",
                      4: "E",
                      5: "F",
                      6: "G",
                      7: "H",
                      8: "I",
                      9: "G",
                      10: "K",
                      11: "L",
                      12: "M",
                      13: "N",
                      14: "O",
                      15: "P",
                      16: "Q",
                      17: "R",
                      18: "S",
                      19: "T",
                      20: "U",
                      21: "V",
                      22: "W",
                      23: "X",
                      24: "Y",
                      25: "Z",
                      26: "nothing",
                      27: "space",}

        image_from_file = np.array(self.image_from_file)

        # Otsu's Segmentation
        _, thresh = cv2.threshold(image_from_file, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Finding contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        # Finding the Maximum contour
        c = max(contours, key=cv2.contourArea)

        # Getting its coordinates
        x, y, w, h = cv2.boundingRect(c)

        # Creating the box
        cv2.rectangle(image_from_file, (x, y), (x + w, y + h), (0, 255, 0), 5)

        # Cropping according to the box
        cropped_image = thresh[y:y + h, x:x + w]

        # Dilation filter size
        kernel = np.ones((5, 5), np.uint8)  # Adjust the kernel size as needed

        # Dilation
        dilated_image = cv2.dilate(cropped_image, kernel, iterations=1)

        # Resizing the image
        smaller_image = cv2.resize(dilated_image, (128, 128))


        coeffs = pywt.wavedec2(smaller_image, wavelet='db1', level=4)

        features_list = []
        features_list.append(np.sum(np.square(coeffs[0])))

        for j in range(1, min(len(coeffs), 3)):
            subband = coeffs[j]
            if isinstance(subband, tuple):
                subband = subband[0]
            features_list.append(np.sum(np.square(subband)))

        # Binary Features Extraction as paper using Lower Binary Pattern Method
        lbp = feature.local_binary_pattern(smaller_image, P=8, R=1, method="uniform")

        # Calculate LBP histogram
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 11))

        # Append histogram values to the features_list
        features_list.extend(hist)
        print(features_list)
        print(np.array(features_list).shape)
        with open('SVM.pkl', 'rb') as file:
            loaded_model = pickle.load(file)

        with open('standard_scaler.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)

        X_test = scaler.transform([features_list])
        prediction = loaded_model.predict(X_test)
        print(prediction)

        self.classifySVM['text'] = categories[prediction[0]]

MyGUI()