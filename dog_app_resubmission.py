
# coding: utf-8

# # Artificial Intelligence Nanodegree
# 
# ## Convolutional Neural Networks
# 
# ## Project: Write an Algorithm for a Dog Identification App 
# 
# ---
# 
# In this notebook, some template code has already been provided for you, and you will need to implement additional functionality to successfully complete this project. You will not need to modify the included code beyond what is requested. Sections that begin with **'(IMPLEMENTATION)'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section, and the specifics of the implementation are marked in the code block with a 'TODO' statement. Please be sure to read the instructions carefully! 
# 
# > **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.
# 
# In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut.  Markdown cells can be edited by double-clicking the cell to enter edit mode.
# 
# The rubric contains _optional_ "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. If you decide to pursue the "Stand Out Suggestions", you should include the code in this IPython notebook.
# 
# 
# 
# ---
# ### Why We're Here 
# 
# In this notebook, you will make the first steps towards developing an algorithm that could be used as part of a mobile or web app.  At the end of this project, your code will accept any user-supplied image as input.  If a dog is detected in the image, it will provide an estimate of the dog's breed.  If a human is detected, it will provide an estimate of the dog breed that is most resembling.  The image below displays potential sample output of your finished project (... but we expect that each student's algorithm will behave differently!). 
# 
# ![Sample Dog Output](images/sample_dog_output.png)
# 
# In this real-world setting, you will need to piece together a series of models to perform different tasks; for instance, the algorithm that detects humans in an image will be different from the CNN that infers dog breed.  There are many points of possible failure, and no perfect algorithm exists.  Your imperfect solution will nonetheless create a fun user experience!
# 
# ### The Road Ahead
# 
# We break the notebook into separate steps.  Feel free to use the links below to navigate the notebook.
# 
# * [Step 0](#step0): Import Datasets
# * [Step 1](#step1): Detect Humans
# * [Step 2](#step2): Detect Dogs
# * [Step 3](#step3): Create a CNN to Classify Dog Breeds (from Scratch)
# * [Step 4](#step4): Use a CNN to Classify Dog Breeds (using Transfer Learning)
# * [Step 5](#step5): Create a CNN to Classify Dog Breeds (using Transfer Learning)
# * [Step 6](#step6): Write your Algorithm
# * [Step 7](#step7): Test Your Algorithm
# 
# ---
# <a id='step0'></a>
# ## Step 0: Import Datasets
# 
# ### Import Dog Dataset
# 
# In the code cell below, we import a dataset of dog images.  We populate a few variables through the use of the `load_files` function from the scikit-learn library:
# - `train_files`, `valid_files`, `test_files` - numpy arrays containing file paths to images
# - `train_targets`, `valid_targets`, `test_targets` - numpy arrays containing onehot-encoded classification labels 
# - `dog_names` - list of string-valued dog breed names for translating labels

# In[1]:


from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob

# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset('data/dog_images/train')
valid_files, valid_targets = load_dataset('data/dog_images/valid')
test_files, test_targets = load_dataset('data/dog_images/test')

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("data/dog_images/train/*/"))]
dog_breeds = len(dog_names)

# print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))


# ### Import Human Dataset
# 
# In the code cell below, we import a dataset of human images, where the file paths are stored in the numpy array `human_files`.

# In[2]:


import random
random.seed(8675309)

# load filenames in shuffled human dataset
human_files = np.array(glob("data/lfw/*/*"))
random.shuffle(human_files)

# print statistics about the dataset
print('There are %d total human images.' % len(human_files))


# ---
# <a id='step1'></a>
# ## Step 1: Detect Humans
# 
# We use OpenCV's implementation of [Haar feature-based cascade classifiers](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html) to detect human faces in images.  OpenCV provides many pre-trained face detectors, stored as XML files on [github](https://github.com/opencv/opencv/tree/master/data/haarcascades).  We have downloaded one of these detectors and stored it in the `haarcascades` directory.
# 
# In the next code cell, we demonstrate how to use this detector to find human faces in a sample image.

# In[3]:


import cv2                
import matplotlib.pyplot as plt                        
get_ipython().run_line_magic('matplotlib', 'inline')

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# load color (BGR) image
# img = cv2.imread(human_files[13232])
img = cv2.imread(train_files[22])
# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find faces in image
faces = face_cascade.detectMultiScale(gray)

# print number of faces detected in the image
print('Number of faces detected:', len(faces))

# get bounding box for each detected face
for (x,y,w,h) in faces:
    # add bounding box to color image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
plt.show()


# Before using any of the face detectors, it is standard procedure to convert the images to grayscale.  The `detectMultiScale` function executes the classifier stored in `face_cascade` and takes the grayscale image as a parameter.  
# 
# In the above code, `faces` is a numpy array of detected faces, where each row corresponds to a detected face.  Each detected face is a 1D array with four entries that specifies the bounding box of the detected face.  The first two entries in the array (extracted in the above code as `x` and `y`) specify the horizontal and vertical positions of the top left corner of the bounding box.  The last two entries in the array (extracted here as `w` and `h`) specify the width and height of the box.
# 
# ### Write a Human Face Detector
# 
# We can use this procedure to write a function that returns `True` if a human face is detected in an image and `False` otherwise.  This function, aptly named `face_detector`, takes a string-valued file path to an image as input and appears in the code block below.

# In[4]:


# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


# ### (IMPLEMENTATION) Assess the Human Face Detector
# 
# __Question 1:__ Use the code cell below to test the performance of the `face_detector` function.  
# - What percentage of the first 100 images in `human_files` have a detected human face?  
# - What percentage of the first 100 images in `dog_files` have a detected human face? 
# 
# Ideally, we would like 100% of human images with a detected face and 0% of dog images with a detected face.  You will see that our algorithm falls short of this goal, but still gives acceptable performance.  We extract the file paths for the first 100 images from each of the datasets and store them in the numpy arrays `human_files_short` and `dog_files_short`.
# 
# __Answer:__ 

# In[5]:


human_files_short = human_files[:100]
dog_files_short = train_files[:100]
# Do NOT modify the code above this line.

## TODO: Test the performance of the face_detector algorithm 
## on the images in human_files_short and dog_files_short.
face_detected_counter = 0
ctr = 0
erroneous_faces_detections = []
for human_filename in human_files_short:
    if face_detector(human_filename):
        face_detected_counter = face_detected_counter + 1
    else:
        erroneous_faces_detections.append(ctr)
    ctr = ctr + 1    
print(" > detected " + str(face_detected_counter) + " faces out of " + str(len(human_files_short)) +
      ", i.e. " + "{0:.2f}%".format(face_detected_counter/len(human_files_short) * 100))
print(" > unable to detect faces at: ", erroneous_faces_detections)

face_detected_counter = 0
ctr = 0
erroneous_dogs_faces = []
for dog_filename in dog_files_short:
    if face_detector(dog_filename):
        face_detected_counter = face_detected_counter + 1
        erroneous_dogs_faces.append(ctr)
    ctr = ctr + 1
print(" > detected " + str(face_detected_counter) + " human faces out of " + str(len(dog_files_short)) +
      " dog images, i.e. " + "{0:.2f}%".format(face_detected_counter/len(dog_files_short) * 100))
print(" > erroneous face predictions: ", erroneous_dogs_faces)


# In[6]:


### see where it fails
import matplotlib.image as mpimg

def show_image(path):
    print("showing an image at " + path)
    img = mpimg.imread(path)
    plt.axis('off')
    imgplot = plt.imshow(img)


# In[7]:


show_image(human_files_short[erroneous_faces_detections[0]])


# In[8]:


show_image(dog_files_short[erroneous_dogs_faces[6]])


# In[9]:


### obviously -- there is a human!
show_image(dog_files_short[erroneous_dogs_faces[5]])


# In[10]:


face_detector(dog_files_short[erroneous_dogs_faces[5]])


# __Question 2:__ This algorithmic choice necessitates that we communicate to the user that we accept human images only when they provide a clear view of a face (otherwise, we risk having unneccessarily frustrated users!). In your opinion, is this a reasonable expectation to pose on the user? If not, can you think of a way to detect humans in images that does not necessitate an image with a clearly presented face?
# 
# __Answer:__
# 
# We suggest the face detector from OpenCV as a potential way to detect human images in your algorithm, but you are free to explore other approaches, especially approaches that make use of deep learning :).  Please use the code cell below to design and test your own face detection algorithm.  If you decide to pursue this _optional_ task, report performance on each of the datasets.

# In[11]:


## (Optional) TODO: Report the performance of another  
## face detection algorithm on the LFW dataset
### Feel free to use as many code cells as needed.


# # Optional task: USING MTCNN for the human face detection
# 
# This addresses the ___Question 2___ technically, because it can detect incomplete human faces as in example below and by tuning the certainty value I am getting less false positive. BUT, yes, I think asking a user to provide a more clear input is perfectly fine, it is a computer program we are working with, and it has its own potential quirks which might not be fixable in any easy way, making the user aware is good way to mitigate possible mistakes.
# 
# ___In addition___, it is possible to look for eyes within a face rectangle -- if not found, then we can invalidate the face discovery. 

# In[12]:


from mtcnn.mtcnn import MTCNN
detector = MTCNN()

def face_detector2(img_path):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    
    ### MY LAPTOP'S gpu IS ONLY 4GB, it fails on large images
    height, width = img.shape[:2]
    if height > 1500 or width > 1500:
        img = cv2.resize(img, (0,0), fx=0.45, fy=0.45)
    height, width = img.shape[:2]
        
    ### yet another issue I saw is that some faces detected 
    ### by this implementation are really small bounding boxes
    ### so we filter by those, expecting a face to be at least 5% of the X or Y dimension
    intermediate_res = detector.detect_faces(img)
    res = []
    for idx in range(len(intermediate_res)):
        bounding_box = intermediate_res[idx]['box']
        if (bounding_box[2] > width/20 or bounding_box[3] > height/20) and           (intermediate_res[idx]['confidence'] > 0.978):
            res.append(bounding_box)
    
    return len(res) > 0


# In[13]:


img = cv2.cvtColor(cv2.imread(human_files_short[erroneous_faces_detections[0]]), cv2.COLOR_BGR2RGB)

res = detector.detect_faces(img)
bounding_box = res[0]['box']
keypoints = res[0]['keypoints']
print(res)
cv2.rectangle(img,
              (bounding_box[0], bounding_box[1]),
              (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
              (0,155,255), 2)
bounding_box = res[1]['box']
keypoints = res[1]['keypoints']
cv2.rectangle(img,
              (bounding_box[0], bounding_box[1]),
              (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
              (0,155,0), 2)
plt.imshow(img)
plt.show()


# ## Test the OPTIONAL human face detector

# In[14]:


human_files_short = human_files[:100]
dog_files_short = train_files[:100]
# Do NOT modify the code above this line.

## TODO: Test the performance of the face_detector algorithm 
## on the images in human_files_short and dog_files_short.
face_detected_counter = 0
ctr = 0
erroneous_faces_detections = []
for human_filename in human_files_short:
    if face_detector2(human_filename):
        face_detected_counter = face_detected_counter + 1
    else:
        erroneous_faces_detections.append(ctr)
    ctr = ctr + 1    
print(" > detected " + str(face_detected_counter) + " faces out of " + str(len(human_files_short)) +
      ", i.e. " + "{0:.2f}%".format(face_detected_counter/len(human_files_short) * 100))
print(" > unable to detect faces at: ", erroneous_faces_detections)

face_detected_counter = 0
ctr = 0
erroneous_dogs_faces = []
for dog_filename in dog_files_short:
    if face_detector2(dog_filename):
        face_detected_counter = face_detected_counter + 1
        erroneous_dogs_faces.append(ctr)
    ctr = ctr + 1
print(" > detected " + str(face_detected_counter) + " human faces out of " + str(len(dog_files_short)) +
      " dog images, i.e. " + "{0:.2f}%".format(face_detected_counter/len(dog_files_short) * 100))
print(" > erroneous face predictions: ", erroneous_dogs_faces)


# In[15]:



img = cv2.cvtColor(cv2.imread(dog_files_short[erroneous_dogs_faces[1]]), cv2.COLOR_BGR2RGB)

res = detector.detect_faces(img)

height, width = img.shape[:2]

print(height, width)

for idx in range(len(res)):
    bounding_box = res[idx]['box']
    keypoints = res[idx]['keypoints']
    print(res[idx]['confidence'])
    cv2.rectangle(img,
              (bounding_box[0], bounding_box[1]),
              (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
              (0,155,255), 2)
    print(bounding_box[2], bounding_box[3], height/20, bounding_box[3] > height/20)

plt.imshow(img)
plt.show()


# ## OPTIONAL TASK Notes: 
# It seems like MTCNN technique is performing better but obviously more computationally expensive. Another good thing about this implementation is the ability to tune the cutoff for certainty of the face detection -- which enables to make better predictions, i.e. with less false positives.

# ---
# <a id='step2'></a>
# ## Step 2: Detect Dogs
# 
# In this section, we use a pre-trained [ResNet-50](http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006) model to detect dogs in images.  Our first line of code downloads the ResNet-50 model, along with weights that have been trained on [ImageNet](http://www.image-net.org/), a very large, very popular dataset used for image classification and other vision tasks.  ImageNet contains over 10 million URLs, each linking to an image containing an object from one of [1000 categories](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a).  Given an image, this pre-trained ResNet-50 model returns a prediction (derived from the available categories in ImageNet) for the object that is contained in the image.

# In[16]:


from keras.applications.resnet50 import ResNet50

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')


# ### Pre-process the Data
# 
# When using TensorFlow as backend, Keras CNNs require a 4D array (which we'll also refer to as a 4D tensor) as input, with shape
# 
# $$
# (\text{nb_samples}, \text{rows}, \text{columns}, \text{channels}),
# $$
# 
# where `nb_samples` corresponds to the total number of images (or samples), and `rows`, `columns`, and `channels` correspond to the number of rows, columns, and channels for each image, respectively.  
# 
# The `path_to_tensor` function below takes a string-valued file path to a color image as input and returns a 4D tensor suitable for supplying to a Keras CNN.  The function first loads the image and resizes it to a square image that is $224 \times 224$ pixels.  Next, the image is converted to an array, which is then resized to a 4D tensor.  In this case, since we are working with color images, each image has three channels.  Likewise, since we are processing a single image (or sample), the returned tensor will always have shape
# 
# $$
# (1, 224, 224, 3).
# $$
# 
# The `paths_to_tensor` function takes a numpy array of string-valued image paths as input and returns a 4D tensor with shape 
# 
# $$
# (\text{nb_samples}, 224, 224, 3).
# $$
# 
# Here, `nb_samples` is the number of samples, or number of images, in the supplied array of image paths.  It is best to think of `nb_samples` as the number of 3D tensors (where each 3D tensor corresponds to a different image) in your dataset!

# In[17]:


from keras.preprocessing import image                  
from tqdm import tqdm

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


# ### Making Predictions with ResNet-50
# 
# Getting the 4D tensor ready for ResNet-50, and for any other pre-trained model in Keras, requires some additional processing.  First, the RGB image is converted to BGR by reordering the channels.  All pre-trained models have the additional normalization step that the mean pixel (expressed in RGB as $[103.939, 116.779, 123.68]$ and calculated from all pixels in all images in ImageNet) must be subtracted from every pixel in each image.  This is implemented in the imported function `preprocess_input`.  If you're curious, you can check the code for `preprocess_input` [here](https://github.com/fchollet/keras/blob/master/keras/applications/imagenet_utils.py).
# 
# Now that we have a way to format our image for supplying to ResNet-50, we are now ready to use the model to extract the predictions.  This is accomplished with the `predict` method, which returns an array whose $i$-th entry is the model's predicted probability that the image belongs to the $i$-th ImageNet category.  This is implemented in the `ResNet50_predict_labels` function below.
# 
# By taking the argmax of the predicted probability vector, we obtain an integer corresponding to the model's predicted object class, which we can identify with an object category through the use of this [dictionary](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a). 

# In[18]:


from keras.applications.resnet50 import preprocess_input, decode_predictions

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))


# In[19]:


show_image(train_files[22])


# In[20]:


# predicts 208, i.e. 208: 'Labrador retriever', https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
ResNet50_predict_labels(train_files[22])


# ### Write a Dog Detector
# 
# While looking at the [dictionary](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a), you will notice that the categories corresponding to dogs appear in an uninterrupted sequence and correspond to dictionary keys 151-268, inclusive, to include all categories from `'Chihuahua'` to `'Mexican hairless'`.  Thus, in order to check to see if an image is predicted to contain a dog by the pre-trained ResNet-50 model, we need only check if the `ResNet50_predict_labels` function above returns a value between 151 and 268 (inclusive).
# 
# We use these ideas to complete the `dog_detector` function below, which returns `True` if a dog is detected in an image (and `False` if not).

# In[21]:


### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 


# ### (IMPLEMENTATION) Assess the Dog Detector
# 
# __Question 3:__ Use the code cell below to test the performance of your `dog_detector` function.  
# - What percentage of the images in `human_files_short` have a detected dog?  
# - What percentage of the images in `dog_files_short` have a detected dog?
# 
# __Answer:__ 

# In[22]:


### TODO: Test the performance of the dog_detector function
### on the images in human_files_short and dog_files_short.

human_files_short = human_files[:100]
dog_files_short = train_files[:100]
# Do NOT modify the code above this line.
dog_detected_counter = 0
ctr = 0
erroneous_dogs_in_humans = []
for human_filename in human_files_short:
    if dog_detector(human_filename):
        dog_detected_counter = dog_detected_counter + 1
        erroneous_dogs_in_humans.append(ctr)
    ctr = ctr + 1    
print(" > detected " + str(dog_detected_counter) + " dogs out of " + str(len(human_files_short)) +
      " human images, i.e. " + "{0:.2f}%".format(dog_detected_counter/len(human_files_short) * 100))
print(" > erroneous dog detection at: ", erroneous_dogs_in_humans)

dog_detected_counter = 0
ctr = 0
erroneous_dogs = []
for dog_filename in dog_files_short:
    if dog_detector(dog_filename):
        dog_detected_counter = dog_detected_counter + 1
    else:
        erroneous_dogs.append(ctr)
    ctr = ctr + 1
print(" > detected " + str(dog_detected_counter) + " dog faces out of " + str(len(dog_files_short)) +
      " dog images, i.e. " + "{0:.2f}%".format(dog_detected_counter/len(dog_files_short) * 100))
print(" > can't detect dogs at: ", erroneous_dogs)


# In[23]:


### Oh, Fidel!
show_image(human_files_short[erroneous_dogs_in_humans[0]])


# ---
# <a id='step3'></a>
# ## Step 3: Create a CNN to Classify Dog Breeds (from Scratch)
# 
# Now that we have functions for detecting humans and dogs in images, we need a way to predict breed from images.  In this step, you will create a CNN that classifies dog breeds.  You must create your CNN _from scratch_ (so, you can't use transfer learning _yet_!), and you must attain a test accuracy of at least 1%.  In Step 5 of this notebook, you will have the opportunity to use transfer learning to create a CNN that attains greatly improved accuracy.
# 
# Be careful with adding too many trainable layers!  More parameters means longer training, which means you are more likely to need a GPU to accelerate the training process.  Thankfully, Keras provides a handy estimate of the time that each epoch is likely to take; you can extrapolate this estimate to figure out how long it will take for your algorithm to train. 
# 
# We mention that the task of assigning breed to dogs from images is considered exceptionally challenging.  To see why, consider that *even a human* would have great difficulty in distinguishing between a Brittany and a Welsh Springer Spaniel.  
# 
# Brittany | Welsh Springer Spaniel
# - | - 
# <img src="images/Brittany_02625.jpg" width="100"> | <img src="images/Welsh_springer_spaniel_08203.jpg" width="200">
# 
# It is not difficult to find other dog breed pairs with minimal inter-class variation (for instance, Curly-Coated Retrievers and American Water Spaniels).  
# 
# Curly-Coated Retriever | American Water Spaniel
# - | -
# <img src="images/Curly-coated_retriever_03896.jpg" width="200"> | <img src="images/American_water_spaniel_00648.jpg" width="200">
# 
# 
# Likewise, recall that labradors come in yellow, chocolate, and black.  Your vision-based algorithm will have to conquer this high intra-class variation to determine how to classify all of these different shades as the same breed.  
# 
# Yellow Labrador | Chocolate Labrador | Black Labrador
# - | -
# <img src="images/Labrador_retriever_06457.jpg" width="150"> | <img src="images/Labrador_retriever_06455.jpg" width="240"> | <img src="images/Labrador_retriever_06449.jpg" width="220">
# 
# We also mention that random chance presents an exceptionally low bar: setting aside the fact that the classes are slightly imabalanced, a random guess will provide a correct answer roughly 1 in 133 times, which corresponds to an accuracy of less than 1%.  
# 
# Remember that the practice is far ahead of the theory in deep learning.  Experiment with many different architectures, and trust your intuition.  And, of course, have fun! 
# 
# ### Pre-process the Data
# 
# We rescale the images by dividing every pixel in every image by 255.

# In[24]:


from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255


# ### (IMPLEMENTATION) Model Architecture
# 
# Create a CNN to classify dog breed.  At the end of your code cell block, summarize the layers of your model by executing the line:
#     
#         model.summary()
# 
# We have imported some Python modules to get you started, but feel free to import as many modules as you need.  If you end up getting stuck, here's a hint that specifies a model that trains relatively fast on CPU and attains >1% test accuracy in 5 epochs:
# 
# ![Sample CNN](images/sample_cnn.png)
#            
# __Question 4:__ Outline the steps you took to get to your final CNN architecture and your reasoning at each step.  If you chose to use the hinted architecture above, describe why you think that CNN architecture should work well for the image classification task.
# 
# __Answer:__ I have used your design, with a minor modifications for strides and pooling. I found this configuration works a bit better. But it's just fiddling, I have no idea why its better. I tried different confiiguration, paddings, etc... Some of those increased the parameters count over 300K but there was a very small gain in the accuracy. I guess that really advanced networks are needed.
# 

# In[25]:


### TODO: Define your architecture.

from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

model = Sequential()

img_width, img_height = 224, 224

model.add(Conv2D(filters = 16, kernel_size = (5,5), strides = (2,2), padding = 'valid', activation = 'relu',
                input_shape = (img_width, img_height, 3)))
model.add(Conv2D(filters = 32, kernel_size = (5,5), strides = (3,3), padding = 'valid', activation = 'relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=None, padding='valid'))
model.add(Conv2D(filters = 64, kernel_size = (2,2), strides = (1,1), padding = 'valid', activation = 'relu'))
model.add(GlobalMaxPooling2D())
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(dog_breeds, activation='softmax'))

model.summary()


# ### Compile the Model

# In[26]:


model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# ### (IMPLEMENTATION) Train the Model
# 
# Train your model in the code cell below.  Use model checkpointing to save the model that attains the best validation loss.
# 
# You are welcome to [augment the training data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html), but this is not a requirement. 

# In[27]:


from keras.callbacks import ModelCheckpoint  

### TODO: specify the number of epochs that you would like to use to train the model.

epochs = 40

### Do NOT modify the code below this line.

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', 
                               verbose=1, save_best_only=True)

model.fit(train_tensors, train_targets, 
          validation_data=(valid_tensors, valid_targets),
          epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)


# ### Load the Model with the Best Validation Loss

# In[28]:


model.load_weights('saved_models/weights.best.from_scratch.hdf5')


# ### Test the Model
# 
# Try out your model on the test dataset of dog images.  Ensure that your test accuracy is greater than 1%.

# In[29]:


# get index of predicted dog breed for each image in test set
dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# report test accuracy
test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)


# ---
# <a id='step4'></a>
# ## Step 4: Use a CNN to Classify Dog Breeds
# 
# To reduce training time without sacrificing accuracy, we show you how to train a CNN using transfer learning.  In the following step, you will get a chance to use transfer learning to train your own CNN.
# 
# ### Obtain Bottleneck Features

# In[30]:


bottleneck_features = np.load('data/bottleneck_features/DogVGG16Data.npz')
train_VGG16 = bottleneck_features['train']
valid_VGG16 = bottleneck_features['valid']
test_VGG16 = bottleneck_features['test']


# ### Model Architecture
# 
# The model uses the the pre-trained VGG-16 model as a fixed feature extractor, where the last convolutional output of VGG-16 is fed as input to our model.  We only add a global average pooling layer and a fully connected layer, where the latter contains one node for each dog category and is equipped with a softmax.

# In[31]:


from keras.layers import GlobalAveragePooling2D

VGG16_model = Sequential()
VGG16_model.add(GlobalAveragePooling2D(input_shape=train_VGG16.shape[1:]))
VGG16_model.add(Dense(133, activation='softmax'))

VGG16_model.summary()


# ### Compile the Model

# In[32]:


VGG16_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# ### Train the Model

# In[87]:


checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.VGG16.hdf5', 
                               verbose=1, save_best_only=True)

VGG16_model.fit(train_VGG16, train_targets, 
          validation_data=(valid_VGG16, valid_targets),
          epochs=50, batch_size=20, callbacks=[checkpointer], verbose=1)


# ### Load the Model with the Best Validation Loss

# In[88]:


VGG16_model.load_weights('saved_models/weights.best.VGG16.hdf5')


# ### Test the Model
# 
# Now, we can use the CNN to test how well it identifies breed within our test dataset of dog images.  We print the test accuracy below.

# In[89]:


# get index of predicted dog breed for each image in test set
VGG16_predictions = [np.argmax(VGG16_model.predict(np.expand_dims(feature, axis=0))) for feature in test_VGG16]

# report test accuracy
test_accuracy = 100*np.sum(np.array(VGG16_predictions)==np.argmax(test_targets, axis=1))/len(VGG16_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)


# ### Predict Dog Breed with the Model

# In[94]:


from extract_bottleneck_features import *

# top_N defines how many predictions to return
top_N = 3

def VGG16_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_VGG16(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = VGG16_model.predict(bottleneck_feature)
    prediction = predicted_vector[0]
    
    breeds_predicted = [dog_names[idx] for idx in np.argsort(prediction)[::-1][:top_N]]
    confidence_predicted = np.sort(prediction)[::-1][:top_N]
    
    # return dog breed that is predicted by the model
    return breeds_predicted, confidence_predicted


# In[95]:


VGG16_predict_breed(train_files[2])


# In[96]:


show_image(train_files[2])


# ---
# <a id='step5'></a>
# ## Step 5: Create a CNN to Classify Dog Breeds (using Transfer Learning)
# 
# You will now use transfer learning to create a CNN that can identify dog breed from images.  Your CNN must attain at least 60% accuracy on the test set.
# 
# In Step 4, we used transfer learning to create a CNN using VGG-16 bottleneck features.  In this section, you must use the bottleneck features from a different pre-trained model.  To make things easier for you, we have pre-computed the features for all of the networks that are currently available in Keras.  These are already in the workspace, at /data/bottleneck_features.  If you wish to download them on a different machine, they can be found at:
# - [VGG-19](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG19Data.npz) bottleneck features
# - [ResNet-50](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz) bottleneck features
# - [Inception](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogInceptionV3Data.npz) bottleneck features
# - [Xception](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogXceptionData.npz) bottleneck features
# 
# The files are encoded as such:
# 
#     Dog{network}Data.npz
#     
# where `{network}`, in the above filename, can be one of `VGG19`, `Resnet50`, `InceptionV3`, or `Xception`.  
# 
# The above architectures are downloaded and stored for you in the `/data/bottleneck_features/` folder.
# 
# This means the following will be in the `/data/bottleneck_features/` folder:
# 
# `DogVGG19Data.npz`
# `DogResnet50Data.npz`
# `DogInceptionV3Data.npz`
# `DogXceptionData.npz`
# 
# 
# 
# ### (IMPLEMENTATION) Obtain Bottleneck Features
# 
# In the code block below, extract the bottleneck features corresponding to the train, test, and validation sets by running the following:
# 
#     bottleneck_features = np.load('/data/bottleneck_features/Dog{network}Data.npz')
#     train_{network} = bottleneck_features['train']
#     valid_{network} = bottleneck_features['valid']
#     test_{network} = bottleneck_features['test']

# In[78]:


### TODO: Obtain bottleneck features from another pre-trained CNN.
bottleneck_features = np.load('data/bottleneck_features/DogInceptionV3Data.npz')
train_inceptionv3 = bottleneck_features['train']
valid_inceptionv3 = bottleneck_features['valid']
test_inceptionv3 = bottleneck_features['test']


# ### (IMPLEMENTATION) Model Architecture
# 
# Create a CNN to classify dog breed.  At the end of your code cell block, summarize the layers of your model by executing the line:
#     
#         <your model's name>.summary()
#    
# __Question 5:__ Outline the steps you took to get to your final CNN architecture and your reasoning at each step.  Describe why you think the architecture is suitable for the current problem.
# 
# __Answer:__ I found that Inception performs much better than VGG-16 and just used it. So, it is an experimental evaluation which set me on this path. Personally, it's difficult for me to tell why exactly Inception outperforms the VGG net. It could be that it trains much slower (a mentions about this I found at few places) and more epochs and images are required to achieve a better performance. It could also be that Inception is just suited better for this task as it combines multiple (sized differently) convolutions within the same module, thus having a better grasp on distinct dog breed features. 
# 
# I tried different batch sizes and numbers of epochs. From my experience this changes the classification results, sometimes they are getting really bad in real case predictions. Who would guess that it can be so unstable, I mean sensitive.
# 
# 

# In[79]:


### TODO: Define your architecture.
from keras.layers import GlobalMaxPooling2D

InceptionV3_model = Sequential()
InceptionV3_model.add(GlobalMaxPooling2D(input_shape=train_inceptionv3.shape[1:]))
InceptionV3_model.add(Dense(133, activation='softmax'))

InceptionV3_model.summary()


# ### (IMPLEMENTATION) Compile the Model

# In[80]:


### TODO: Compile the model.
InceptionV3_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# ### (IMPLEMENTATION) Train the Model
# 
# Train your model in the code cell below.  Use model checkpointing to save the model that attains the best validation loss.  
# 
# You are welcome to [augment the training data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html), but this is not a requirement. 

# In[84]:


### TODO: Train the model.
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.InceptionV3.hdf5', 
                               verbose=1, save_best_only=True)

InceptionV3_model.fit(train_inceptionv3, train_targets, 
          validation_data=(valid_inceptionv3, valid_targets),
          epochs=60, batch_size=20, callbacks=[checkpointer], verbose=1)


# ### (IMPLEMENTATION) Load the Model with the Best Validation Loss

# In[85]:


### TODO: Load the model weights with the best validation loss.
InceptionV3_model.load_weights('saved_models/weights.best.InceptionV3.hdf5')


# ### (IMPLEMENTATION) Test the Model
# 
# Try out your model on the test dataset of dog images. Ensure that your test accuracy is greater than 60%.

# In[86]:


### TODO: Calculate classification accuracy on the test dataset.
# get index of predicted dog breed for each image in test set
InceptionV3_predictions = [np.argmax(InceptionV3_model.predict(np.expand_dims(feature, axis=0)))                            for feature in test_inceptionv3]

# report test accuracy
test_accuracy = 100*np.sum(np.array(InceptionV3_predictions)==                           np.argmax(test_targets, axis=1))/len(InceptionV3_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)


# ### (IMPLEMENTATION) Predict Dog Breed with the Model
# 
# Write a function that takes an image path as input and returns the dog breed (`Affenpinscher`, `Afghan_hound`, etc) that is predicted by your model.  
# 
# Similar to the analogous function in Step 5, your function should have three steps:
# 1. Extract the bottleneck features corresponding to the chosen CNN model.
# 2. Supply the bottleneck features as input to the model to return the predicted vector.  Note that the argmax of this prediction vector gives the index of the predicted dog breed.
# 3. Use the `dog_names` array defined in Step 0 of this notebook to return the corresponding breed.
# 
# The functions to extract the bottleneck features can be found in `extract_bottleneck_features.py`, and they have been imported in an earlier code cell.  To obtain the bottleneck features corresponding to your chosen CNN architecture, you need to use the function
# 
#     extract_{network}
#     
# where `{network}`, in the above filename, should be one of `VGG19`, `Resnet50`, `InceptionV3`, or `Xception`.

# In[45]:


### TODO: Write a function that takes a path to an image as input
### and returns the dog breed that is predicted by the model.
from extract_bottleneck_features import *

def InceptionV3_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_InceptionV3(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = InceptionV3_model.predict(bottleneck_feature)
    prediction = predicted_vector[0]
    
    breeds_predicted = [dog_names[idx] for idx in np.argsort(prediction)[::-1][:top_N]]
    confidence_predicted = np.sort(prediction)[::-1][:top_N]
    
    # return dog breed that is predicted by the model
    return breeds_predicted, confidence_predicted


# In[46]:


InceptionV3_predict_breed(train_files[2])


# In[47]:


VGG16_predict_breed(train_files[2])


# In[48]:


show_image(train_files[2])


# ## Let's do a bit more thorough testing

# In[49]:


human_files_set = human_files[:1000]
dog_files_set = train_files[:1000]


# In[59]:


### Test dog_detector
dog_detected_counter = 0
ctr = 0
erroneous_dogs_in_humans = []
for human_filename in human_files_set:
    if dog_detector(human_filename):
        dog_detected_counter = dog_detected_counter + 1
        erroneous_dogs_in_humans.append(ctr)
    ctr = ctr + 1    
print(" > detected " + str(dog_detected_counter) + " dogs out of " + str(len(human_files_set)) +
      " human images, i.e. " + "{0:.2f}%".format(dog_detected_counter/len(human_files_set) * 100))
print(" > erroneous dog detection at: ", erroneous_dogs_in_humans)

dog_detected_counter = 0
ctr = 0
erroneous_dogs = []
for dog_filename in dog_files_set:
    if dog_detector(dog_filename):
        dog_detected_counter = dog_detected_counter + 1
    else:
        erroneous_dogs.append(ctr)
    ctr = ctr + 1
print(" > detected " + str(dog_detected_counter) + " dogs out of " + str(len(dog_files_set)) +
      " dog images, i.e. " + "{0:.2f}%".format(dog_detected_counter/len(dog_files_set) * 100))
print(" > can't detect dogs at: ", erroneous_dogs)


# In[60]:


show_image(human_files_set[erroneous_dogs_in_humans[2]])


# In[61]:


show_image(dog_files_set[erroneous_dogs[2]])


# In[51]:


### Test human face detector, v1

face_detected_counter = 0
ctr = 0
erroneous_faces = []
for human_filename in human_files_set:
    if face_detector(human_filename):
        face_detected_counter = face_detected_counter + 1
    else:
        erroneous_faces.append(ctr)
    ctr = ctr + 1
print(" > detected " + str(face_detected_counter) + " faces out of " + str(len(human_files_set)) +
      ", i.e. " + "{0:.2f}%".format(face_detected_counter/len(human_files_set) * 100))
print(" > cant detect face at: ", erroneous_faces)

face_detected_counter = 0
ctr = 0
erroneous_faces_in_dogs = []
for dog_filename in dog_files_set:
    if face_detector(dog_filename):
        face_detected_counter = face_detected_counter + 1
        erroneous_faces_in_dogs.append(ctr)
    ctr = ctr + 1
print(" > detected " + str(face_detected_counter) + " human faces out of " + str(len(dog_files_set)) +
      " dog images, i.e. " + "{0:.2f}%".format(face_detected_counter/len(dog_files_set) * 100))
print(" > erroneously detected face at: ", erroneous_faces_in_dogs)


# In[52]:


### Test human face detector, v2

face_detected_counter = 0
ctr = 0
erroneous_faces = []
for human_filename in human_files_set:
    if face_detector2(human_filename):
        face_detected_counter = face_detected_counter + 1
    else:
        erroneous_faces.append(ctr)
    ctr = ctr + 1
print(" > detected " + str(face_detected_counter) + " faces out of " + str(len(human_files_set)) +
      ", i.e. " + "{0:.2f}%".format(face_detected_counter/len(human_files_set) * 100))
print(" > cant detect face at: ", erroneous_faces)

face_detected_counter = 0
ctr = 0
erroneous_faces_in_dogs = []
for dog_filename in dog_files_set:
    if face_detector2(dog_filename):
        face_detected_counter = face_detected_counter + 1
        erroneous_faces_in_dogs.append(ctr)
    ctr = ctr + 1
print(" > detected " + str(face_detected_counter) + " human faces out of " + str(len(dog_files_set)) +
      " dog images, i.e. " + "{0:.2f}%".format(face_detected_counter/len(dog_files_set) * 100))
print(" > erroneously detected face at: ", erroneous_faces_in_dogs)


# In[77]:


### whatup MTCNN
show_image(dog_files_set[erroneous_faces_in_dogs[18]])


# ---
# <a id='step6'></a>
# ## Step 6: Write your Algorithm
# 
# Write an algorithm that accepts a file path to an image and first determines whether the image contains a human, dog, or neither.  Then,
# - if a __dog__ is detected in the image, return the predicted breed.
# - if a __human__ is detected in the image, return the resembling dog breed.
# - if __neither__ is detected in the image, provide output that indicates an error.
# 
# You are welcome to write your own functions for detecting humans and dogs in images, but feel free to use the `face_detector` and `dog_detector` functions developed above.  You are __required__ to use your CNN from Step 5 to predict dog breed.  
# 
# Some sample output for our algorithm is provided below, but feel free to design your own user experience!
# 
# ![Sample Human Output](images/sample_human_output.png)
# 
# 
# ### (IMPLEMENTATION) Write your Algorithm

# In[54]:


### TODO: Write your algorithm.
### Feel free to use as many code cells as needed.
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def make_prediction(path, multiple_breeds = False):
    img = mpimg.imread(path)
    plt.axis('off')
    imgplot = plt.imshow(img)
    breeds, confidence = InceptionV3_predict_breed(path)
    # we would use the dog_detector function as it's more accurate
    # first check for dogs, if there are no dogs -- check for humans
    found_dog = dog_detector(path)
    found_face = face_detector2(path)
    if found_face:
        if found_dog:
            print("Hello! Woof woof!")
            print("Seems like a hooman and a dog... The dog looks like " + 
                  "a {}.".format(breeds[0].replace("_", " ")))
        else:
            print("Hello hooman!")
            print("If you were a dog, you\'d be a {}.".format(breeds[0].replace("_", " ")))
            
    elif found_dog:
            print("Woof woof!")
            print("You look like a {}.".format(breeds[0].replace("_", " ")))
    else:
        raise ValueError("Could not detect dogs or hoomans in the image.")


# ---
# <a id='step7'></a>
# ## Step 7: Test Your Algorithm
# 
# In this section, you will take your new algorithm for a spin!  What kind of dog does the algorithm think that __you__ look like?  If you have a dog, does it predict your dog's breed accurately?  If you have a cat, does it mistakenly think that your cat is a dog?
# 
# ### (IMPLEMENTATION) Test Your Algorithm on Sample Images!
# 
# Test your algorithm at least six images on your computer.  Feel free to use any images you like.  Use at least two human and two dog images.  
# 
# __Question 6:__ Is the output better than you expected :) ?  Or worse :( ?  Provide at least three possible points of improvement for your algorithm.
# 
# __Answer:__ I think it works with a pretty decent performance for something I can run on my laptop. I really like this project and enjoyed it very much. I like that I was able to write a wrapper which says that there is a dog and a hooman! Potentially: (1) it can be useful to detect multiple dogs and characterize them, in a way the second human face detector (MTCNN) works, (2) would be nice to outline dog faces as well, (3) I think I'd need to try yet another face detector to improve the whole thing accuracy by voting or something... or to use a threshold in MTCNN to reduce human face detection in dog images... Se the performance comparison on a 1000 of images above...

# In[55]:


## TODO: Execute your algorithm from Step 6 on
## at least 6 images on your computer.
## Feel free to use as many code cells as needed.


# In[97]:


make_prediction(dog_files_set[erroneous_faces_in_dogs[5]])


# In[98]:


make_prediction("data/dog_images/train/015.Basset_hound/Basset_hound_01112.jpg")


# In[99]:


make_prediction("my_images/internet_dawg1.jpg")


# In[100]:


make_prediction("my_images/internet_dawg2.jpg")


# In[101]:


make_prediction("my_images/internet_dawg3.jpg")


# In[102]:


make_prediction("my_images/internet_dawg4.jpg")


# In[103]:


make_prediction("my_images/internet_dawg5.jpg")


# In[104]:


make_prediction("my_images/internet_dawg6.jpg")


# In[105]:


make_prediction(dog_files_short[erroneous_dogs_faces[1]])


# # Some people

# In[106]:


make_prediction("my_images/pavel.jpg")


# In[107]:


make_prediction("my_images/dasha.jpg")


# # The Good Wife cast

# In[108]:


make_prediction("my_images/Julianna_Margulies.jpg")


# In[109]:


make_prediction("my_images/Chris_Noth.jpg")


# In[110]:


make_prediction("my_images/Josh_Charles.jpg")


# In[111]:


make_prediction("my_images/Christine_Baranski_at_Met_Opera_cropped.jpg")


# In[112]:


make_prediction("my_images/Carrie_Preston.jpg")


# # Sneaky cats

# In[90]:


make_prediction("my_images/pepe.jpg")


# In[93]:


make_prediction("my_images/totoro.jpg")


# # Please download your notebook to submit
