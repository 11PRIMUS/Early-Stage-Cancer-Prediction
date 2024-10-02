# Early-Stage-Cancer-Prediction

![cancer_detection_project_image](https://github.com/user-attachments/assets/e97022a0-2f09-44be-a7e2-5b1123851ca0)


<ol>
<li>Cancer is responsible for an estimated 10 million deaths worldwide each year, making it the second leading cause of death.</li>
<li>In 2020, there were an estimated 19.3 million new cases of cancer worldwide, with the number expected to rise to 28.4 million by 2040.</li>
<li>More specifically regarding Brain Tumor there were about 265,000 deaths worldwide in 2019, representing approximately 1.6% of all deaths from all causes.</li>
<li>Additionally with regards to Leukemia it was estimated (in 2020) that there were approximately 417,000 deaths due to this worldwide, representing approximately 3% of all cancer deaths.</li>
<li>Breast cancer is the most common cancer in women worldwide, accounting for 25% of all cancers in women.</li>
<li>More than two-thirds of cancer deaths occur in Low and Middle Income Countries, despite these countries accounting for about one-third of the global population. This meant, that even though these countries suffer from a high number of cancer patients, their economic backgrounds hinder them from being detected early on.</li>
</ol>
<br>


This model classify differnt types of cancer ,Majorly we focused on classifying:-
<ol>
 
 <ins>**Brain Tumor**</ins><br />
 <ins>**Breast Cancer**</ins><br />
 <ins>**Skin Cancer**</ins><br />
</ol>
We started searching for methods/cures available for Brain Tumor and Leukemia, what we discovered was that even though we as humans have made great strides and advancements in treating cases of Brain Tumor and Leukemia most treatments rely on the condition that the cancer is detected at an early stage. We took this as the problem we were willing and eager to solve. We recognized that the rate at which cancer cases are increasing is putting stress on the hospital as well as their staff.
So we built this model which not only detect cancer but one can predict cancer earlier from the symptomps and calssify the type of Cancer.
<p>
 <br>
</p>
<p>
We provide the users with a web interface (integrated with deep learning models) wherein they can upload the MRI or CT scans and in a matter of few seconds, they get to know whether they have Cancer (Brain Tumor/ Breast Cancer/ Leukemia). With this we aim to make it sustainable for doctors to detect a patient's cancer as well as for low income individuals to atleast get the fundamental right to basic healthcare, i.e. knowing if they have cancer and thus getting the required diagnosis.</p>

<h1>
  Brain Tumor Detection Using Deep Learning
</h1>

<ins>**💡Motivation**</ins><br />

Brain tumors are a significant health concern worldwide, often leading to severe complications if not diagnosed early. The complexity and variability of MRI images make manual analysis time-consuming and prone to human error. Our solution leverages the power of deep learning to enhance diagnostic accuracy and speed, potentially improving patient outcomes.

<ins>**🏗️Model Architecture**</ins><br />

We employ the MobileNet architecture due to its efficiency and effectiveness in image classification tasks. The model is fine-tuned for our specific use case with the following layers:

Input Shape: (224, 224, 3) (standard size for MobileNet)
Flatten Layer: Converts 2D matrices into a 1D vector
Output Layer: A Dense layer with a sigmoid activation function for binary classification.

<ins>**🛠️ Installation**</ins><br />

To run this project, please ensure you have the following prerequisites installed:

<li>Python</li> 
<li>TensorFlow/Keras</li>
<li>NumPy</li>
<li>Matplotlib</li>
<ol>
<ins>**🛠️ Code Snippet**</ins><br />

```python
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Flatten,Dense
from keras.models import Model,load_model
from keras.applications.mobilenet import MobileNet,preprocess_input
import keras
     

base_model=MobileNet(input_shape=(224,224,3),include_top=False)

for layer in base_model.layers:
  layer.trainable=False
     

X=Flatten()(base_model.output)
X=Dense(units=1,activation='sigmoid')(X)

model=Model(base_model.input,X)
model.summary()
```

<ins>**📊 Results**</ins><br />

![Image](https://github.com/user-attachments/assets/a8e59efa-4117-4506-a558-69bd64f12b25)

