# Scene-Description
Being able to automatically describe the content of an image using properly formed English sentences is a very challenging task, but it could have a great impact, for instance by helping visually impaired people better understand the content of images on the web. The problem introduces a captioning task; we planned to do image-to-sentence generation. This application bridges vision and natural language, which requires a computer vision system to both localize and describe salient regions in images in natural language. The image captioning task generalizes object detection when the descriptions consist of a single word. Given a set of images and prior knowledge about the content, find the correct semantic label for the entire image.

## Image-Captioning model:
### Dataset used:
For our task, we used <a href=https://cocodataset.org/#download> MSCOCO-2017</a>, it contains 118K images each with approximately 5 different human-annotated captions.

### Data Pre-processing: 
The preprocessing phase can be split into three main procedures:  
**1. Creating the captions vocabulary:**  
First we added <start> and <end> tokens to each caption, then we created a vocabulary that contains the 5000 most frequent words in all captions.  
**2. Image preprocessing and feature extraction:**  
We first resized the images into (224, 244) to be compatible with the VGG-16 input layer, then the images were converted from RGB to BGR, and each color channel is zero-centered.
We then used a pretrained VGG-16 model to extract the features from these pre-processed images and stored them to be later passed to our model.

Captions preprocessing     |  Image preprocessing
:-------------------------:|:-------------------------:
![image](https://user-images.githubusercontent.com/44211916/191972821-e93c2c16-940e-4ac3-9edc-b0239f3ea421.png)  |  ![image](https://user-images.githubusercontent.com/44211916/191975171-84058241-dbcc-48e9-bcd3-35c0256418fa.png)

### Model Architecture:
Our image-captioning model follows the same architecture as the one proposed in the famous “Show, attend and tell” paper; it's a neural network that consists of a CNN encoder that extracts the most important features of the input image and an RNN decoder that produces the next word in the caption at each time-step and it utilizes the Bahdanau’s additive attention to focus on different parts of the image when producing each word.  
![image](https://user-images.githubusercontent.com/44211916/191975832-de6d0470-3178-47e9-abfb-eea4f54c5514.png)


### Model Training:
All of the model training was done using local gpu (nvidia gtx 1060 with 6GB).  
We used the teacher forcing technique, where we compare the word that the model produced with the correct word that is given in the target caption and compute the losses and then give the correct word to the next decoder unit.  
While during inference, we give the word that the model to the next decoder unit to produce the next word.

Training     |  Inference
:-------------------------:|:-------------------------:
![image](https://user-images.githubusercontent.com/44211916/191979775-418c730f-59cc-47ce-9b00-dd4c27324af7.png)  |  ![image](https://user-images.githubusercontent.com/44211916/191979818-3d3e9d63-9004-4d7a-b4cb-6c7d5579993f.png)

### Model Deployment:
We used plotly-dash library to deploy our model, also we added a clear dashboard to show the model architecture and the structure of our project.
![image](https://user-images.githubusercontent.com/44211916/191980118-ca97fc03-cea5-4468-86ec-b7e87a239b3d.png)
  
### References:
<ol>
    <li>Vinyals, Oriol, Alexander Toshev, Samy Bengio, and Dumitru Erhan. “Show and tell: A neural image caption generator.” In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 3156- 3164. [2015].</li>
    <li>Xu, Kelvin, Jimmy Ba, Ryan Kiros, Aaron Courville, Ruslan Salakhutdinov, Richard Zemel, and Yoshua Bengio.“Show, attend and tell: Neural image caption generation with visual attention.” arXiv preprint arXiv:1502.03044 [2015]. </li>
    <li>Karen Simonyan, Andrew Zisserman: Very Deep Convolutional Networks for Large-Scale Image Recognition. ICLR [2015]. </li>
    <li>Junyoung Chung, Caglar Gulcehre, Kyunghyun Cho, Yoshua Bengio, Empirical evaluation of gated recurrent neural networks on sequence modeling. NIPS [2014]. </li>
    <li>Karpathy, Andrej, and Li Fei-Fei. “Deep visual semantic alignments for generating image descriptions” In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 3128-3137. [2015]. </li>
    <li> <a href=https://www.tensorflow.org/tutorials/text/image_captioning>Tensorflow image-captioning tutorial.</li>
</ol>

## Team Members:
  - <a href="https://github.com/habebamohamed"> Habiba Mohamed Abdelrazek </a>
  - <a href="https://github.com/Mohamed-AN"> Mohamed Abdelrahman Nasser </a>
  - <a href="https://github.com/Mostafa-Nafie"> Mostafa Alaa Nafie </a>
  - <a href="https://github.com/SalmaElmoghazy"> Salma Elmoghazy </a>
  - <a href="https://github.com/SalmaHisham"> Salma Hisham </a>
