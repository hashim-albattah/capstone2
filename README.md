

The Shoe Image Classiﬁer

Hashim Al-Battah

![](Capstone%202.001.jpeg)



Convolutional Neural Networks (CNNs)

● A class of deep neural networks which is

mostly used to do image recognition,

image classiﬁcation, object detection, etc.

● An image is taken as an input

● It assigns importance to its various

aspects/features in the image

● This is the basis of how it is able to

diﬀerentiate one image to another

![](Capstone%202.002.jpeg)



My CNN Shoe Image Classiﬁer

● Inspired by the UT Zappos50K project at University of Texas - Austin

● UT Zappos50K is a large shoe dataset consisting of 50,025 catalog images

collected from Zappos.com

○ This was the data used for my model

● The images are divided into 4 categories (shoes, sandals, slippers, and boots)

○ Followed by a subcategory of functionality type, of which is followed by brands

● For the sake of my project, I had cleaned that directory up into just 4 categories

![](Capstone%202.003.jpeg)



Image Preprocessing

● Before training the model on the data, the images were preprocessed

○ Used ﬂow\_from\_directory to help subset data into Training, Validation, and Testing sets

○ Training images were rescaled, sheared, zoomed, and ﬂipped horizontally

■ The purpose was to help the model detect deeper features

■ Adding any more augmentations than these did not appear helpful

● The Validation and Test Images were not augmented

![](Capstone%202.004.jpeg)



The CNN Model and Results

● Training halted due to GPU conﬁguration issues

● The trained model up to that halt provided the following accuracy metrics:

○ A Training Accuracy of 93.3%

○ A slightly higher Validation Accuracy

■ Due to Training Preprocessing

○ A Test Accuracy of 93.2%

● Original best was about ~91%

○ Improved with Dropout Decrease

■ 0.5 -> 0.2

![](Capstone%202.005.jpeg)



Future Considerations

● Training that model for a longer period of time (with diﬀerent parameters, too)

● Further categorizing data into shoe\_type-functionality\_type (i.e. “shoes-oxfords”)

○ Might be able to detect deeper features than just the main shoe categories

● Training a more complicated CNN

● Considering AWS in a timely manner

Questions?

![](Capstone%202.006.jpeg)
