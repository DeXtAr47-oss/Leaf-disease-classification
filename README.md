# Leaf-disease-classification
This repository contains a deep learning-based image classification project to detect and classify leaf diseases in potato and tomato plants using Convolutional Neural Networks (CNNs) built with PyTorch.

# 📌 Problem Statement
Agriculture plays a vital role in sustaining the global economy and food supply. However, the productivity of crops is significantly impacted by plant diseases, many of which manifest visibly on the leaves. Manual identification of leaf diseases is time-consuming, labor-intensive, and often inaccurate, especially in large-scale farming.

The objective of this project is to develop an automated system for classifying plant leaf diseases using deep learning techniques. The system will be trained on a dataset containing images of healthy and diseased leaves, specially of potato and tomato plants, with multiple classes of diseases. By leveraging Convolutional Neural Networks (CNNs), the model will learn to accurately distinguish between various disease types and healthy leaves.

This solution aims to assist farmers and agricultural experts by providing a fast, scalable, and accurate method to diagnose tomato and potato plant diseases early, thereby improving crop management, reducing losses, and increasing yield.

# 🎯 Goal
  * Preprocess and augment image data to enhance model robustness.
  * Train a CNN to classify leaf images into appropriate disease categories.
  * Evaluate the model using metrics such as accuracy, precision and recall.

# 🖼️ Dataset
The datset that I have used for the project is taken from the Kaggle website (https://www.kaggle.com/datasets/arjuntejaswi/plant-village/data) Well kaggle is a famous website known for organizing many data science competition and for delivering dataset.
 
# 🔍 Exploratory Data Analysis (EDA)

## 📌 Dataset overview and class distribution
This dataset consist of 15 different types of leafs, the total size of the dataset is 20605 outof which 2475 are of pepper bell leafs, 2152 are of potato leafs and 15978 images are of tomato leafs.
  * Pepper__bell___Bacterial_spot -> 997 images.
    <div align="center">
       <img src="https://github.com/DeXtAr47-oss/Leaf-disease-classification/blob/d9e4f86325657c9e531297d773f936b482b4e231/images/0b47ce18-7cfe-45e8-b21e-b83cb6282455___JR_B.Spot%203162.JPG" width="300"/>
    </div>

  * Pepper__bell___healthy -> 1478 images.
    ![](https://github.com/DeXtAr47-oss/Leaf-disease-classification/blob/e6647c09f2f7484c8b6f144058a417db9acaa410/images/0ba474dd-0cfd-4fd2-a58c-8e3d18dbe7c3___JR_HL%208395.JPG)
    
  * ![Potato___Early_blight -> 1000 images](images/0a0744dc-8486-4fbb-a44b-4d63e6db6197___RS_Early.B 7575.JPG)
  * ![Potato___healthy -> 152](images/00bce074-967b-4d50-967a-31fdaa35e688___RS_HL 0223.JPG)
  * ![Potato___Late_blight -> 1000](images/0b092cda-db8c-489d-8c46-23ac3835310d___RS_LB 4480.JPG)
  * ![Tomato_Bacterial_spot -> 2127](images/0ad88d7a-c14a-4ac9-8520-c11a0ade3a8f___UF.GRC_BS_Lab Leaf 0996.JPG)
  * ![Tomato_Early_blight -> 1000](images/0c221e01-a85e-4794-b45b-3a91e349a8d2___RS_Erly.B 6425.JPG)
  * ![Tomato_healthy -> 1591](images/2dee1571-ef6b-40ef-8c46-334e89aad3f1___RS_HL 1950.JPG)
  * ![Tomato_Late_blight -> 1909](images/0a4b3cde-c83a-4c83-b037-010369738152___RS_Late.B 6985.JPG)
  * ![Tomato_Leaf_Mold -> 952](images/0db4cbf4-fa94-42c8-8bf5-90114281c569___Crnl_L.Mold 8681.JPG)
  * ![Tomato_Septoria_leaf_spot -> 1771](images/0b36c5a2-6c9f-40d9-af4b-d4b0e66997da___Matt.S_CG 6653.JPG)
  * ![Tomato_Spider_mites_Two_spotted_spider_mite -> 1676](images/0b5cda10-da2f-4647-b159-69647b42212f___Com.G_SpM_FL 1784.JPG)
  * ![Tomato__Target_Spot -> 1404](images/0c4c7140-1059-4e3a-a6e2-15c4bdd46743___Com.G_TgS_FL 8142.JPG)
  * ![Tomato__Tomato_mosaic_virus -> 373](images/0dae2780-43e7-40ac-ae45-95e5318c8f32___PSU_CG 2290.JPG)
  * ![Tomato__Tomato_YellowLeaf__Curl_Virus -> 3202](images/0a14b65b-2e45-4bed-be45-c482a40a4f7c___UF.GRC_YLCV_Lab 02090.JPG)
    
