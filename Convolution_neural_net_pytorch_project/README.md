# Deep Into CNN

#### Deep into CNN project was offered by Programming Club, IIT Kanpur and has the following main goals - 
- Introduce different types of Neural networks.  
    - MLP
    - CNN
- Why do we need different Networks?:
    - Ineffectiveness of Existing Ones, New Needs0
    - Solution Architecture
    - Effectiveness of Proposed Network , Performance Comparison
- 2 Competitions
    - One on Numerical Data and other on Image Data
- 1 Paper Implementation (SOTA Model) - ( I chose Xception)
- All the models in this projects were made in pytorch

#### This repository has the following directories/ files -



- **Hackathon 2** :  
    - This competition was based on the following dataset - [https://www.kaggle.com/gpiosenka/275-bird-species](https://www.kaggle.com/gpiosenka/100-bird-species)
    - It had high quality RGB images of 275 species of birds.
    - The dataset was complex and required a deep model to process it. Therefore, I chose Xception Model.
    - After about 32 epochs, **I was able to get an accuracy of about 90 %** !!  which is pretty high for RGB dataset.
    - We found a huge increase in accuracy when we compared it with hackathon 1. 
- **Xception Paper Implementation**:
    - The research paper for Xception model can be found [here](https://openaccess.thecvf.com/content_cvpr_2017/papers/Chollet_Xception_Deep_Learning_CVPR_2017_paper.pdf)
    - This repository contains the implemention of Xception paper ( I used cifar 10 data to implement it). 
    - The repo also contains 2 of the best performing model/parameters that were given in the output of the code. The models can be loaded for further training and  testing.
    - **I achieved an accuracy of about 85% !!** in minimal implementation of xception on cifar 10.

- **Hackathon 1** :
    - This competition was conducted in the early part of the project and was based on data classifications with neural networks.  
    - It was conducted on Kaggle and the best performer was able to achieve the error of 1.74 and I achieved 1.77 (-> [Tabular Playground Series - Jun 2021](https://www.kaggle.com/c/tabular-playground-series-jun-2021) <- ).
    - Also, this competition was used as a benchmark to understand how the accuracies could have been increased with CNN.

- **Resource_Practice** :
    - It contains the practice materials that helped to achieve the goals of this project.  
    - The practice material contains exercises on pytorch, simple neural networks, CNN models, GAN's and data cleaning and processing.  
