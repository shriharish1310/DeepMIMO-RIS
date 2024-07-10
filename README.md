# DeepMIMO-RIS
# Deep Learning Enhanced RIS Configuration for Urban Scenario
This is a MATLAB code package related to the following article: 

Ramana Srivats, Shri Harish, Aravindan SM, P Kasthuri, P Prakash[Deep Learning Enhanced RIS Configuration for Urban Scenario]

# Abstract of the Article

This work introduces a significant advancement in Reconfigurable Intelligent Surfaces (RIS) to optimize wireless systems. Our RIS architecture integrates sparse channel sensors and a deep learning model inspired by AlexNet, addressing the challenge of channel estimation efficiently. With most elements passive and only a few actively connected to the baseband, our approach requires just 4000 data points to achieve performance close to the upper bound, outperforming existing methods. Utilizing the DeepMIMO dataset enhances data efficiency, minimizing training overhead while achieving high accuracy and efficiency gains. Specifically, our method attains the maximum achievable data rate with a significantly reduced dataset size. Traditional methods required over 30,000 samples for similar performance, but our model achieves comparable results with only 4000 data points. This represents a significant advancement in data efficiency, reducing the required dataset size by more than 86%.

# Code Package Content
To generate the standaloneDataset refer to the steps mentioned in the word file DeepMIMO Dataset Generation.

The main script for generating the outputs  is named `Fig12_generator.m`. Two additional MATLAB function named `Main_fn.m` and `Main_fn_2` is called by the main script. Another additional MATLAB function named `UPA_codebook_generator.m` is called by the functions `Main_fn.m` and `Main_fn_2`.

The script adopts the first version of the publicly available parameterized [DeepMIMO dataset](https://www.deepmimo.net/versions/v2-matlab/) published for deep learning applications in mmWave and massive MIMO systems. The ['O1_28'](https://deepmimo.net/scenarios/o1-scenario/) scenario is adopted for this case.

**To reproduce the results, please follow these steps:**
1. Download all the files of this project. The `RayTracing Scenarios` folder has been kept empty due to storage constraints.
2. Download the ['O1_28'](https://deepmimo.net/scenarios/o1-scenario/) scenario which is used in this project and extract the files to the `RayTracing Scenarios` folder.
3. To get the output of the model used in this project (AlexNet Model) for documentation refer to (https://pytorch.org/vision/main/models/alexnet), Run the file name `Fig12_generator.m` in MATLAB, make sure to include all the folders and subfolders in the source directory to the path and the script will sequentially execute the following tasks:
    1. Generate the inputs and outputs of the deep learning model.
    2. Build, train, and test the deep learning model.
    3. Process the deep learning outputs and generate the performance results.
4. To get the output of the base Model, Run the file named `Fig12_generator.m` and modify the line 49 replace with the code "[Rate_DL,Rate_OPT]=Main_fn_2(L,My_ar(rr),Mz_ar(rr),M_bar,K_DL,Pt,kbeams,Training_Size);" in MATLAB and the script will execute the base code.
 
The code was tested on MATLAB R2023b.
