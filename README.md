This code is a supplement to the following publication:

Collin, Daniel; Shprits, Yuri; Hofmeister, Stefan J.; Bianco, Stefano; Gallego, Guillermo (2024): 
High-Speed Solar Wind Stream Prediction from Solar Images Using a Distribution Transformation. (In review)

It can be used to reproduce the results presented in the paper, or to reuse or further develop the methodology for other
purposes. It is published under MIT license. Copyright (c) Daniel Collin (2024).
In case of questions or bugs, please contact the author Daniel Collin at collin@gfz-potsdam.de.

The data, needed to reproduce the results, can be obtained from the following data publication:

Collin, Daniel; Shprits, Yuri; Hofmeister, Stefan J.; Bianco, Stefano; Gallego, Guillermo (2024): 
Solar Wind Speed Prediction from Coronal Holes. GFZ Data Services. https://doi.org/10.5880/GFZ.2.7.2024.001

To execute the code and reproduce the results, download the data and create a 'data' folder next to the 'code' folder,
using the following structure:

    - data
        - hyperparameters.ods
        - datasets
            - alpha.pickle
            - ml_data
                - 4x3.pickle
                - ... (machine learning datasets based on specific grid resolutions)
                - 10x10.pickle
            - enhancements
                - cme_list.pickle
                - hss_list.pickle
                - enhancement_list.pickle
        - segmentation_maps
            - curated
                - 2010-05.pickle
                - ...
                - 2019-12.pickle

Additionally, set up the virtual environment and activate it, using conda: 

conda env create -f environment.yml

Optionally, there is 'requirementx.txt' if pip is preferred.

Then, open the script main.py. It runs a cross-validation for the specified model configurations. The model can be 
configured in the top part of the script. Further explanations are given in the script as comments, and a detailed 
explanation of the methodology can be found in the paper publication.

The script is currently configured to compute the cross-validation results for the 4x3 grid model, optimized towards 
minimizing the timeline RMSE. To compute the results for the second model that is mainly used in the paper, 
set grid = '10x10' and target_metric = 'peak_rmse'. This will compute the results for the 10x10 grid model, 
optimized towards the HSS peak velocity RMSE.

Results are stored in an additional 'results' folder, containing the subfolders 'model_eval' and 'model_pred', 
the first one containing excel files summarizing the evaluation metrics, model coefficients and feature importance, 
and the second one containing pickle files storing computed data products, e.g., predictions and observations of the 
time series and HSSs.

An example of how to access and plot the predictions stored in model_pred is given in the script look_at_results.py. 
After running main.py and computing results, look_at_results.py can be executed and plots a section of the predicted 
time series. Modify this script to get the visualizations or evaluations needed.

To create new datasets, based on other grid resolutions, specify the grid resolution and hyperparameters in main.py. 
Then, the program will automatically compute a new dataset based on the downsampled and curated coronal hole 
segmentation maps and save the dataset in the data folder.

Copyright (c) Daniel Collin (2024).
