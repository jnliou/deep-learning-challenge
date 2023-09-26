# Alphabet Soup Charity Deep Learning Challenge

![Extra/image.avif](Extra/MIT-Neural-Networks-SL.gif)

## Background

The nonprofit foundation Alphabet Soup aims to make informed decisions about funding applicants with the highest chances of success. To achieve this goal, we are using machine learning and neural networks to build a binary classifier that predicts whether organizations funded by Alphabet Soup will be successful in their ventures.

We have received a dataset containing information about over 34,000 organizations that have previously received funding from Alphabet Soup. The dataset includes columns capturing metadata about each organization, such as application type, affiliation, classification, use case for funding, income classification, and more.

## Step 1: Preprocess the Data

In this step, we prepared the dataset for building and training the neural network model:

- Identified the target variable(s) for the model.
- Identified the feature(s) for the model.
- Dropped the EIN and NAME columns, which are identification columns.
- Determined the number of unique values for each column.
- Binned "rare" categorical variables together in a new value, "Other," for columns with more than 10 unique values.
- Used one-hot encoding (pd.get_dummies()) to encode categorical variables.
- Split the data into features (X) and target (y).
- Scaled the features using StandardScaler.
- Split the preprocessed data into training and testing datasets.

## Step 2: Compile, Train, and Evaluate the Model

In this step, we designed, compiled, trained, and evaluated a neural network model for binary classification. Key details include:

- Creation of a neural network model with an appropriate number of input features, hidden layers, and nodes.
- Addition of appropriate activation functions to the hidden layers.
- Compilation and training of the model.
- Implementation of a callback to save model weights every five epochs.
- Evaluation of the model using test data to calculate loss and accuracy.
- Saving the model to an HDF5 file named "AlphabetSoupCharity.h5."

## Step 3: Optimize the Model

To achieve a target predictive accuracy higher than 75%, we optimized the model using various techniques:

- Experimented with adding more neurons, hidden layers, and different activation functions.
- Tuned the number of epochs in the training regimen.

The optimized model results were saved in a new HDF5 file named "AlphabetSoupCharity_Optimization_model1.h5."

## Step 4: Report on the Neural Network Model

The report summarizes the performance and outcomes of the deep learning model:

### Data Preprocessing

- Target variable(s): IS_SUCCESSFUL
- Feature(s): All columns except EIN and NAME.
- Removed variable(s): EIN and NAME.

### Compiling, Training, and Evaluating the Model

- Number of neurons, layers, and activation functions: Varying configurations were explored in the optimization process.
- Target model performance: The target was to achieve an accuracy higher than 75%.
- Steps taken to increase model performance: Adjustments to data preprocessing, neural network architecture, and hyperparameters.

### Summary

The deep learning models underwent multiple iterations to optimize their performance. While the initial models achieved reasonable accuracy, the optimization process resulted in a model with improved accuracy.

**Recommendation for Future Work:**

To further improve the classification problem, we recommend exploring more complex neural network architectures, conducting extensive hyperparameter tuning, and potentially incorporating other machine learning techniques. Additionally, gathering more data or engineering additional features could contribute to better model performance.

## Step 5: Copy Files Into Your Repository

The project files are organized in the "deep-learning-challenge" repository as follows:

- **AlphabetSoupCharity_Optimization.ipynb**: The Jupyter Notebook containing the code for the optimized deep learning model.
- **AlphabetSoupCharity_model1.ipynb** and **AlphabetSoupCharity_model2.ipynb**: Jupyter Notebooks containing code for initial model iterations.
- **AlphabetSoupCharity_model1.h5** and **AlphabetSoupCharity_model2.h5**: The HDF5 file containing the trained deep learning model.
- **AlphabetSoupCharity_Optimization.h5**: The HDF5 file containing the optimized deep learning model.
- **Dataset**: https://static.bc-edx.com/data/dl-1-2/m21/lms/starter/charity_data.csv

For the complete project and code files, please refer to the GitHub repository: [deep-learning-challenge](https://github.com/jnliou/deep-learning-challenge).

**NOTE** Although the project instructions suggested us to utilize Google Collab, I was having trouble with running the data as I kept getting an error saying I had utilized all the available RAM, thus I ran the code through VS Code and Jupyter Notebook. 

---

*This project was completed as part of the Alphabet Soup Charity Deep Learning Challenge.*