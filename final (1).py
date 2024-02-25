################################
###### Final Home Exercise #####
################################

################################
# Student ID: 315589937
# First and Last Names: Aviv Weidenfeld
################################

# In this exercise you should implement a classification pipeline which aims at predicting the amount of hours
# a worker will be absent from work based on the worker characteristics and the work day missed.
# Download the dataset from the course website, which is provided as a .csv file. The target label is 'TimeOff'.
# You are free to use any library functions from numpy, pandas and sklearn, etc...
#
# You should implement the body of the functions below. The main two points of entry to your code are DataPreprocessor class and
# the train_model function. In the '__main__' section you are provided with an example of how your submission will be evaluated. 
# You are free to change the body of the functions and classes as you like - as long as it adheres to the provided input & output structure.
# In all methods and functions the input structure and the required returned variables are explicitly stated.
# Note that in order to evaluate the generalization error, you'll need to run cross validation as we demonstrated in class,
# However!!! In the final submission your file needs to contain only the methods of DataPreprocessor and the train_model function.
# Tip: You are encouraged to run gridsearch to find the best model and hyperparameters as demonstrated in class.
#
# To make things clear: you need to experiment with the preprocessing stage and the final model that will be used to fit. To get the
# sense of how your model performs, you'll need to apply the CV approach and, quite possibly, do a grid search of the meta parameters. 
# In the end, when you think that you've achieved your best, you should make a clean - and runnable!!! - version of your insights,
# which must adhere to the api provided below. In the evaluation stage, your code will be run on the entire train data,
# and then run once on the test data.
#
# You are expected to get results between 50% and 100% accuracy on the test set.
# Of course, the test set is not provided to you. However, as previously mentioned, running cross validation
# (with enough folds) will give you a good estimation of the accuracy.
#
# Important: obtaining accuracy less than 60%, will grant you 65 points for this exercise.
# Obtaining accuracy score above 60% will grant you 75 points minimum, however, your final score
# will be according to the distribution of all submissions. Therefore, due to the competition nature of this exercise, 
# you may use any method or library that will grant you the highest score, even if not learned in class.
#
# Identical or equivalent submissions will give rise to a suspicion of plagiarism.
#
# In addition to stating your names and ID numbers in the body of this file, name the file as follows:
# ex4_FirstName_LastName.py



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin



def load_dataset(train_csv_path):
    data = pd.read_csv(train_csv_path, sep=',')
    return data


class DataPreprocessor(object):

    """ 
    This class is a mandatory API.

    The purpose of this class is to unify data preprocessing step between the training and the testing stages. 
    This may include, but not limited to, the following transformations:
    1. Filling missing (NA / nan) values
    2. Dropping non-descriptive columns
    3 ...

    The test data is unavailable when building the ML pipeline, thus it is necessary to determine the 
    preprocessing steps and values on the train set and apply them on the test set.


    *** Mandatory structure ***
    The ***fields*** are ***not*** mandatory
    The ***methods***  - "fit" and "transform" - are ***required***.

    You're more than welcome to use sklearn.pipeline for the "heavy lifting" of the preprocessing tasks, but it is not an obligation. 
    Any class that implements the methods "fit" and "transform", with the required inputs & outputs will be accepted.
    Even if "fit" performs nothing at all.
    """

    def __init__(self):
        self._resident_distance_regressor = None
        self._weight_regressor = None
        self._season_classifier = None
        self._height = None
        self._Service_time = None
        self._time_off_mapper = self.TimeOff_maper()
        self._age_mapper = self.get_age_group_mapper()
        self._yes_no_mapper = self.get_drinker_or_smoker_mapper()
        self._age_group_classifier = None
        self._son_regressor = None
        self._pet_regressor = None


    def data_visualing(self, df):
        """
        Visualizes data characteristics using heatmaps.

        Parameters:
        - df (DataFrame): The input DataFrame containing the data to visualize.

        This function creates a figure with two subplots:
        1. Correlation Heatmap: Shows the correlation between different features in the DataFrame.
        2. Missing Values Heatmap: Visualizes the presence of missing values in the DataFrame.

        The Correlation Heatmap helps in understanding the relationships between different variables in the dataset. 
        It uses color intensity to represent the strength and direction of correlation between variables.
        Annotating the heatmap with correlation values provides additional insight into the relationships.

        The Missing Values Heatmap is useful for identifying the presence and patterns of missing data in the dataset. 
        It helps in determining if missing values are randomly distributed or if there are systematic patterns, 
        which can inform data preprocessing strategies such as imputation or exclusion.

        Returns:
        None
        """
        # print age group uniqu
        print(df['Smoker'].unique())

        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Plot correlation heatmap
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=axes[0])
        axes[0].set_title('Correlation Heatmap')

        # Plot missing values heatmap
        sns.heatmap(df.isnull(), ax=axes[1])
        axes[1].set_title('Missing Values Heatmap')

        # Show the plot
        plt.tight_layout()
        plt.show(block=True)
        print(df.info())
        print('\n')
        print(df.describe())


    def create_linear_regressor(self, X, y):
        # Prepare the training data
        X = X.fillna(X.mean())
        df = pd.concat([X, y], axis=1)
        df_non_missing = df.dropna()

        if df_non_missing.empty:
            raise ValueError("No non-missing values in the input data")

        X_train = df_non_missing.drop(columns=y.name)  # Drop the target column from X
        y_train = df_non_missing[y.name]

        # Train the linear regression model
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)

        # Assign the trained regressor as an attribute
        return regressor
    

    def create_SVM_classifier(self, X, y, kernel='linear'):
        # Fill missing values in X with the median for each column
        X_filled = X.fillna(X.mean())

        # Ensure that y is aligned with X
        df = pd.concat([X_filled, y], axis=1)

        # Drop rows with missing values in y (if any)
        df_non_missing = df.dropna(subset=[y.name])

        if df_non_missing.empty:
            raise ValueError("No non-missing values in the input data")

        X_train = df_non_missing.drop(columns=y.name)  # Drop the target column from X
        y_train = df_non_missing[y.name]

        # Train the SVM classifier
        svm_classifier = SVC(kernel=kernel, decision_function_shape='ovr')  # One-vs-rest strategy
        svm_classifier.fit(X_train, y_train)

        # Assign the trained classifier as an attribute
        return svm_classifier


    def fill_missing_values_classifier(self, df, feature_columns, target_column, classifier):
        # Predict missing 'Season' values using the trained SVM classifier
        df_missing = df[df[target_column].isnull()]
        if not df_missing.empty and classifier is not None:
            X_missing = df_missing[feature_columns]
            predicted_values = classifier.predict(X_missing)
            df.loc[df[target_column].isnull(), target_column] = predicted_values
        
        return df
    

    def fill_missing_values_regression(self, df, feature_columns, target_column, regressor):
        # Predict missing values using the trained regressor
        df_missing = df[df[target_column].isnull()]
        if not df_missing.empty and regressor is not None:
            X_missing = df_missing[feature_columns]
            predicted_values = regressor.predict(X_missing)
            df.loc[df[target_column].isnull(), target_column] = predicted_values

        return df
    

    def get_age_group_mapper(self):
        # Define a mapping dictionary
        age_group_mapping = {
            'Middle Aged': 4,
            'Adult': 3,
            'Young Adult': 2,
            'Senior': 1
        }
        
        # Define a function for mapping age groups to numbers
        def map_age_group(age_group):
            return age_group_mapping.get(age_group, age_group)  # Return the mapped value or the original value if not found
    
        return map_age_group
    

    def get_drinker_or_smoker_mapper(self):
        # Define a mapping dictionary
        drinker_or_smoker_mapping = {
            'No': -1,
            'Yes': 1,
        }
        
        # Define a function for mapping age groups to numbers
        def map_yes_or_no(yes_no):
            return drinker_or_smoker_mapping.get(yes_no, yes_no)  # Return the mapped value or the original value if not found
    
        return map_yes_or_no
     
        
    def fit(self, features, labels):

        """
        Input:
        dataset_df: the training data loaded from the csv file as a dataframe containing only the features
        (not the target - see the main function).

        Output:
        None

        Functionality:
        Based on all the provided training data, this method learns with which values to fill the NA's, 
        how to scale the features, how to encode categorical variables etc.
        Handle the relevant columns and save the information needed to transform the fields in the instance state.

        """
        # Handling missing values
        #create regressors for the missing values that we found that have correlation
        self._Service_time = features['Service time'].mode().iloc[0]
        self._height = features['Height'].mode().iloc[0]
        self._age_group = features['Age Group'].map(self._age_mapper).mode().iloc[0]
        self._pet = features["Pet"].mean()
        self._son = features["Son"].median()

        # init regressors and classifiers
        self._resident_distance_regressor = self.create_linear_regressor(features[['Transportation expense', 'Height']], features['Residence Distance'])
        self._season_classifier = self.create_SVM_classifier(features['Month'], features['Season'], kernel='rbf') 
        self._weight_regressor = self.create_linear_regressor(features[['Height', 'Service time']], features['Weight'])


    def transform(self, df):

        """
        Input:
        df:  *any* data similarly structured to the train data (dataset_df input of "fit")

        Output: 
        A processed dataframe or ndarray containing only the input features (X).
        It should maintain the same row order as the input.
        Note that the labels vector (y) should not exist in the returned ndarray object or dataframe.

        Functionality:
        Based on the information learned in the "fit" method, apply the required transformations to the passed data (df)
        *** This method will be called exactly once during evaluation. See the main section for details ***

        """
        transformed_df = df
        # Apply the mapper functions
        transformed_df['Age Group'] = transformed_df['Age Group'].map(self._age_mapper)
        transformed_df['Smoker'] = transformed_df['Smoker'].map(self._yes_no_mapper)
        transformed_df['Drinker'] = transformed_df['Drinker'].map(self._yes_no_mapper)
        # Fill Nan with median value
        transformed_df['Smoker'] = transformed_df['Smoker'].fillna(1)
        transformed_df['Drinker'] = transformed_df['Drinker'].fillna(1)
        transformed_df['Service time'] = transformed_df['Service time'].fillna(self._Service_time)
        transformed_df['Height'] = transformed_df['Height'].fillna(self._height)
        transformed_df['Age Group'] = transformed_df['Age Group'].fillna(self._age_group)
        transformed_df['Pet'] = transformed_df['Pet'].fillna(self._pet)
        transformed_df['Son'] = transformed_df['Son'].fillna(self._son)

        # # Predict missing 'resident_distance' values using the trained regressor
        transformed_df = self.fill_missing_values_regression(transformed_df, ['Transportation expense', 'Height'], 'Residence Distance', self._resident_distance_regressor)
        transformed_df = self.fill_missing_values_regression(transformed_df, ['Height', 'Service time'], 'Weight', self._weight_regressor) 
        transformed_df = self.fill_missing_values_classifier(transformed_df, ['Month'], 'Season', self._season_classifier)

        categories_reason = [np.arange(0, 29)]  # Adjust this if your 'Reason' range is different
        encoder_reason = OneHotEncoder(categories=categories_reason, sparse=False)
        encoded_reason = encoder_reason.fit_transform(transformed_df[['Reason']])
        encoded_reason_df = pd.DataFrame(encoded_reason, columns=[f"Reason_{i}" for i in range(0, 29)], index=transformed_df.index)
        transformed_df = pd.concat([transformed_df.drop('Reason', axis=1), encoded_reason_df], axis=1)

        categories_season = [np.arange(1, 5)]  # Adjust this if your 'Month' range is different (e.g., [np.arange(1, 13)] for 1-12)
        encoder_season = OneHotEncoder(categories=categories_season, sparse=False)
        encoded_season = encoder_season.fit_transform(transformed_df[['Season']])
        encoded_season_df = pd.DataFrame(encoded_season, columns=[f"Season{i}" for i in range(1, 5)], index=transformed_df.index)  # Adjust the range if necessary
        transformed_df = pd.concat([transformed_df.drop('Season', axis=1), encoded_season_df], axis=1) 


        transformed_df['BMI'] = transformed_df['Weight'] / ((transformed_df['Height'] / 100) ** 2)

        # Drop columns
        transformed_df.drop(['Weight', 'Height', 'Education', 'ID', 'Month', 'Day'], axis=1, inplace=True)

        return transformed_df


    def TimeOff_maper(self):
        # Define a mapping dictionary
        time_off_mapping = {
            'Very High': 4,
            'High': 3,
            'Medium': 2,
            'Low': 1
        }
        
        # Define a function for mapping age groups to numbers
        def time_off_group(time_group):
            return time_off_mapping.get(time_group, time_group)  # Return the mapped value or the original value if not found
    
        return time_off_group


    def clean_data(self, df):
        df['TimeOff'] = df['TimeOff'].map(self._time_off_mapper)


class ModelSelector(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator=SVC()):
        self.estimator = estimator

    def fit(self, X, y=None, **kwargs):
        self.estimator.fit(X, y)
        return self

    def predict(self, X, y=None):
        return self.estimator.predict(X)

    def score(self, X, y):
        return self.estimator.score(X, y)


def train_model(processed_X, y):
    """
    Performs grid search to find the best combination of model, feature selection, and model parameters.
    Parameters:
    processed_X (DataFrame or ndarray): Feature dataset for training.
    y (Series or ndarray): Target labels for training.
    Returns:
    best_model: The model with the best parameters found.
    """
    # I tried a lot of model and the logistic regression give the best results
    # Define the pipeline steps
    pipe = Pipeline([
        ('feature_selection', RFECV(estimator=SVC(kernel="linear"))),
        ('model', ModelSelector())
    ])

    param_grid = {
        'feature_selection__estimator': [LogisticRegression(solver='liblinear')],
        'model__estimator': [LogisticRegression(solver='liblinear')],
        'model__estimator__C': [5, 10, 20, 40],
        'model__estimator__penalty': ['l1'],
        'model__estimator__fit_intercept': [True], 
        'model__estimator__max_iter': [80],  
        'model__estimator__tol': [0.01],  
        'model__estimator__random_state': [42],  
    }


    # Initialize GridSearchCV
    grid_search = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1)

    # Perform grid search
    grid_search.fit(processed_X, y)

    # Print best parameters
    print(f"Best Parameters: {grid_search.best_params_}")

    # Return the best model
    return grid_search.best_estimator_


if __name__ == '__main__':
    preprocessor = DataPreprocessor()
    train_csv_path = 'real_time_off_data_train.csv'
    train_dataset_df = load_dataset(train_csv_path)

    X_train = train_dataset_df.iloc[:, :-1]
    y_train = train_dataset_df['TimeOff']
    preprocessor.fit(X_train, y_train)
    model = train_model(preprocessor.transform(X_train), y_train)

    ### Evaluation Section ####
    test_csv_path = 'real_time_off_data_train.csv'
    # Obviously, this will be different during evaluation. For now, you can keep it to validate proper execution
    test_csv_path = train_csv_path
    test_dataset_df = load_dataset(test_csv_path)

    X_test = test_dataset_df.iloc[:, :-1]
    y_test = test_dataset_df['TimeOff']

    processed_X_test = preprocessor.transform(X_test)
    predictions = model.predict(processed_X_test)
    test_score = accuracy_score(y_test, predictions)

    predictions = model.predict(preprocessor.transform(X_train))
    train_score = accuracy_score(y_train, predictions)

    print("<==============================================>")
    print("Accuracy on test:", test_score)
    print('Accuracy on train:', train_score)