import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from matplotlib.colors import ListedColormap
from sklearn import svm

# Function to upload data
def page_data_upload():
    st.title("Data Upload")
    st.write("Upload your CSV or Excel file.")
    
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.write("Preview of the uploaded data:")
        st.write(df.head())
        st.session_state['data'] = df
        st.success("File uploaded successfully!")

# Introduction Section
def introduction():
    st.title("Predictive Analysis of Breast Tumor Diagnosis")
    
    st.header("Problem Statement")
    col1, col2 = st.columns([2, 3])

    with col1:
        st.image("pms.jfif", caption='breast_cancer', width=200)
        st.image("ps3.jfif", caption='breast_cancer', width=200)

    with col2:
        st.write("""
    Breast cancer is the most common malignancy among women, accounting for nearly 1 in 3 cancers diagnosed among women in the United States, and it is the second leading cause of cancer death among women. Breast Cancer occurs as a result of abnormal growth of cells in the breast tissue, commonly referred to as a Tumor. A tumor does not mean cancer - tumors can be benign (not cancerous), pre-malignant (pre-cancerous), or malignant (cancerous). Tests such as MRI, mammogram, ultrasound, and biopsy are commonly used to diagnose breast cancer performed.
    """)


    st.header("Expected Outcome")
    
    st.write("""
    Given breast cancer results from breast fine-needle aspiration (FNA) test (a quick and simple procedure to perform, which removes some fluid or cells from a breast lesion or cyst (a lump, sore, or swelling) with a fine needle similar to a blood sample needle). Since this build a model that can classify a breast cancer tumor using two training classifications:
    
    - 1 = Malignant (Cancerous) - Present
    - 0 = Benign (Not Cancerous) - Absent
    """)

    st.header("Objectives of Data Exploration")
    st.image("ex1.webp", caption="", use_column_width=True)
    st.write("""
    Exploratory data analysis (EDA) is a very important step which takes place after feature engineering and acquiring data and it should be done before any modeling. This is because it is very important for a data scientist to be able to understand the nature of the data without making assumptions. The results of data exploration can be extremely useful in grasping the structure of the data, the distribution of the values, and the presence of extreme values and interrelationships within the data set.
    
    The purpose of EDA is to use summary statistics and visualizations to better understand data, find clues about the tendencies of the data, its quality and to formulate assumptions and the hypothesis of our analysis.
    """)

    st.header("Unimodal Data Visualizations")
    st.image("un1.webp", caption="", use_column_width=True)
    st.write("""
    One of the main goals of visualizing the data here is to observe which features are most helpful in predicting malignant or benign cancer. The other is to see general trends that may aid us in model selection and hyperparameter selection. Apply 3 techniques that you can use to understand each attribute of your dataset independently:
    
    - Histograms
    - Density Plots
    - Box and Whisker Plots
    """)

    st.header("Pre-Processing the Data")
    st.image("pr1.png", caption="", use_column_width=True)
    st.write("""
    Data preprocessing is a crucial step for any data analysis problem. It is often a very good idea to prepare your data in such way to best expose the structure of the problem to the machine learning algorithms that you intend to use. This involves a number of activities such as:
    
    - Assigning numerical values to categorical data
    - Handling missing values
    - Normalizing the features (so that features on small scales do not dominate when fitting a model to the data)
    
    In this section, we will use feature selection to reduce high-dimension data, feature extraction and transformation for dimensionality reduction.
    """)

    st.header("Predictive Model using Support Vector Machine (SVM)")
    st.image("svm.jfif", caption="", use_column_width=True)
    st.write("""
    Support vector machines (SVMs) learning algorithm will be used to build the predictive model. SVMs are one of the most popular classification algorithms, and have an elegant way of transforming nonlinear data so that one can use a linear algorithm to fit a linear model to the data (Cortes and Vapnik 1995).
    
    Kernelized support vector machines are powerful models and perform well on a variety of datasets.
    
    - SVMs allow for complex decision boundaries, even if the data has only a few features.
    - They work well on low-dimensional and high-dimensional data (i.e., few and many features), but don’t scale very well with the number of samples.
    - Running an SVM on data with up to 10,000 samples might work well, but working with datasets of size 100,000 or more can become challenging in terms of runtime and memory usage.
    - SVMs requires careful preprocessing of the data and tuning of the parameters. This is why, these days, most people instead use tree-based models such as random forests or gradient boosting (which require little or no preprocessing) in many applications.
    - SVM models are hard to inspect; it can be difficult to understand why a particular prediction was made, and it might be tricky to explain the model to a nonexpert.
    """)
    

    st.header("Model Accuracy: Receiver Operating Characteristic (ROC) Curve")
    
    st.write("""
    In statistical modeling and machine learning, a commonly-reported performance measure of model accuracy for binary classification problems is Area Under the Curve (AUC).
    
    To understand what information the ROC curve conveys, consider the so-called confusion matrix that essentially is a two-dimensional table where the classifier model is on one axis (vertical), and ground truth is on the other (horizontal) axis, as shown below. Either of these axes can take two values (as depicted)
    
    In an ROC curve, you plot “True Positive Rate” on the Y-axis and “False Positive Rate” on the X-axis, where the values “true positive”, “false negative”, “false positive”, and “true negative” are events (or their probabilities) as described above. The rates are defined according to the following:
    
    - True positive rate (or sensitivity): tpr = tp / (tp + fn)
    - False positive rate: fpr = fp / (fp + tn)
    - True negative rate (or specificity): tnr = tn / (fp + tn)
    """)

    st.header("Automate the ML Process using Pipelines")
    
    st.write("""
    There are standard workflows in a machine learning project that can be automated. In Python scikit-learn, Pipelines help to clearly define and automate these workflows.
    
    - Pipelines help overcome common problems like data leakage in your test harness.
    - Python scikit-learn provides a Pipeline utility to help automate machine learning workflows.
    - Pipelines work by allowing for a linear sequence of data transforms to be chained together culminating in a modeling process that can be evaluated.
    """)

    st.header("Summary")
    
    st.write("""
    Worked through a classification predictive modeling machine learning problem from end-to-end using Python. Specifically, the steps covered were:
    
    - Problem Definition (Breast Cancer data).
    - Loading the Dataset.
    - Analyze Data (same scale but different distributions of data).
    - Evaluate Algorithms (KNN looked good).
    - Evaluate Algorithms with Standardization (KNN and SVM looked good).
    - Algorithm Tuning (K=19 for KNN was good, SVM with an RBF kernel and C=100 was best).
    - Finalize Model (use all training data and confirm using validation dataset).
    """)

# Function to inspect data
def inspect_data(data):
    st.header("Inspecting the Data")
    
    st.subheader("Data Preview")
    st.write(data.head())

    st.subheader("Data Summary")
    st.write(data.describe())
    
    st.subheader("Data Information")
    st.write(data.info())
    
    st.subheader("Missing Values")
    st.write(data.isnull().sum())
    
    st.subheader("Distribution of Target Variable (Diagnosis)")
    st.write(data['diagnosis'].value_counts())
    
    st.subheader("Frequency of Cancer Diagnosis")
    sns.set_style("white")
    sns.set_context({"figure.figsize": (10, 8)})
    fig, ax = plt.subplots()
    sns.countplot(data['diagnosis'], label='Count', palette="Set3", ax=ax)
    st.pyplot(fig)

# Function for exploratory data analysis
def exploratory_data_analysis(data):
    st.header("Exploratory Data Analysis")
    
    # Ensure the data types are appropriate for plotting
    numeric_data = data.select_dtypes(include=[np.number])
    
    st.subheader("Feature Histograms")
    df_mean = numeric_data.iloc[:, 1:11]
    df_se = numeric_data.iloc[:, 11:21]
    df_worst = numeric_data.iloc[:, 21:]

    st.write("Histograms of Mean Features")
    hist_mean = df_mean.hist(bins=10, figsize=(15, 10), grid=False)
    st.pyplot(plt.gcf())

    st.write("Histograms of SE Features")
    hist_se = df_se.hist(bins=10, figsize=(15, 10), grid=False)
    st.pyplot(plt.gcf())

    st.write("Histograms of Worst Features")
    hist_worst = df_worst.hist(bins=10, figsize=(15, 10), grid=False)
    st.pyplot(plt.gcf())
    
    st.subheader("Density Plots")
    fig, axs = plt.subplots(ncols=3, nrows=10, figsize=(20, 50))
    index = 0
    axs = axs.flatten()
    for k, v in df_mean.items():
        if np.issubdtype(v.dtype, np.number):
            sns.kdeplot(v, ax=axs[index], shade=True, color='g')
        index += 1
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Box Plots")
    box_mean = df_mean.plot(kind='box', subplots=True, layout=(2, 5), sharex=False, sharey=False, figsize=(20, 10), grid=False)
    st.pyplot(plt.gcf())
    
    st.subheader("Correlation Matrix")
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_mean.corr(), annot=True, fmt='.2f', cmap='coolwarm')
    st.pyplot(plt.gcf())
      
    st.subheader("Pair Plot")
    g = sns.pairplot(data[[data.columns[1], data.columns[2], data.columns[3],
                           data.columns[4], data.columns[5], data.columns[6]]], hue='diagnosis')
    st.pyplot(g.fig)

# Function for preprocessing data
def preprocess_data(data):
    st.header("Preprocess the Data")
    
    st.write("Label Encoding of Diagnosis Column")
    data['diagnosis'] = LabelEncoder().fit_transform(data['diagnosis'])
    st.write(data.head())
    
    st.write("Checking for Missing Values")
    st.write(data.isnull().sum())
    
    st.write("Splitting the data into Train and Test Sets")
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']
    
    st.write("Applying StandardScaler to the Features")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    st.session_state['X'] = X_scaled
    st.session_state['y'] = y
    
    st.success("Data Preprocessing Completed Successfully!")

# Function for training the model
def train_model():
    st.header("Train and Evaluate the Model")
    
    X = st.session_state['X']
    y = st.session_state['y']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define a pipeline combining a standard scaler with an SVM classifier
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=2)),
        ('svc', SVC(probability=True))
    ])
    
    # Define the parameter grid for grid search
    param_grid = {
        'svc__C': [0.1, 1, 10, 100],
              'svc__gamma': [1, 0.1, 0.01, 0.001]
    }
    
    # Perform Grid Search to find the best parameters
    grid = GridSearchCV(pipeline, param_grid, cv=5, verbose=2, n_jobs=-1)
    grid.fit(X_train, y_train)
    
    # Output the best parameters found by Grid Search
    st.write("Best Parameters:", grid.best_params_)
    
    # Output the classification report and confusion matrix
    st.subheader("Classification Report")
    y_pred = grid.predict(X_test)
    st.write(classification_report(y_test, y_pred))
    
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    st.write(cm)
    
    # Output the ROC curve and AUC score
    st.subheader("ROC Curve and AUC")
    y_score = grid.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    st.pyplot(plt.gcf())
    
    # Output the cross-validation scores
    st.subheader("Cross Validation Scores")
    cv_scores = cross_val_score(grid, X, y, cv=KFold(n_splits=10, shuffle=True, random_state=42))
    st.write(cv_scores)

def decision_plot():
    st.header("Finalize Model: Decision Boundary Visualization")
    
    # Check if the necessary data is available
    if "data" not in st.session_state or "X" not in st.session_state or "y" not in st.session_state:
        st.warning("Please upload, preprocess the data, and train the model first.")
        return
    
    X_train = st.session_state['X']
    y_train = st.session_state['y']
    data = st.session_state['data']
    
    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    
    # Create an instance of SVM and fit the data
    clf = svm.SVC(kernel='linear', C=1.0)
    clf.fit(X_train, y_train)
    
    # Define the mesh grid
    h = .02
    x_min, x_max = data.iloc[:, 0].min() - 1, data.iloc[:, 0].max() + 1
    y_min, y_max = data.iloc[:, 1].min() - 1, data.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Preprocess the mesh grid data to match the number of features in the training data
    # Assuming the training data has 31 features
    X_mesh = np.c_[xx.ravel(), yy.ravel()]
    # Add dummy features to match the expected number of features
    for _ in range(29):
        X_mesh = np.c_[X_mesh, np.zeros_like(X_mesh[:, 0])]
    
    # Predict the labels for each point in the mesh grid
    Z = clf.predict(X_mesh)
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot the training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("SVM Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar(label='Class')
    st.pyplot(plt.gcf())


def main():
    st.sidebar.title("Navigation")
    pages = {
        "Introduction": introduction,
        "Data Upload": page_data_upload,
        "Inspecting Data": lambda: inspect_data(st.session_state.get('data')),
        "Exploratory Data Analysis": lambda: exploratory_data_analysis(st.session_state.get('data')),
        "Preprocess Data": lambda: preprocess_data(st.session_state.get('data')),
        "Train Model": train_model,
        "Finalize Model": decision_plot
    }
    choice = st.sidebar.radio("Go to", list(pages.keys()))
    
    if choice == "Introduction":
        introduction()
    elif choice == "Data Upload" and "data" not in st.session_state:
        page_data_upload()
    elif "data" not in st.session_state:
        st.sidebar.warning("Please upload data first.")
    else:
        pages[choice]()

if __name__ == "__main__":
    main()
