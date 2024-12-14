import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import subprocess
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve



class LinearRegressionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Regression Analysis")
        
        # Load background image
        self.background_image_path = "C:/2nd year/PA PROJECT/1.jpg" # Replace with your image file path
        self.background_image = Image.open(self.background_image_path)
        self.background_image = self.background_image.resize((root.winfo_screenwidth(), root.winfo_screenheight()), Image.LANCZOS)
        self.background_photo = ImageTk.PhotoImage(self.background_image)

        # Get screen width and height
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        
        # Create canvas for background image with screen size
        self.canvas = tk.Canvas(root, width=screen_width, height=screen_height)
        self.canvas.pack()
        
        # Create image on canvas
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.background_photo)

        # Add text to the canvas at the top
        self.canvas.create_text(screen_width // 2, 50, text="       STROKE      PREDICTION", font=("Times New Roman", 34), fill="black")



        
        # Create custom style for buttons
        style = ttk.Style()
        style.configure('Custom.TButton', font=('Times New Roman', 14), padding=(10, 5), background='AliceBlue', foreground='black')

        # Perform Regression button
        regression_button = ttk.Button(root, text="PERFORM REGRESSION", command=self.perform_regression, style='Custom.TButton')
        regression_button.place(relx=0.7, rely=0.1, relwidth=0.2, relheight=0.08)

        # Perform Hypothesis button
        hypothesis_button = ttk.Button(root, text="PERFORM HYPOTHESIS", command=self.perform_hypothesis, style='Custom.TButton')
        hypothesis_button.place(relx=0.7, rely=0.2, relwidth=0.2, relheight=0.08)

        # Perform Clustering button
        clustering_button = ttk.Button(root, text="PERFORM CLUSTERING", command=self.perform_clustering, style='Custom.TButton')
        clustering_button.place(relx=0.7, rely=0.3, relwidth=0.2, relheight=0.08)

        # Perform Prediction button
        prediction_button = ttk.Button(root, text="PERFORM PREDICTION", command=self.perform_prediction, style='Custom.TButton')
        prediction_button.place(relx=0.7, rely=0.4, relwidth=0.2, relheight=0.08)

        # Perform Factor Analysis button
        prediction_button = ttk.Button(root, text="PERFORM ANALYSIS", command=self.perform_analysis, style='Custom.TButton')
        prediction_button.place(relx=0.7, rely=0.5, relwidth=0.2, relheight=0.08)
                    
                    

        # Initialize variables
        self.data = pd.read_csv("C:/Users/saran/Downloads/healthcare-dataset-stroke-data (1).csv")  # Change this path to your CSV file
        self.result = None
    def perform_hypothesis(self):
        try:
        # Replace "path_to_another_python_file.py" with the actual path of your Python file
            result = subprocess.run(["python", "C:/2nd year/PA PROJECT/hypothesis_testing.py"], capture_output=True, text=True)
        
        # Display the output in a message box or print it
            print(result.stdout)
        except FileNotFoundError:
            messagebox.showerror("Error", "The Python file could not be found.")
        
    def perform_regression(self):
        # Create a new window for regression options
        self.regression_window = tk.Toplevel(self.root)
        self.regression_window.title("Choose Regression Model")

        # Load background image for regression window
        try:
            regression_background_image_path = "C:/2nd year/PA PROJECT/img3.jpg"  # Replace with your image file path
            regression_background_image = Image.open(regression_background_image_path)
            resized_background_image = regression_background_image.resize((self.root.winfo_screenwidth(), self.root.winfo_screenheight()), Image.LANCZOS)
            regression_background_photo = ImageTk.PhotoImage(resized_background_image)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load background image: {e}")
            self.regression_window.destroy()  # Close the window if image loading fails
            return

        # Create canvas for background image with screen size
        regression_canvas = tk.Canvas(self.regression_window, width=self.root.winfo_screenwidth(), height=self.root.winfo_screenheight())
        regression_canvas.pack()

        # Create image on canvas at (0, 0) coordinates
        regression_canvas.create_image(0, 0, anchor=tk.NW, image=regression_background_photo)

        # Ensure that the image is not garbage collected
        regression_canvas.image = regression_background_photo

        # Create custom style for buttons
        style = ttk.Style()
        style.configure('Custom.TButton', font=('Times New Roman', 14), padding=(8, 3), background='AliceBlue', foreground='black')

        # Logistic Regression button
        logistic_regression_button = ttk.Button(self.regression_window, text="LOGISTIC REGRESSION", command=self.perform_logistic_regression, style='Custom.TButton')
        logistic_regression_button.place(relx=0.1, rely=0.1, relwidth=0.2, relheight=0.08)

        # Polynomial Regression button
        polynomial_regression_button = ttk.Button(self.regression_window, text="POLYNOMIAL REGRESSION", command=self.perform_polynomial_regression, style='Custom.TButton')
        polynomial_regression_button.place(relx=0.1, rely=0.2, relwidth=0.2, relheight=0.08)

        # Multi-Linear Regression button
        multi_linear_regression_button = ttk.Button(self.regression_window, text="MULTILINEAR REGRESSION", command=self.perform_multi_linear_regression, style='Custom.TButton')
        multi_linear_regression_button.place(relx=0.1, rely=0.3, relwidth=0.2, relheight=0.08)

        # Create a frame on the right side for displaying the result
        result_frame = tk.Frame(self.regression_window, bg="white")
        result_frame.place(relx=0.5, rely=0.1, relwidth=0.4, relheight=0.8)

        # Create a label to display the result (initially empty)
        self.result_label = tk.Label(result_frame, text="", bg="white", justify="left", anchor="nw", padx=10, pady=10)
        self.result_label.pack(fill="both", expand=True)

        # Initialize variables to store the results of each regression analysis
        self.logistic_result = None
        self.polynomial_result = None
        self.multi_linear_result = None

    def perform_logistic_regression(self):
        # Perform logistic regression and get the result message
        self.logistic_result = self.perform_logistic_regression_analysis()
        print("Logistic Regression Result:", self.logistic_result)  # Add this line

        # Display the result message
        self.result_label.config(text=self.logistic_result)

        # Check if all three regressions are completed, then display the best model
        self.check_and_display_best_model()

        # Visualize logistic regression results
        self.visualize_logistic_regression()

    def visualize_logistic_regression(self):
        # Calculate mean BMI for each age group
        mean_bmi_by_age = self.data.groupby('age')['bmi'].mean()

        # Replace missing BMI values with mean BMI of corresponding age group
        self.data['bmi'] = self.data['bmi'].fillna(self.data['age'].map(mean_bmi_by_age))

        # Splitting the dataset into independent (X) and dependent (y) variables for logistic regression
        X = self.data[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']]  # Independent variables
        y = self.data['stroke']  # Dependent variable

        # Splitting the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fitting logistic regression model
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, model.predict(X_test))
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", cbar=False)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix for Logistic Regression')
        plt.show()

        # ROC curve
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='orange', label='ROC')
        plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.show()

    def perform_logistic_regression_analysis(self):
        # Calculate mean BMI for each age group
        mean_bmi_by_age = self.data.groupby('age')['bmi'].mean()

        # Replace missing BMI values with mean BMI of corresponding age group
        self.data['bmi'] = self.data['bmi'].fillna(self.data['age'].map(mean_bmi_by_age))

        # Storing mean BMI values in X_imputed
        X_imputed = mean_bmi_by_age.values.reshape(-1, 1)

        # Splitting the dataset into independent (X) and dependent (y) variables for logistic regression
        X = self.data[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']]  # Independent variables
        y = self.data['stroke']  # Dependent variable

        # Splitting the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fitting logistic regression model
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Get the accuracy on the test set
        accuracy = model.score(X_test, y_test)

        # Get the coefficients and intercept
        coefficients = model.coef_
        intercept = model.intercept_

        # Create the result message including accuracy, coefficients, and intercept
        message = f"Accuracy: {accuracy}\n\n"
        message += "Coefficients:\n"
        for i, coef in enumerate(coefficients[0]):
            message += f"    Coefficient {i+1}: {coef}\n"
        message += f"Intercept: {intercept[0]}\n"

        return message

    def perform_polynomial_regression(self):
        # Perform polynomial regression and get the result message
        self.polynomial_result = self.perform_polynomial_regression_analysis()

        # Display the result message
        self.result_label.config(text=self.polynomial_result)

        # Check if all three regressions are completed, then display the best model
        self.check_and_display_best_model()

        # Visualize polynomial regression results
        self.visualize_polynomial_regression()

    def visualize_polynomial_regression(self):
        # Calculate mean BMI for each age group
        mean_bmi_by_age = self.data.groupby('age')['bmi'].mean()

        # Replace missing BMI values with mean BMI of corresponding age group
        self.data['bmi'] = self.data['bmi'].fillna(self.data['age'].map(mean_bmi_by_age))

        # Splitting the dataset into independent (X) and dependent (y) variables for polynomial regression
        X = self.data[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']]  # Independent variables
        y = self.data['stroke']  # Dependent variable

        # Handling missing values by imputing with mean for other columns
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)

        # Splitting the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

        # Fitting polynomial regression model
        model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
        model.fit(X_train, y_train)

        # Predictions using the polynomial regression model
        y_pred = model.predict(X_test)

        # Scatter plot of actual vs predicted values
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values for Polynomial Regression')
        plt.show()

        # Residual plot
        residuals = y_test - model.predict(X_test)
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, residuals)
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot for Polynomial Regression')
        plt.axhline(y=0, color='r', linestyle='-')
        plt.show()

    def perform_polynomial_regression_analysis(self):
        # Calculate mean BMI for each age group
        mean_bmi_by_age = self.data.groupby('age')['bmi'].mean()

        # Replace missing BMI values with mean BMI of corresponding age group
        self.data['bmi'] = self.data['bmi'].fillna(self.data['age'].map(mean_bmi_by_age))

        # Splitting the dataset into independent (X) and dependent (y) variables for polynomial regression
        X = self.data[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']]  
        y = self.data['stroke']  
        # Handling missing values by imputing with mean for other columns
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)

        # Splitting the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

        # Fitting polynomial regression model
        model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
        model.fit(X_train, y_train)

        # Get the R-squared score on the test set
        r_squared = model.score(X_test, y_test)

        # Create a message box for displaying the information
        message = f"Polynomial Regression Summary:\n\n"
        message += f"R-Squared: {r_squared}\n\n"

        # Coefficients
        coefficients = model.named_steps['linearregression'].coef_
        message += "Coefficients:\n"
        for i, coef in enumerate(coefficients):
            message += f"    Coefficient {i+1}: {coef}\n"
        
        # Intercept
        intercept = model.named_steps['linearregression'].intercept_
        message += f"Intercept: {intercept}\n"

        return message

    def perform_multi_linear_regression(self):
        # Perform multi-linear regression and get the result message
        self.multi_linear_result = self.perform_multi_linear_regression_analysis()

        # Display the result message
        self.result_label.config(text=self.multi_linear_result)

        # Check if all three regressions are completed, then display the best model
        self.check_and_display_best_model()

        # Visualize polynomial regression results
        self.visualize_multi_linear_regression()

    def visualize_multi_linear_regression(self):
        # Calculate mean BMI for each age group
        mean_bmi_by_age = self.data.groupby('age')['bmi'].mean()

        # Replace missing BMI values with mean BMI of corresponding age group
        self.data['bmi'] = self.data['bmi'].fillna(self.data['age'].map(mean_bmi_by_age))

        # Splitting the dataset into independent (X) and dependent (y) variables for multi-linear regression
        X = self.data[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']]  # Independent variables
        y = self.data['stroke']  # Dependent variable

        # Handling missing values by imputing with mean for other columns
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)

        # Splitting the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

        # Fitting multi-linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions using the multi-linear regression model
        y_pred = model.predict(X_test)

        # Residual plot
        residuals = y_test - model.predict(X_test)
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, residuals)
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot for Multi-linear Regression')
        plt.axhline(y=0, color='r', linestyle='-')
        plt.show()

    def perform_multi_linear_regression_analysis(self):
        # Calculate mean BMI for each age group
        mean_bmi_by_age = self.data.groupby('age')['bmi'].mean()

        # Replace missing BMI values with mean BMI of corresponding age group
        self.data['bmi'] = self.data['bmi'].fillna(self.data['age'].map(mean_bmi_by_age))

        # Splitting the dataset into independent (X) and dependent (y) variables for multi-linear regression
        X = self.data[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']]  
        y = self.data['stroke']  

        # Handling missing values by imputing with mean for other columns
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)

        # Splitting the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

        # Fitting multi-linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Get the R-squared score on the test set
        r_squared = model.score(X_test, y_test)

        # Create a message box for displaying the information
        message = f"Multi-Linear Regression Summary:\n\n"
        message += f"R-Squared: {r_squared}\n\n"

        # Coefficients
        coefficients = model.coef_
        message += "Coefficients:\n"
        for i, coef in enumerate(coefficients):
            message += f"    Coefficient {i+1}: {coef}\n"
        
        # Intercept
        intercept = model.intercept_
        message += f"Intercept: {intercept}\n"

        return message

    def check_and_display_best_model(self):
        # Check if all three regressions are completed
        if self.logistic_result and self.polynomial_result and self.multi_linear_result:
            # Compare regression models
            best_model_name = self.compare_regression_models()
            # Display the best model name
            if best_model_name is not None:
                messagebox.showinfo("Best Regression Model", f"The best regression model is: {best_model_name}")

    def compare_regression_models(self):
        # Get the accuracy or R-squared of each regression model
        logistic_accuracy = self.get_accuracy_from_result(self.logistic_result)
        polynomial_r_squared = self.get_r_squared_from_result(self.polynomial_result)
        multi_linear_r_squared = self.get_r_squared_from_result(self.multi_linear_result)

        # Determine the best model based on the highest accuracy or R-squared value
        best_model_name = None
        if logistic_accuracy is not None:
            if polynomial_r_squared is not None:
                if multi_linear_r_squared is not None:
                    # Compare all three models
                    if logistic_accuracy > polynomial_r_squared and logistic_accuracy > multi_linear_r_squared:
                        best_model_name = "Logistic Regression"
                    elif polynomial_r_squared > multi_linear_r_squared:
                        best_model_name = "Polynomial Regression"
                    else:
                        best_model_name = "Multi-Linear Regression"
                else:
                    # Compare logistic regression and polynomial regression
                    if logistic_accuracy > polynomial_r_squared:
                        best_model_name = "Logistic Regression"
                    else:
                        best_model_name = "Polynomial Regression"
            elif multi_linear_r_squared is not None:
                # Compare logistic regression and multi-linear regression
                if logistic_accuracy > multi_linear_r_squared:
                    best_model_name = "Logistic Regression"
                else:
                    best_model_name = "Multi-Linear Regression"
            else:
                # Only logistic regression is available
                best_model_name = "Logistic Regression"
        elif polynomial_r_squared is not None:
            if multi_linear_r_squared is not None:
                # Compare polynomial regression and multi-linear regression
                if polynomial_r_squared > multi_linear_r_squared:
                    best_model_name = "Polynomial Regression"
                else:
                    best_model_name = "Multi-Linear Regression"
            else:
                # Only polynomial regression is available
                best_model_name = "Polynomial Regression"
        elif multi_linear_r_squared is not None:
            # Only multi-linear regression is available
            best_model_name = "Multi-Linear Regression"

        return best_model_name

    def get_accuracy_from_result(self, result):
        # Extract accuracy from the result message
        if result:
            lines = result.split('\n')
            for line in lines:
                if line.startswith("Accuracy:"):
                    accuracy = float(line.split(":")[1].strip())
                    return accuracy
        return None

    def get_r_squared_from_result(self, result):
        # Extract R-squared from the result message
        if result:
            lines = result.split('\n')
            for line in lines:
                if line.startswith("R-Squared:"):
                    r_squared = float(line.split(":")[1].strip())
                    return r_squared
        return None

    def perform_clustering(self):
        try:
        # Replace "path_to_another_python_file.py" with the actual path of your Python file
            result = subprocess.run(["python", "C:/2nd year/PA PROJECT/cluster rough.py"], capture_output=True, text=True)
        
        # Display the output in a message box or print it
            print(result.stdout)
        except FileNotFoundError:
            messagebox.showerror("Error", "The Python file could not be found.")

    def perform_prediction(self):
        try:
            result = subprocess.run(["python", "C:/2nd year/PA PROJECT/prediction for stroke.py"], capture_output=True, text=True)
        
        # Display the output in a message box or print it
            print(result.stdout)
        except FileNotFoundError:
            messagebox.showerror("Error", "The Python file could not be found.")

    def perform_analysis(self):
        try:
            result = subprocess.run(["python", "C:/2nd year/PA PROJECT/factor analysis.py"], capture_output=True, text=True)
        
        # Display the output in a message box or print it
            print(result.stdout)
        except FileNotFoundError:
            messagebox.showerror("Error", "The Python file could not be found.")

    

if __name__ == "__main__":
    root = tk.Tk()
    app = LinearRegressionApp(root)
    root.mainloop()
