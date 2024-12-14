import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from tkinter import PhotoImage
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class HypothesisTestingApp:
    def __init__(self, root, data):
        self.root = root
        self.root.title("Hypothesis Testing")
        self.root.state('zoomed')

        # Load and display the animated GIF image
        self.gif_frames = []
        self.load_gif_frames("C:/2nd year/PA PROJECT/heart5.gif")  # Replace "your_gif_image.gif" with the path to your GIF image
        self.current_frame_index = 0
        self.background_label = tk.Label(self.root)
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)
        self.animate_gif()

        # Initialize variables
        self.data = data
        self.regression_result = None

        # Create custom style for buttons
        style = ttk.Style()
        style.configure('Custom.TButton', font=('Times New Roman', 14), padding=(10, 5), background='light blue', foreground='black')

        # T-test button
        t_test_button = ttk.Button(root, text="Perform T-Test", command=self.perform_t_test, style='Custom.TButton')
        t_test_button.place(relx=0.1, rely=0.05, relwidth=0.2, relheight=0.08)

        # Z-test button
        z_test_button = ttk.Button(root, text="Perform Z-Test", command=self.perform_z_test, style='Custom.TButton')
        z_test_button.place(relx=0.4, rely=0.05, relwidth=0.2, relheight=0.08)

        # ANOVA-like Test button
        anova_test_button = ttk.Button(root, text="Perform ANOVA-like Test", command=self.perform_anova_test, style='Custom.TButton')
        anova_test_button.place(relx=0.7, rely=0.05, relwidth=0.2, relheight=0.08)

    def load_gif_frames(self, gif_path):
        gif = Image.open(gif_path)
        try:
            while True:
                gif_frame = gif.copy()
                self.gif_frames.append(gif_frame)
                gif.seek(len(self.gif_frames))  # Move to the next frame
        except EOFError:
            pass

    def animate_gif(self):
        if self.gif_frames:
            self.current_frame_index += 1
            if self.current_frame_index >= len(self.gif_frames):
                self.current_frame_index = 0
            frame = self.gif_frames[self.current_frame_index]
            self.gif_image = ImageTk.PhotoImage(frame)
            self.background_label.configure(image=self.gif_image)
            self.background_label.image = self.gif_image
            self.root.after(100, self.animate_gif)



    def perform_multi_linear_regression_analysis(self):
        # Splitting the dataset into independent (X) and dependent (y) variables for multi-linear regression
        X = self.data[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']]  # Independent variables
        y = self.data['stroke']  # Dependent variable

        # Handling missing values by imputing with mean
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)

        # Splitting the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

        # Fitting multi-linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Store the regression result in self.regression_result
        self.regression_result = model

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

    def perform_t_test(self):
        # Check if regression analysis has been performed
        if self.regression_result is None:
            print("Regression analysis not performed. Performing now...")
            self.perform_multi_linear_regression_analysis()
            print("Regression analysis completed.")
            
        # Check again if regression analysis has been performed
        if self.regression_result is None:
            messagebox.showerror("Error", "Please provide a regression result.")
            return

        # Get the coefficients from the regression result
        coefficients = self.regression_result.coef_
        
        # Calculate residuals
        X = self.data[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']]
        y = self.data['stroke']
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)
        predictions = self.regression_result.predict(X_imputed)
        residuals = y - predictions

        # Calculate standard errors of the coefficients
        X_with_intercept = np.column_stack([np.ones(X_imputed.shape[0]), X_imputed])
        XtX_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
        mse = (residuals ** 2).mean()
        std_errors = np.sqrt(np.diagonal(mse * XtX_inv))

        # Choose the coefficient index for which you want to perform the T-test
        coefficient_of_interest_index = 0  # Change this index to the coefficient you want to test

        # Get the coefficient and standard error for the chosen coefficient
        coef_interest = coefficients[coefficient_of_interest_index]
        se_interest = std_errors[coefficient_of_interest_index + 1]  # Adjust for intercept term

        # Calculate the T-value and p-value
        t_value = coef_interest / se_interest
        p_value = 2 * (1 - stats.t.cdf(abs(t_value), df=len(X_imputed) - X_imputed.shape[1]))

        # Set the significance level
        significance_level = 0.05

        # Determine the inference
        if p_value <= significance_level:
            inference = "Reject the null hypothesis."
        else:
            inference = "Fail to reject the null hypothesis."

        # Format the result string
        t_test_result = f"T-Value: {t_value}\nP-Value: {p_value}\nInference: {inference}"

        # Show the T-Test plot
        self.plot_t_test(residuals, t_value, p_value)

        # Show the result in a messagebox
        messagebox.showinfo("T-Test Result", t_test_result)

    def plot_t_test(self, residuals, t_value, p_value, significance_level=0.05):
        # Plot the distribution of residuals
        plt.figure(figsize=(8, 6))
        plt.hist(residuals, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Residuals Distribution')
        plt.xlabel('Residuals')
        plt.ylabel('Density')

        # Plot critical values
        critical_value = stats.t.ppf(1 - significance_level / 2, len(residuals) - 2)
        plt.axvline(x=critical_value, color='red', linestyle='--', label=f'Critical Value: ±{critical_value:.2f}')
        plt.axvline(x=-critical_value, color='red', linestyle='--')

        # Plot the calculated t-value
        plt.axvline(x=t_value, color='green', linestyle='-', label=f'Calculated T-Value: {t_value:.2f}')

        plt.legend()
        plt.show()

    def perform_z_test(self):
        # Subset the data for males and females
        male_data = self.data[self.data['gender'] == 'Male']
        female_data = self.data[self.data['gender'] == 'Female']

        # Calculate the proportion of stroke occurrences for males and females
        male_stroke_proportion = male_data['stroke'].mean()
        female_stroke_proportion = female_data['stroke'].mean()

        # Sample sizes
        n_male = len(male_data)
        n_female = len(female_data)

        # Calculate the standard error of the difference in proportions
        se_diff = ((male_stroke_proportion * (1 - male_stroke_proportion)) / n_male + (female_stroke_proportion * (1 - female_stroke_proportion)) / n_female) ** 0.5

        # Calculate the z-score
        z_score = (male_stroke_proportion - female_stroke_proportion) / se_diff

        # Calculate the p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed test

        # Plot Z-Test
        self.plot_z_test(male_stroke_proportion, female_stroke_proportion, z_score, p_value)

        # Print results
        if p_value <= 0.05:
            inference = "Reject the null hypothesis. There is a significant difference in the proportion of stroke occurrences between males and females."
        else:
            inference = "Fail to reject the null hypothesis. There is no significant difference in the proportion of stroke occurrences between males and females."
        
        z_test_result = f"Z-Value: {z_score}\nP-Value: {p_value}\nInference: {inference}"
        messagebox.showinfo("Z-Test Result", z_test_result)

    def plot_z_test(self, male_stroke_proportion, female_stroke_proportion, z_score, p_value, significance_level=0.05):
        # Plot the distribution of stroke occurrences for males and females
        plt.figure(figsize=(8, 6))
        plt.bar(['Male', 'Female'], [male_stroke_proportion, female_stroke_proportion], color=['skyblue', 'pink'])
        plt.title('Stroke Occurrences by Gender')
        plt.xlabel('Gender')
        plt.ylabel('Proportion of Stroke Occurrences')

        # Plot critical values
        critical_value = stats.norm.ppf(1 - significance_level / 2)
        plt.axhline(y=critical_value, color='red', linestyle='--', label=f'Critical Value: {critical_value:.2f}')
        plt.axhline(y=-critical_value, color='red', linestyle='--')

        # Plot the calculated z-value
        plt.axhline(y=z_score, color='green', linestyle='-', label=f'Calculated Z-Value: {z_score:.2f}')

        plt.legend()
        plt.show()

    def perform_anova_test(self):
        # Prepare data for ANOVA test
        # For example, let's say you want to compare the average glucose level across different smoking statuses
        
        # Select relevant columns
        anova_data = self.data[['smoking_status', 'avg_glucose_level']]

        # Drop missing values
        anova_data = anova_data.dropna()

        # Perform ANOVA test
        groups = anova_data.groupby('smoking_status')['avg_glucose_level']
        f_value, p_value = stats.f_oneway(*[group for name, group in groups])

        # Set the significance level
        significance_level = 0.05

        # Determine the inference
        if p_value <= significance_level:
            inference = "Reject the null hypothesis. There is a significant difference in average glucose level among different smoking statuses."
        else:
            inference = "Fail to reject the null hypothesis. There is no significant difference in average glucose level among different smoking statuses."

        # Show the ANOVA-like Test plot
        self.plot_anova_test(anova_data)

        # Format the result string
        anova_test_result = f"F-Value: {f_value}\nP-Value: {p_value}\nInference: {inference}"

        # Show the result in a messagebox
        messagebox.showinfo("ANOVA-like Test Result", anova_test_result)

    def plot_anova_test(self, anova_data):
        # Plot boxplot or violin plot
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='smoking_status', y='avg_glucose_level', data=anova_data)
        plt.title('Average Glucose Level by Smoking Status')
        plt.xlabel('Smoking Status')
        plt.ylabel('Average Glucose Level')

        plt.show()

if __name__ == "__main__":
    data = pd.read_csv("C:/Users/saran/Downloads/healthcare-dataset-stroke-data (1).csv")  # Change this path to your CSV file
    root = tk.Tk()
    app = HypothesisTestingApp(root, data)
    root.mainloop()

       

