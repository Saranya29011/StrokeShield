import pandas as pd
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import os

# Step 1: Read the dataset
data = pd.read_csv("C:/Users/saran/Downloads/healthcare-dataset-stroke-data (1).csv")

# Step 2: Handle missing values by dropping rows with any NaN values
data.dropna(inplace=True)

# Step 3: Select features and target variable
X = data.drop(columns=['stroke'])  # Features (excluding 'stroke' column)
y = data['stroke']  # Target variable

# Step 4: Separate numeric and categorical columns
numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Step 5: Compute the correlation matrix for numeric features
correlation_matrix = data[numeric_cols].corr()

# Step 6: Preprocessing for numeric features
numeric_transformer = StandardScaler()

# Step 7: Preprocessing for categorical features
categorical_transformer = OneHotEncoder(drop='first')

# Step 8: Bundle preprocessing for numeric and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Step 9: Preprocess the data
X_preprocessed = preprocessor.fit_transform(X)

# Step 10: Retrieve feature names after preprocessing
preprocessor.fit(X)
feature_names = numeric_cols.tolist() + preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols).tolist()

# Step 11: Perform Principal Component Analysis (PCA)
pca = PCA(n_components=2)  # Specify the number of principal components to keep
pca_result = pca.fit_transform(X_preprocessed)
pca_components = pca.components_

# Step 12: Perform Factor Analysis (FA)
fa = FactorAnalysis(n_components=2)  # Specify the number of factors to extract
fa_result = fa.fit_transform(X_preprocessed)
fa_components = fa.components_

# Step 13: Create DataFrames for PCA and FA loadings
pca_loadings = pd.DataFrame(pca_components.T, index=feature_names, columns=['Principal Component 1', 'Principal Component 2'])
fa_loadings = pd.DataFrame(fa_components.T, index=feature_names, columns=['Factor 1', 'Factor 2'])



# GUI Application
class PCA_FA_GUI():
    def __init__(self, root):
        self.root = root
        self.root.title("PCA and Factor Analysis")
        self.create_widgets()

    def create_widgets(self):
        # Frame for buttons on the left
        self.button_frame = ttk.Frame(self.root)
        self.button_frame.pack(side=tk.LEFT, fill=tk.Y, pady=10)

        # Create custom style for buttons
        style = ttk.Style()
        style.configure('Custom.TButton', font=('Times New Roman', 14), padding=(10, 5), background='AliceBlue', foreground='black')

        # PCA Loadings Button
        self.pca_button = ttk.Button(self.button_frame, text="SHOW PCA LOADINGS", command=self.show_pca_loadings, style='Custom.TButton')
        self.pca_button.pack(side=tk.TOP, pady=10)

        # FA Loadings Button
        self.fa_button = ttk.Button(self.button_frame, text="SHOW FA LOADINGS", command=self.show_fa_loadings, style='Custom.TButton')
        self.fa_button.pack(side=tk.TOP, pady=10)
        # Load the GIF animation
        gif_path ="C:/2nd year/PA PROJECT/GIF1.gif"   # Replace this with the actual path to your GIF file
        self.gif_label = tk.Label(self.button_frame)
        self.gif_label.pack(side=tk.TOP, pady=10)

        self.load_animation_frames(gif_path)

    def load_animation_frames(self, gif_path):
        # Load the GIF animation frames
        self.animation_frames = []
        gif = Image.open(gif_path)
        try:
            while True:
                self.animation_frames.append(ImageTk.PhotoImage(gif.copy()))
                gif.seek(gif.tell() + 1)
        except EOFError:
            pass

        # Start the animation
        self.animate_gif()

    def animate_gif(self):
        gif_idx = 0
        def update_gif():
            nonlocal gif_idx
            gif_idx = (gif_idx + 1) % len(self.animation_frames)
            self.gif_label.config(image=self.animation_frames[gif_idx])
            self.root.after(100, update_gif)  # Change the delay (in milliseconds) for frame change
        update_gif()

    def show_pca_loadings(self):
        self.create_result_window(pca_loadings, "PCA Loadings", pca.explained_variance_ratio_, pca_result, 'Principal Component 1', 'Principal Component 2', "PCA Result")

    def show_fa_loadings(self):
        self.create_result_window(fa_loadings, "Factor Analysis Loadings", correlation_matrix, fa_result, 'Factor 1', 'Factor 2', "Factor Analysis Result")

    def create_result_window(self, loadings_df, title, extra_info, result, xlabel, ylabel, plot_title):
        new_window = tk.Toplevel(self.root)
        new_window.title(title)
        new_window.state('zoomed')
        
        # Load the background image
        background_image = Image.open("C:/2nd year/PA PROJECT/img2.jpg")
        
        # Resize the background image to fit the window size
        window_width = self.root.winfo_screenwidth()
        window_height = self.root.winfo_screenheight()
        background_image = background_image.resize((window_width, window_height), Image.LANCZOS)
        background_photo = ImageTk.PhotoImage(background_image)

        # Create a canvas and set the background image
        canvas = tk.Canvas(new_window, width=window_width, height=window_height)
        canvas.pack(fill='both', expand=True)
        canvas.create_image(0, 0, image=background_photo, anchor='nw')

        # Keep a reference to the image to prevent garbage collection
        new_window.background_photo = background_photo

        text_box = tk.Text(new_window, height=20, width=100)
        canvas.create_window(50, 50, anchor='nw', window=text_box)
        
        if title == "PCA Loadings":
            text_box.insert(tk.END, f"{title}:\n")
            text_box.insert(tk.END, loadings_df.to_string() + '\n\n')
            text_box.insert(tk.END, "Explained variance ratio for each component:\n")
            for i, variance in enumerate(extra_info):
                text_box.insert(tk.END, f"Principal Component {i+1}: {variance:.4f}\n")
        else:
            text_box.insert(tk.END, f"{title}:\n")
            text_box.insert(tk.END, loadings_df.to_string() + '\n\n')
            text_box.insert(tk.END, "Correlation Matrix of Original Numeric Features:\n")
            text_box.insert(tk.END, extra_info.to_string() + '\n\n')

        button_frame = ttk.Frame(new_window)
        canvas.create_window(50, 400, anchor='nw', window=button_frame)

        visualize_button = ttk.Button(button_frame, text="Visualization", command=lambda: self.plot_results(result, xlabel, ylabel, plot_title))
        visualize_button.pack(side=tk.LEFT, padx=10)

        inference_button = ttk.Button(button_frame, text="Inference", command=lambda: self.show_inference(title))
        inference_button.pack(side=tk.LEFT, padx=10)

    def plot_results(self, result, xlabel, ylabel, title):
        plot_window = tk.Toplevel(self.root)
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(result[:, 0], result[:, 1], c=y, cmap='viridis')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        fig.colorbar(scatter, ax=ax, label='Stroke')
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def show_inference(self, title):
        if title == "PCA Loadings":
            inference_text = (
                "The PCA results indicate that health-related features like age, hypertension, heart_disease, "
                "avg_glucose_level, and bmi are the main drivers of variance, capturing significant patterns related "
                "to stroke risk. Together, the first two principal components explain 38% of the total variance in the dataset."
            )
        else:
            inference_text = (
                "Factor 1 predominantly reflects occupational factors, while Factor 2 primarily captures health-related attributes like age, hypertension, and BMI, "
                "indicating distinct underlying structures contributing to stroke risk. This suggests a multifaceted approach to stroke risk assessment, considering both occupational and health factors."
            )
        messagebox.showinfo("Inference", inference_text)

# Create the main Tkinter window
root = tk.Tk()
app = PCA_FA_GUI(root)
root.state('zoomed')
root.mainloop()
