import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2

def calculate_bmi(weight_kg, height_m):
    return weight_kg / (height_m ** 2)

def predict_stroke(user_data):
    glucose_level = user_data['glucose_level']
    bmi = calculate_bmi(user_data['weight'], user_data['height'])

    if glucose_level > 130 and (bmi >= 25 or bmi >= 30):
        return "Stroke will occur"
    else:
        return "No stroke will occur"

def get_user_data():
    user_data = {}
    user_data['gender'] = gender_var.get().lower()
    user_data['age'] = age_entry.get()
    user_data['hypertension'] = int(hypertension_var.get())
    user_data['heart_disease'] = int(heart_disease_var.get())
    user_data['ever_married'] = True if ever_married_var.get() == 1 else False
    user_data['work_type'] = work_type_var.get().lower()
    user_data['residence_type'] = residence_type_var.get().lower()
    user_data['glucose_level'] = float(glucose_level_entry.get())
    user_data['height'] = float(height_entry.get())
    user_data['weight'] = float(weight_entry.get())
    user_data['smoking_status'] = smoking_status_var.get().lower()
    return user_data

def show_prediction():
    user_data = get_user_data()
    prediction = predict_stroke(user_data)
    prediction_label.config(text=prediction, font=("Helvetica", 24), fg="blue")

# Function to update the frame
def update_frame():
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (canvas.winfo_width(), canvas.winfo_height()))
    
    # Convert OpenCV frame to PIL Image
    pil_img = Image.fromarray(frame)
    
    # Convert PIL Image to PhotoImage
    img = ImageTk.PhotoImage(pil_img)
    
    # Clear previous image and text
    canvas.delete("all")
    
    # Display the video frame
    canvas.create_image(0, 0, anchor=tk.NW, image=img)
    
    # Display the text
    canvas.create_text(screen_width // 2, screen_height // 10, text="STROKE PREDICTION APP", font=("Times New Roman", 24), fill="white")
    
    canvas.image = img
    root.after(10, update_frame)


# Create main window
root = tk.Tk()
root.title("Stroke Prediction")
root.state('zoomed')

# Get screen width and height
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Create a canvas
canvas = tk.Canvas(root, bg="white", width=screen_width, height=screen_height)
canvas.pack(fill="both", expand=True)

# Add heading directly on the canvas
canvas.create_text(screen_width // 2, screen_height // 10, text="Stroke Prediction Application", font=("Helvetica", 24), fill="black")

# Frame for form elements
frame = tk.Frame(root, bg="white", bd=5)
frame.place(relx=0.7, rely=0.55, anchor="center")


# Padding for labels and entries
pad_x = 20
pad_y = 10

# Load video
video_path = "C:/2nd year/PA PROJECT/pa1.mp4"
cap = cv2.VideoCapture(video_path)




# Gender
gender_label = tk.Label(frame, text="Gender:", font=("Helvetica", 14), fg="black", bg="white")
gender_label.grid(row=0, column=0, padx=pad_x, pady=pad_y, sticky="e")
gender_var = tk.StringVar()
gender_combobox = ttk.Combobox(frame, textvariable=gender_var, values=["Male", "Female"], font=("Helvetica", 14))
gender_combobox.grid(row=0, column=1, padx=pad_x, pady=pad_y, sticky="w")

# Age
age_label = tk.Label(frame, text="Age:", font=("Helvetica", 14), fg="black", bg="white")
age_label.grid(row=1, column=0, padx=pad_x, pady=pad_y, sticky="e")
age_entry = tk.Entry(frame, font=("Helvetica", 14))
age_entry.grid(row=1, column=1, padx=pad_x, pady=pad_y, sticky="w")

# Hypertension
hypertension_label = tk.Label(frame, text="Hypertension:", font=("Helvetica", 14), fg="black", bg="white")
hypertension_label.grid(row=2, column=0, padx=pad_x, pady=pad_y, sticky="e")
hypertension_var = tk.IntVar()
hypertension_checkbox = tk.Checkbutton(frame, variable=hypertension_var, onvalue=1, offvalue=0, bg="white")
hypertension_checkbox.grid(row=2, column=1, padx=pad_x, pady=pad_y, sticky="w")

# Heart Disease
heart_disease_label = tk.Label(frame, text="Heart Disease:", font=("Helvetica", 14), fg="black", bg="white")
heart_disease_label.grid(row=3, column=0, padx=pad_x, pady=pad_y, sticky="e")
heart_disease_var = tk.IntVar()
heart_disease_checkbox = tk.Checkbutton(frame, variable=heart_disease_var, onvalue=1, offvalue=0, bg="white")
heart_disease_checkbox.grid(row=3, column=1, padx=pad_x, pady=pad_y, sticky="w")

# Ever Married
ever_married_label = tk.Label(frame, text="Ever Married:", font=("Helvetica", 14), fg="black", bg="white")
ever_married_label.grid(row=4, column=0, padx=pad_x, pady=pad_y, sticky="e")
ever_married_var = tk.IntVar()
ever_married_checkbox = tk.Checkbutton(frame, variable=ever_married_var, onvalue=1, offvalue=0, bg="white")
ever_married_checkbox.grid(row=4, column=1, padx=pad_x, pady=pad_y, sticky="w")

# Work Type
work_type_label = tk.Label(frame, text="Work Type:", font=("Helvetica", 14), fg="black", bg="white")
work_type_label.grid(row=5, column=0, padx=pad_x, pady=pad_y, sticky="e")
work_type_var = tk.StringVar()
work_type_entry = tk.Entry(frame, textvariable=work_type_var, font=("Helvetica", 14))
work_type_entry.grid(row=5, column=1, padx=pad_x, pady=pad_y, sticky="w")

# Residence Type
residence_type_label = tk.Label(frame, text="Residence Type:", font=("Helvetica", 14), fg="black", bg="white")
residence_type_label.grid(row=6, column=0, padx=pad_x, pady=pad_y, sticky="e")
residence_type_var = tk.StringVar()
residence_type_entry = tk.Entry(frame, textvariable=residence_type_var, font=("Helvetica", 14))
residence_type_entry.grid(row=6, column=1, padx=pad_x, pady=pad_y, sticky="w")

# Average Glucose Level
glucose_level_label = tk.Label(frame, text="Average Glucose Level:", font=("Helvetica", 14), fg="black", bg="white")
glucose_level_label.grid(row=7, column=0, padx=pad_x, pady=pad_y, sticky="e")
glucose_level_entry = tk.Entry(frame, font=("Helvetica", 14))
glucose_level_entry.grid(row=7, column=1, padx=pad_x, pady=pad_y, sticky="w")

# Height
height_label = tk.Label(frame, text="Height (in meters):", font=("Helvetica", 14), fg="black", bg="white")
height_label.grid(row=8, column=0, padx=pad_x, pady=pad_y, sticky="e")
height_entry = tk.Entry(frame, font=("Helvetica", 14))
height_entry.grid(row=8, column=1, padx=pad_x, pady=pad_y, sticky="w")

# Weight
weight_label = tk.Label(frame, text="Weight (in kilograms):", font=("Helvetica", 14), fg="black", bg="white")
weight_label.grid(row=9, column=0, padx=pad_x, pady=pad_y, sticky="e")
weight_entry = tk.Entry(frame, font=("Helvetica", 14))
weight_entry.grid(row=9, column=1, padx=pad_x, pady=pad_y, sticky="w")

# Smoking Status
smoking_status_label = tk.Label(frame, text="Smoking Status:", font=("Helvetica", 14), fg="black", bg="white")
smoking_status_label.grid(row=10, column=0, padx=pad_x, pady=pad_y, sticky="e")
smoking_status_var = tk.StringVar()
smoking_status_combobox = ttk.Combobox(frame, textvariable=smoking_status_var, values=["Yes", "No"], font=("Helvetica", 14))
smoking_status_combobox.grid(row=10, column=1, padx=pad_x, pady=pad_y, sticky="w")

# Predict Button
predict_button = tk.Button(frame, text="Predict", command=show_prediction, font=("Helvetica", 14), bg="green", fg="white")
predict_button.grid(row=11, column=0, columnspan=2, pady=pad_y)

# Prediction Label
prediction_label = tk.Label(frame, text="", font=("Helvetica", 24), bg="white")
prediction_label.grid(row=12, column=0, columnspan=2, pady=pad_y)

# Start the video update
update_frame()

root.mainloop()
