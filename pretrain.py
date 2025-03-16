import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("workout_fitness_tracker.csv")  # Update with your actual dataset

# ✅ Define categorical & numerical columns
categorical_cols = ["Gender", "Workout Type", "Workout Intensity", "Mood Before Workout", "Mood After Workout"]
numerical_cols = ["Age", "Height (cm)", "Weight (kg)", "Heart Rate (bpm)", "Steps Taken",
                  "Distance (km)", "Sleep Hours", "Water Intake (liters)", "Daily Calories Intake",
                  "Resting Heart Rate (bpm)", "VO2 Max", "Body Fat (%)"]

# ✅ Encode categorical columns
encoders = {}
for col in categorical_cols:
    encoders[col] = LabelEncoder()
    df[col] = encoders[col].fit_transform(df[col])

# Save encoders
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

# ✅ Scale numerical features (excluding "Calories Burned")
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# ✅ Define target variable (before dropping "Calories Burned")
df["Workout Efficiency"] = df["Calories Burned"].apply(lambda x: "Low" if x < 200 else ("Medium" if x < 400 else "High"))

# ✅ Encode target variable
efficiency_encoder = LabelEncoder()
df["Workout Efficiency"] = efficiency_encoder.fit_transform(df["Workout Efficiency"])

# Save efficiency encoder
with open("efficiency_encoder.pkl", "wb") as f:
    pickle.dump(efficiency_encoder, f)

# ✅ Drop unnecessary columns
df.drop(["User ID", "Calories Burned"], axis=1, inplace=True)

# ✅ Split dataset
X = df.drop("Workout Efficiency", axis=1)
y = df["Workout Efficiency"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train ML Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
with open("workout_model.pkl", "wb") as f:
    pickle.dump(model, f)