# =====================================
# SpaceX Data Analysis - Python Script
# =====================================

# 1️⃣ Import des librairies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")  # style Seaborn

# 2️⃣ Charger le dataset depuis l'URL
URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv"
df = pd.read_csv(URL)

print("Dataset chargé. Nombre de lignes :", len(df))
print(df.head())

# =====================================
# 3️⃣ TASK 1: Flight Number vs Launch Site
# =====================================
sns.catplot(x="FlightNumber", y="LaunchSite", hue="Class", data=df, aspect=5)
plt.xlabel("Flight Number", fontsize=14)
plt.ylabel("Launch Site", fontsize=14)
plt.title("Flight Number vs Launch Site", fontsize=16)
plt.show()

# =====================================
# 4️⃣ TASK 2: Payload Mass vs Launch Site
# =====================================
sns.catplot(x="PayloadMass", y="LaunchSite", hue="Class", data=df, aspect=5)
plt.xlabel("Payload Mass (kg)", fontsize=14)
plt.ylabel("Launch Site", fontsize=14)
plt.title("Payload Mass vs Launch Site", fontsize=16)
plt.show()

# =====================================
# 5️⃣ TASK 3: Success Rate by Orbit
# =====================================
orbit_success = df.groupby("Orbit")["Class"].mean().reset_index()
sns.barplot(x="Orbit", y="Class", data=orbit_success)
plt.ylabel("Average Success Rate", fontsize=14)
plt.xlabel("Orbit Type", fontsize=14)
plt.title("Success Rate by Orbit", fontsize=16)
plt.show()

# =====================================
# 6️⃣ TASK 4: Flight Number vs Orbit
# =====================================
sns.catplot(x="FlightNumber", y="Orbit", hue="Class", data=df, aspect=5)
plt.xlabel("Flight Number", fontsize=14)
plt.ylabel("Orbit Type", fontsize=14)
plt.title("Flight Number vs Orbit", fontsize=16)
plt.show()

# =====================================
# 7️⃣ TASK 5: Payload Mass vs Orbit
# =====================================
sns.catplot(x="PayloadMass", y="Orbit", hue="Class", data=df, aspect=5)
plt.xlabel("Payload Mass (kg)", fontsize=14)
plt.ylabel("Orbit Type", fontsize=14)
plt.title("Payload Mass vs Orbit", fontsize=16)
plt.show()

# =====================================
# 8️⃣ TASK 6: Yearly Launch Success Trend
# =====================================
df['Year'] = df['Date'].apply(lambda x: x.split("-")[0])
yearly_success = df.groupby('Year')['Class'].mean().reset_index()

plt.figure(figsize=(12,5))
sns.lineplot(x='Year', y='Class', data=yearly_success, marker='o')
plt.ylabel("Average Launch Success", fontsize=14)
plt.xlabel("Year", fontsize=14)
plt.title("Yearly Launch Success Trend", fontsize=16)
plt.show()

# =====================================
# 9️⃣ Feature Engineering - Selection des Features
# =====================================
features = df[['FlightNumber', 'PayloadMass', 'Orbit', 'LaunchSite', 'Flights', 
               'GridFins', 'Reused', 'Legs', 'LandingPad', 'Block', 'ReusedCount', 'Serial']]

print("Features sélectionnées :")
print(features.head())

# =====================================
# 10️⃣ TASK 7: One-Hot Encoding des colonnes catégorielles
# =====================================
categorical_cols = ['Orbit', 'LaunchSite', 'LandingPad', 'Serial']
features_one_hot = pd.get_dummies(features, columns=categorical_cols)
print("Features après One-Hot Encoding :")
print(features_one_hot.head())

# =====================================
# 11️⃣ TASK 8: Convertir toutes les colonnes en float64
# =====================================
features_one_hot = features_one_hot.astype('float64')
print("Types des colonnes après conversion :")
print(features_one_hot.dtypes)
