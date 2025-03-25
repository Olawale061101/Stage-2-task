# Stage-2-task
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# Load dataset
df = pd.read_csv("nhanes.csv")

# Display first few rows
print(df.head())

# Check for missing values
print(df.isna().sum())

# Mean 60-second pulse rate
df_clean = df.dropna(subset=['Pulse'])
mean_pulse = df_clean['Pulse'].mean()
print(f"Mean 60-second pulse rate (after dropping NA): {mean_pulse}")

# Diastolic Blood Pressure Range
df_clean = df.dropna(subset=['BPDia'])
min_bp = df_clean['BPDia'].min()
max_bp = df_clean['BPDia'].max()
print(f"Diastolic Blood Pressure Range (after dropping NA): {min_bp} to {max_bp}")

# Variance and Standard Deviation of Income
df_clean = df.dropna(subset=['Income'])
income_var = df_clean['Income'].var()
income_std = df_clean['Income'].std()
print(f"Variance of Income: {income_var}")
print(f"Standard Deviation of Income: {income_std}")

# Clean data for visualization
df_clean = df.dropna(subset=['Weight', 'Height', 'Gender', 'Diabetes', 'SmokingStatus'])
print(df_clean['Gender'].value_counts())
print(df_clean['Diabetes'].value_counts())
print(df_clean['SmokingStatus'].value_counts())

# Calculate BMI
df_clean['BMI'] = df_clean['Weight'] / (df_clean['Height']/100)**2

# Violin Plots of BMI by Category
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.violinplot(x='Gender', y='BMI', data=df_clean)
plt.title('BMI Distribution by Gender')
plt.subplot(1, 3, 2)
sns.violinplot(x='Diabetes', y='BMI', data=df_clean)
plt.title('BMI Distribution by Diabetes')
plt.subplot(1, 3, 3)
sns.violinplot(x='SmokingStatus', y='BMI', data=df_clean)
plt.title('BMI Distribution by Smoking Status')
plt.tight_layout()
plt.show()

# KDE Plots for Weight vs Height
# For Gender
g = sns.FacetGrid(df_clean, col='Gender', height=5, aspect=1.2)
g.map(sns.kdeplot, x='Height', y='Weight', fill=True, cmap='Blues')
g.set_titles('Weight vs Height: {col_name}')
g.set_axis_labels('Height (cm)', 'Weight (kg)')
plt.show()

# For Diabetes
g = sns.FacetGrid(df_clean, col='Diabetes', height=5, aspect=1.2)
g.map(sns.kdeplot, x='Height', y='Weight', fill=True, cmap='Oranges')
g.set_titles('Weight vs Height: {col_name}')
g.set_axis_labels('Height (cm)', 'Weight (kg)')
plt.show()

# For SmokingStatus
g = sns.FacetGrid(df_clean, col='SmokingStatus', height=5, aspect=1.2)
g.map(sns.kdeplot, x='Height', y='Weight', fill=True, cmap='Greens')
g.set_titles('Weight vs Height: {col_name}')
g.set_axis_labels('Height (cm)', 'Weight (kg)')
plt.show()

# Histograms (keeping your original code)
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
sns.histplot(df['BMI'], bins=30)
plt.title('BMI Distribution')
plt.subplot(2, 2, 2)
sns.histplot(df['Weight'], bins=30)
plt.title('Weight (kg) Distribution')
df['Weight_lbs'] = df['Weight'] * 2.2
plt.subplot(2, 2, 3)
sns.histplot(df['Weight_lbs'], bins=30)
plt.title('Weight (lbs) Distribution')
plt.subplot(2, 2, 4)
sns.histplot(df['Age'], bins=30)
plt.title('Age Distribution')
plt.tight_layout()
plt.show()

# T-Tests
df_ttest = df.dropna(subset=['Age', 'Gender', 'BMI', 'Diabetes', 'AlcoholYear', 'RelationshipStatus'])
print("Gender categories:")
print(df_ttest['Gender'].value_counts())
print("\nDiabetes categories:")
print(df_ttest['Diabetes'].value_counts())
print("\nRelationshipStatus categories:")
print(df_ttest['RelationshipStatus'].value_counts())

# Fix RelationshipStatus categories
df_ttest['RelationshipStatus'] = df_ttest['RelationshipStatus'].replace({'Married': 'Committed', 'LivePartner': 'Committed'})

# 1. T-test between Age and Gender
group_male_age = df_ttest[df_ttest['Gender'] == 'male']['Age']
group_female_age = df_ttest[df_ttest['Gender'] == 'female']['Age']
t_stat_age_gender, p_val_age_gender = ttest_ind(group_male_age, group_female_age)
print(f"\nT-statistic for Age and Gender: {t_stat_age_gender}")
print(f"P-value for Age and Gender: {p_val_age_gender}")
if p_val_age_gender < 0.05:
    print("There is a significant difference in age between males and females.")
else:
    print("There is no significant difference in age between males and females.")
print("Mean Age by Gender:")
print(df_ttest.groupby('Gender')['Age'].mean())

# 2. T-test between BMI and Diabetes
group_diabetic_bmi = df_ttest[df_ttest['Diabetes'] == 'Yes']['BMI']
group_non_diabetic_bmi = df_ttest[df_ttest['Diabetes'] == 'No']['BMI']
t_stat_bmi_diabetes, p_val_bmi_diabetes = ttest_ind(group_diabetic_bmi, group_non_diabetic_bmi)
print(f"\nT-statistic for BMI and Diabetes: {t_stat_bmi_diabetes}")
print(f"P-value for BMI and Diabetes: {p_val_bmi_diabetes}")
if p_val_bmi_diabetes < 0.05:
    print("There is a significant difference in BMI between individuals with and without diabetes.")
else:
    print("There is no significant difference in BMI between individuals with and without diabetes.")
print("Mean BMI by Diabetes:")
print(df_ttest.groupby('Diabetes')['BMI'].mean())

# 3. T-test between AlcoholYear and RelationshipStatus
group_single_alcohol = df_ttest[df_ttest['RelationshipStatus'] == 'Single']['AlcoholYear']
group_committed_alcohol = df_ttest[df_ttest['RelationshipStatus'] == 'Committed']['AlcoholYear']
t_stat_alcohol_relationship, p_val_alcohol_relationship = ttest_ind(group_single_alcohol, group_committed_alcohol)
print(f"\nT-statistic for Alcohol Year and Relationship Status: {t_stat_alcohol_relationship}")
print(f"P-value for Alcohol Year and Relationship Status: {p_val_alcohol_relationship}")
if p_val_alcohol_relationship < 0.05:
    print("There is a significant difference in alcohol consumption between single and committed individuals.")
else:
    print("There is no significant difference in alcohol consumption between single and committed individuals.")
print("Mean AlcoholYear by Relationship Status:")
print(df_ttest.groupby('RelationshipStatus')['AlcoholYear'].mean())
