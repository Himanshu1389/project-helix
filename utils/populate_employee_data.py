import pandas as pd
import numpy as np
import os
import random

# Define the path to the CSV file
data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
csv_path = os.path.join(data_dir, "employee_data.csv")

# Lists of common names for generation
male_names = ["Liam", "Noah", "Oliver", "Elijah", "James", "William", "Benjamin", "Lucas", "Henry", "Theodore", "Jack", "Levi", "Alexander", "Jackson", "Mateo", "Daniel", "Michael", "Ethan", "Logan", "Owen", "Samuel", "Jacob", "Asher", "Aiden", "John", "Joseph", "David", "Matthew", "Luke", "Anthony", "Christopher", "Andrew", "Joshua", "Gabriel", "Dylan", "Caleb", "Isaac", "Ryan", "Nathan", "Adam", "Arthur", "Louis", "Felix", "Leon", "Ezra", "Leo", "Finn", "Jasper", "Milo"]
female_names = ["Olivia", "Emma", "Ava", "Sophia", "Isabella", "Charlotte", "Amelia", "Mia", "Harper", "Evelyn", "Abigail", "Emily", "Ella", "Elizabeth", "Camila", "Luna", "Sofia", "Avery", "Mila", "Aria", "Scarlett", "Penelope", "Layla", "Chloe", "Victoria", "Madison", "Eleanor", "Grace", "Nora", "Riley", "Zoey", "Hannah", "Hazel", "Violet", "Aurora", "Savannah", "Alice", "Audrey", "Bella", "Maya", "Stella", "Clara", "Ruby", "Eva", "Sophie", "Lucy", "Autumn", "Willow", "Nova"]

# Load existing data
try:
    employee_df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"Error: {csv_path} not found.")
    exit()

# Ensure necessary columns exist for the logic
required_columns = ['name', 'department', 'role', 'experience', 'salary', 'target_salary', 'comp_ratio', 'gender', 'budget']
for col in required_columns:
    if col not in employee_df.columns:
        # If a column is missing, it might be due to a malformed CSV or new data.
        # For simplicity, let's just add it with a default value.
        # In a real application, you might want to handle this more robustly.
        employee_df[col] = np.nan
        if col == 'comp_ratio':
            employee_df[col] = employee_df['salary'] / employee_df['target_salary']
        print(f"Warning: Column '{col}' not found. Added with default values.")

# Get unique departments and roles
unique_departments = employee_df['department'].unique()
unique_roles = employee_df['role'].unique()

new_records = []
# No need for employee_id_counter for names anymore, as we'll use random names

for dept in unique_departments:
    for role in unique_roles:
        subset_df = employee_df[(employee_df['department'] == dept) & (employee_df['role'] == role)]
        
        male_count = subset_df[subset_df['gender'] == 'M'].shape[0]
        female_count = subset_df[subset_df['gender'] == 'F'].shape[0]
        
        # Determine average salary and experience for this role/department for realistic new entries
        avg_salary = subset_df['salary'].mean() if not subset_df['salary'].empty else 1000000 # Default if no data
        avg_experience = subset_df['experience'].mean() if not subset_df['experience'].empty else 5 # Default if no data

        # Add male employees if needed
        for _ in range(max(0, 2 - male_count)):
            salary = avg_salary * (1 + np.random.uniform(-0.1, 0.1)) # +/- 10%
            target_salary = salary / np.random.uniform(0.9, 1.0) # Comp ratio between 0.9 and 1.0
            new_records.append({
                'name': random.choice(male_names) + " " + random.choice(["Smith", "Jones", "Williams", "Brown", "Davis", "Miller", "Wilson", "Moore", "Taylor", "Anderson"]),
                'department': dept,
                'role': role,
                'experience': max(1, round(avg_experience + np.random.uniform(-2, 2))),
                'salary': round(salary, -3), # Round to nearest thousand
                'target_salary': round(target_salary, -3),
                'comp_ratio': round(salary / target_salary, 2),
                'gender': 'M',
                'budget': round(target_salary * 1.2, -3) # Budget slightly above target
            })
            
        # Add female employees if needed
        for _ in range(max(0, 2 - female_count)):
            salary = avg_salary * (1 + np.random.uniform(-0.1, 0.1)) # +/- 10%
            target_salary = salary / np.random.uniform(0.9, 1.0) # Comp ratio between 0.9 and 1.0
            new_records.append({
                'name': random.choice(female_names) + " " + random.choice(["Smith", "Jones", "Williams", "Brown", "Davis", "Miller", "Wilson", "Moore", "Taylor", "Anderson"]),
                'department': dept,
                'role': role,
                'experience': max(1, round(avg_experience + np.random.uniform(-2, 2))),
                'salary': round(salary, -3),
                'target_salary': round(target_salary, -3),
                'comp_ratio': round(salary / target_salary, 2),
                'gender': 'F',
                'budget': round(target_salary * 1.2, -3)
            })

if new_records:
    new_df = pd.DataFrame(new_records)
    updated_employee_df = pd.concat([employee_df, new_df], ignore_index=True)
    updated_employee_df.to_csv(csv_path, index=False)
    print(f"Added {len(new_records)} new records to {csv_path}")
else:
    print("No new records needed. Dataset already meets gender distribution criteria for all roles/departments.")

# Display a sample of the updated data (optional)
# print(updated_employee_df.head()) 