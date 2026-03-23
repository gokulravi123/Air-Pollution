import os
import subprocess

# List of scripts to run
scripts = [
    'linear_regression.py',
    'logistic_regression.py',
    'decision_tree.py',
    'random_forest.py'
]

print("--- Running Indoor Air Pollution Analysis Project ---")
print("Executing models and comparing results...\n")

for script in scripts:
    result = subprocess.run(['python', script], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"Error in {script}: {result.stderr}")

print("\n--- Final Project Summary ---")
print("Target: Predict CO level (Regression) and Air_Quality (Classification)")
print("Based on the accuracy scores above, compare Logistic Regression, Decision Tree, and Random Forest.")
print("Recommendation: Use Random Forest for best classification performance.")
