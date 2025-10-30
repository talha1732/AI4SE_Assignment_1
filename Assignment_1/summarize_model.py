
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV
df = pd.read_csv("./testsets/benchmark_results.csv")

# Ensure prediction_score is numeric
df['prediction_score'] = pd.to_numeric(df['prediction_score'], errors='coerce').fillna(0)

# Filter rows where an 'if' was actually masked
if_rows = df[df['has_if'] == True]

# Average similarity for actual if statements
avg_similarity = if_rows['prediction_score'].mean()
print(f"Average similarity (for rows with masked if): {avg_similarity:.2f}%")

# Top 10 worst predictions
worst_preds = if_rows.nsmallest(10, 'prediction_score')
print("\nTop 10 worst predictions:")
print(worst_preds[['target_text', 'predicted_if', 'prediction_score']])

# Top 10 best predictions
best_preds = if_rows.nlargest(10, 'prediction_score')
print("\nTop 10 best predictions:")
print(best_preds[['target_text', 'predicted_if', 'prediction_score']])

# Histogram of prediction scores
plt.figure(figsize=(10,5))
plt.hist(if_rows['prediction_score'], bins=20, color='skyblue', edgecolor='black')
plt.title("Distribution of Similarity Scores for Masked If Statements")
plt.xlabel("Similarity (%)")
plt.ylabel("Number of If Statements")
plt.show()
