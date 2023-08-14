import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

# Create a DataFrame from the sample data
df = pd.read_csv('Data')


# Data Preprocessing
# Handle missing values
df.fillna("No data", inplace=True)

# Remove brackets from Visitor ID
df["Visitor ID"] = df["Visitor ID"].str.strip("[]")

# Encode categorical variables
label_encoder = LabelEncoder()
df["Page title"] = label_encoder.fit_transform(df["Page title"])
df["Previous page view title"] = label_encoder.transform(df["Previous page view title"])

# Prepare the interaction matrix (pivot table) with "Previous page view title" as a fallback
interaction_matrix = df.pivot_table(
    index="Visitor ID", columns="Page title", values="Content interactions", fill_value=0
)

# Display the interaction matrix
print("Interaction Matrix:")
print(interaction_matrix)

# Collaborative Filtering Model
# Compute user similarity using cosine similarity based on the interaction matrix
user_similarity = cosine_similarity(interaction_matrix)

# Generate item recommendations for each user (excluding items the user has already interacted with)
def recommend_items(visitor_id, interaction_matrix, user_similarity, n=5):
    visitor_idx = interaction_matrix.index.get_loc(visitor_id)
    visitor_interactions = interaction_matrix.iloc[visitor_idx].values.reshape(1, -1)
    non_interacted_items_mask = visitor_interactions == 0
    item_scores = user_similarity[visitor_idx].dot(interaction_matrix.values)
    items_to_recommend = item_scores[non_interacted_items_mask[0]].argsort()[-n:][::-1]
    return items_to_recommend

# Testing the Recommendations
# Replace "0298cc962e684211" (without brackets) with the visitor ID you want to test
visitor_id_to_test = "0298cc962e684211"
recommended_items = recommend_items(visitor_id_to_test, interaction_matrix, user_similarity, n=5)

print("\nRecommended items for Visitor ID", visitor_id_to_test, ":")
for item_idx in recommended_items:
    item_name = label_encoder.inverse_transform([item_idx])[0]
    print("-", item_name)
