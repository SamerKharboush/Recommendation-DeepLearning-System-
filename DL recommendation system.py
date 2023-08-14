import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dot, Flatten, Dense
from tensorflow.keras.models import Model

# Create a DataFrame from the sample data
df = pd.read_csv('data')

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

# Collaborative Filtering Model using Matrix Factorization
num_users = len(interaction_matrix)
num_items = len(interaction_matrix.columns)
embedding_size = 50  # You can experiment with the embedding size

# Define user and item inputs
user_input = Input(shape=(1,), name='user_input')
item_input = Input(shape=(1,), name='item_input')

# User and item embeddings
user_embedding = Embedding(num_users, embedding_size, name='user_embedding')(user_input)
item_embedding = Embedding(num_items, embedding_size, name='item_embedding')(item_input)

# Dot product of user and item embeddings
dot_product = Dot(axes=2, name='dot_product')([user_embedding, item_embedding])

# Flatten the dot product
flatten_output = Flatten(name='flatten')(dot_product)

# Add a dense layer for final recommendation scores
dense_output = Dense(1, activation='linear', name='output')(flatten_output)

# Model
model = Model(inputs=[user_input, item_input], outputs=dense_output)
model.compile(loss='mean_squared_error', optimizer='adam')

# Generate numpy arrays for user and item indices
users = interaction_matrix.index.values
items = interaction_matrix.columns.values

# Prepare data for training
user_indices = np.repeat(np.arange(num_users), num_items)
item_indices = np.tile(np.arange(num_items), num_users)
target_values = interaction_matrix.values.flatten()

# Train the model
history = model.fit(x=[user_indices, item_indices], y=target_values, epochs=100, verbose=10)

# Testing the Recommendations
# Replace "0298cc962e684211" (without brackets) with the visitor ID you want to test
visitor_id_to_test = "0298cc962e684211"

# Get the index of the visitor
visitor_idx = np.where(users == visitor_id_to_test)[0][0]

# Predict item scores for the given visitor
item_scores = model.predict([np.array([visitor_idx] * num_items), np.arange(num_items)])

# Find the top N items to recommend
n = 5
recommended_items = np.argsort(item_scores, axis=None)[::-1][:n]

print("\nRecommended items for Visitor ID", visitor_id_to_test, ":")
for item_idx in recommended_items:
    item_name = label_encoder.inverse_transform([item_idx])[0]
    print("-", item_name)
