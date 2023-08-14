import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Flatten, Concatenate, Reshape
from tensorflow.keras.models import Model

# Create a DataFrame from the sample data
df = pd.read_csv('/content/Deep_Analysis_-_Academy.csv')

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

# Collaborative Filtering Model using LSTM
num_users = len(interaction_matrix)
num_items = len(interaction_matrix.columns)
embedding_size = 50  # You can experiment with the embedding size

# Define user and item inputs
user_input = Input(shape=(None,), name='user_input')  # Variable-length sequence
item_input = Input(shape=(None,), name='item_input')  # Variable-length sequence

# User and item embeddings for sequences
user_embedding = Embedding(num_users, embedding_size, name='user_embedding')(user_input)
item_embedding = Embedding(num_items, embedding_size, name='item_embedding')(item_input)

# LSTM layers to capture sequential patterns
lstm_units = 64  # You can experiment with the number of LSTM units
user_lstm = LSTM(lstm_units, return_sequences=True)(user_embedding)
item_lstm = LSTM(lstm_units, return_sequences=True)(item_embedding)

# Concatenate LSTM outputs
concatenated_output = Concatenate(axis=1)([user_lstm, item_lstm])

# Reshape the concatenated output to match the LSTM layer input shape
reshaped_output = Reshape((-1, lstm_units * 2))(concatenated_output)

# LSTM layers for recommendation
lstm_units_recommendation = 64
lstm_output = LSTM(lstm_units_recommendation, return_sequences=False)(reshaped_output)

# Add a dense layer for intermediate representation
dense_output = Dense(128, activation='relu')(lstm_output)

# Final recommendation output
recommendation_output = Dense(1, activation='linear', name='output')(dense_output)


# Model
model = Model(inputs=[user_input, item_input], outputs=recommendation_output)
model.compile(loss='mean_squared_error', optimizer='adam')

# Generate numpy arrays for user and item indices
users = interaction_matrix.index.values
items = interaction_matrix.columns.values

# Prepare data for training
user_indices = np.repeat(np.arange(num_users), num_items)
item_indices = np.tile(np.arange(num_items), num_users)
target_values = interaction_matrix.values.flatten()

# Define the sequence length (number of interactions in each sequence)
sequence_length = num_items  # You can change this based on your use case

# Convert user and item indices to sequences of interactions
user_sequences = [user_indices[i:i+sequence_length] for i in range(0, len(user_indices), sequence_length)]
item_sequences = [item_indices[i:i+sequence_length] for i in range(0, len(item_indices), sequence_length)]
target_sequences = [target_values[i:i+sequence_length] for i in range(0, len(target_values), sequence_length)]

# Convert the sequences to numpy arrays
user_sequences = np.array(user_sequences)
item_sequences = np.array(item_sequences)
target_sequences = np.array(target_sequences)

# Train the model
history = model.fit(
    x=[user_sequences, item_sequences],
    y=target_sequences,
    epochs=10,
    verbose=1
)

# Testing the Recommendations
# Replace "0298cc962e684211" (without brackets) with the visitor ID you want to test
visitor_id_to_test = "0298cc962e684211"


# Get the index of the visitor
visitor_idx = np.where(users == visitor_id_to_test)[0][0]

# Repeat the visitor index for all items
visitor_indices = np.repeat(visitor_idx, num_items)

# Convert the visitor index into a sequence of interactions (items)
visitor_interaction_sequence = np.repeat(np.arange(num_items), len(users))
visitor_indices = np.tile(np.where(users == visitor_id_to_test)[0][0], num_items)

# Use the same number of items for visitor_interaction_sequence as the length of visitor_indices
num_items_to_repeat = len(visitor_indices) // len(users)
visitor_interaction_sequence = np.tile(np.arange(num_items), len(users) * num_items_to_repeat)
visitor_indices = np.repeat(np.where(users == visitor_id_to_test)[0][0], len(visitor_interaction_sequence))
# ... (Previous code for training the model)

# Predict item scores for the given visitor
visitor_indices = np.repeat(visitor_idx, num_items)  # Repeat visitor index for all items
visitor_interaction_sequence = np.tile(np.arange(num_items), len(users))  # Interaction sequence for the visitor

# Reshape the interaction sequence to match the expected shape
num_interactions = len(visitor_interaction_sequence)
num_sequences = num_interactions // sequence_length
visitor_interaction_sequence = visitor_interaction_sequence[:num_sequences * sequence_length]  # Truncate if necessary
visitor_interaction_sequence = visitor_interaction_sequence.reshape(-1, sequence_length)
# Repeat the visitor index for all items
visitor_indices = np.repeat(visitor_idx, num_items)

# Create an interaction sequence for the visitor with the same shape as item_indices
visitor_interaction_sequence = np.tile(np.arange(num_items), len(users))
visitor_indices = np.repeat(np.where(users == visitor_id_to_test)[0][0], len(visitor_interaction_sequence))

# Predict item scores for the given visitor interaction sequence
item_scores = model.predict([visitor_indices, visitor_interaction_sequence])

# Find the top N items to recommend
n = 5
recommended_items = np.argsort(item_scores.flatten())[::-1][:n]

print("\nRecommended items for Visitor ID", visitor_id_to_test, ":")
for item_idx in recommended_items:
    item_name = label_encoder.inverse_transform([item_idx])[0]
    print("-", item_name)
    
