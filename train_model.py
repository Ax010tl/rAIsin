import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import sys

# Define column names
columns = [
  'inner_word_ngrams',
  'inner_pos_ngrams',
  'inner_average_sentence_length',
  'inner_normalized_standard_deviation_of_sentence_length',
  'inner_average_word_length',
  'inner_normalized_standard_deviation_of_word_length',
  'inner_average_syllables_per_word',
  'inner_normalized_standard_deviation_of_syllables_per_word',
  'inner_average_flesch_kincaid_reading_ease_score_per_sentence',
  'inner_normalized_standard_deviation_of_flesch_kincaid_reading_ease_score_per_sentence',
  'inner_average_gunning_fog_index_per_sentence',
  'inner_normalized_standard_deviation_of_gunning_fog_index_per_sentence',
  'external_word_ngrams',
  'external_pos_ngrams',
  'external_average_sentence_length',
  'external_normalized_standard_deviation_of_sentence_length',
  'external_average_word_length',
  'external_normalized_standard_deviation_of_word_length',
  'external_average_syllables_per_word',
  'external_normalized_standard_deviation_of_syllables_per_word',
  'external_average_flesch_kincaid_reading_ease_score_per_sentence',
  'external_normalized_standard_deviation_of_flesch_kincaid_reading_ease_score_per_sentence',
  'external_average_gunning_fog_index_per_sentence',
  'external_normalized_standard_deviation_of_gunning_fog_index_per_sentence',
  'sentence_vector_similarity_max',
  'sentence_vector_similarity_avg',
]

# Design neural network architecture
model = Sequential()
model.add(Dense(64, input_shape=(len(columns),), activation='sigmoid'))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0003), metrics=['accuracy'])

def train_raisin_model(data):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data[columns], data['result'], test_size=0.2)
    # Train model
    model.fit(X_train, y_train, epochs=8_000, batch_size=32, validation_data=(X_test, y_test))
    # Evaluate model
    accuracy = model.evaluate(X_test, y_test)[1]
    print("Accuracy: {}".format(accuracy))
    # Save model
    model.save('raisin_model.h5')

def main():
    # Read raisin_comparison.csv for clean and raisin_comparison.csv for plag
    if len(sys.argv) > 2:
        clean_csv = sys.argv[1]
        plag_csv = sys.argv[2]
    else:
        raise Exception("Please provide clean and plag raisin_comparison.csv file paths as command line arguments")
    # Load clean and plag data into dataframes
    clean_data = pd.read_csv(clean_csv)
    plag_data = pd.read_csv(plag_csv)
    # Add result column to dataframes
    clean_data['result'] = 0
    plag_data['result'] = 1
    # Concatenate clean and plag dataframes
    data = pd.concat([clean_data, plag_data])
    # Train model
    train_raisin_model(data)

if __name__ == '__main__':
    main()