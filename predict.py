import sys
import pandas as pd
from compare_file import compare_file
from keras.models import load_model

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

def main():
    # Read file path, model, raisin_vector_path, and raisin_stylometry_path from command line arguments
    if len(sys.argv) > 4:
        file_path = sys.argv[1]
        model = sys.argv[2]
        raisin_vector_path = sys.argv[3]
        raisin_stylometry_path = sys.argv[4]
    else:
        raise Exception("Please provide file path, model, raisin vector path and raisin stylometry path as command line arguments")
    # Get the dataframes
    raisin_vectors_df = pd.read_csv(raisin_vector_path)
    raisin_stylometry_df = pd.read_csv(raisin_stylometry_path)
    # Analyze file
    file_analysis = compare_file(file_path, raisin_vectors_df, raisin_stylometry_df)
    # Convert to dataframe
    file_analysis_df = pd.DataFrame([file_analysis], columns=columns)
    # Load tensorflow model
    model = load_model(model)
    # Predict
    prediction = model.predict(file_analysis_df)
    # Print prediction
    print(f"Prediction: {prediction[0][0]}. \x1b[1m {'Plagiarism' if prediction[0][0] > 0.5 else 'Not plagiarism'}\x1b[0m")

if __name__ == "__main__": 
    main()