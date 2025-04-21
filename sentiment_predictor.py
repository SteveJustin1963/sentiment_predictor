import json
import os
from sklearn.linear_model import LinearRegression
import numpy as np

# Initial training data (100 examples)
training_data = [
    "3 The day is quite average today",
    "7 I enjoy this pleasant weather",
    "5 Work was okay nothing special",
    "8 Love the beautiful sunshine now",
    "2 Feeling pretty down this morning",
    "6 Nice to see some clouds",
    "4 Traffic was annoying today",
    "9 Fantastic party last night",
    "1 Really terrible experience yesterday",
    "5 Meh just another day",
    "7 Great food at the restaurant",
    "3 Boring meeting this afternoon",
    "8 Awesome movie tonight",
    "2 Rain ruined my plans",
    "6 Decent workout session",
    "4 Slightly tiring day overall",
    "9 Amazing sunset view",
    "1 Awful customer service",
    "5 Average lunch break",
    "7 Pretty good book",
    "3 Dull weather outside",
    "8 Exciting game tonight",
    "2 Sad news this morning",
    "6 Cool breeze today",
    "4 Okay but stressful day",
    "9 Wonderful family time",
    "1 Horrible traffic jam",
    "5 Standard workday routine",
    "7 Fun evening with friends",
    "3 Mediocre movie choice",
    "8 Lovely garden walk",
    "2 Bad headache today",
    "6 Interesting podcast episode",
    "4 Slightly boring class",
    "9 Perfect vacation day",
    "1 Terrible food quality",
    "5 So-so weather today",
    "7 Nice chat with neighbor",
    "3 Bland dinner tonight",
    "8 Great concert experience",
    "2 Lousy internet connection",
    "6 Fairly good meeting",
    "4 Minor car trouble",
    "9 Super happy moment",
    "1 Dreadful work task",
    "5 Typical Monday feeling",
    "7 Cool new hobby",
    "3 Unexciting news update",
    "8 Brilliant sunset colors",
    "2 Poor sleep last night",
    "6 Alright day overall",
    "4 Bit of a struggle",
    "9 Awesome beach trip",
    "1 Rotten luck today",
    "5 Neutral about this",
    "7 Good workout results",
    "3 Tedious chores done",
    "8 Happy family dinner",
    "2 Gloomy weather mood",
    "6 Fine morning run",
    "4 Slightly off day",
    "9 Terrific news received",
    "1 Miserable cold symptoms",
    "5 Okay team meeting",
    "7 Pleasant walk outside",
    "3 Dull presentation given",
    "8 Great shopping find",
    "2 Awful flight delay",
    "6 Decent coffee break",
    "4 Minor work stress",
    "9 Fantastic book ending",
    "1 Lousy weather change",
    "5 Average gym visit",
    "7 Nice team lunch",
    "3 Boring traffic wait",
    "8 Lovely evening out",
    "2 Bad mood today",
    "6 Okay movie night",
    "4 Slight headache now",
    "9 Amazing travel plans",
    "1 Terrible phone call",
    "5 So-so work progress",
    "7 Good friend visit",
    "3 Uninspired day overall",
    "8 Excellent food taste",
    "2 Poor quality item",
    "6 Fair weather shift",
    "4 Bit tired now",
    "9 Super fun event",
    "1 Crummy old car",
    "5 Regular school day",
    "7 Tasty homemade meal",
    "3 Monotonous routine",
    "8 Thrilling sports match",
    "2 Lost my wallet",
    "6 Calm evening at home",
    "4 Small argument today",
    "10 Best day ever today",
    "1 Worst mistake made ever",
    "5 Just a normal afternoon"
]

# Load additional data from file
additional_data_file = "additional_training_data.json"
if os.path.exists(additional_data_file):
    print(f"Loading additional training data from {additional_data_file}...")
    with open(additional_data_file, 'r') as f:
        additional_data = json.load(f)
    training_data.extend(additional_data)
    print(f"Added {len(additional_data)} additional examples")
else:
    additional_data = []

# Process initial data
X_train = [line.split(' ', 1)[1] for line in training_data]
y_train = [float(line.split(' ', 1)[0]) for line in training_data]

print(f"Processed data: {len(X_train)} sentences, {len(y_train)} labels")

# Feature extraction
def build_features(sentences):
    all_words = []
    for sentence in sentences:
        words = sentence.lower().split()
        all_words.extend(words)
    unique_words = list(set(all_words))
    word_counts = {word: all_words.count(word) for word in unique_words}
    freq_words = [word for word, count in word_counts.items() if count >= 2]
    if len(freq_words) > 100:
        freq_words = sorted(freq_words, key=lambda w: word_counts[w], reverse=True)[:100]
    return freq_words

freq_words = build_features(X_train)
print(f"Using {len(freq_words)} frequent words as features")
print(f"Frequent words: {freq_words}")  # Debug vocabulary

# Create feature matrix
def create_feature_matrix(sentences, vocab):
    X = []
    for sentence in sentences:
        words = sentence.lower().split()
        row = [words.count(word) for word in vocab]
        X.append(row)
    return np.array(X)

X_matrix = create_feature_matrix(X_train, freq_words)
print(f"X_matrix dimensions: {X_matrix.shape}")
print(f"y_train dimensions: {len(y_train)}")

# Response construction with improved grammar
def generate_response(sentence, score, training_sentences, training_scores, recent_sentences, recent_scores):
    words = sentence.lower().split()
    is_question = sentence.strip().endswith('?')
    
    # Prioritize recent inputs
    if recent_sentences:
        closest_idx = min(range(len(recent_scores)), key=lambda i: abs(recent_scores[i] - score))
        closest_sentence = recent_sentences[closest_idx]
        closest_words = closest_sentence.lower().split()
    else:
        closest_idx = min(range(len(training_scores)), key=lambda i: abs(training_scores[i] - score))
        closest_sentence = training_sentences[closest_idx]
        closest_words = closest_sentence.lower().split()

    # Extract keywords (prefer nouns/adjectives, fall back to any word)
    stop_words = {'is', 'was', 'to', 'the', 'a', 'this', 'that', 'i', 'have', 'can', 'for'}
    input_keywords = [w for w in words if w in freq_words and w not in stop_words]
    if not input_keywords:
        input_keywords = [w for w in words if w not in stop_words] or [words[-1]]  # Last word as fallback
    closest_keywords = [w for w in closest_words if w in freq_words and w not in stop_words]
    if not closest_keywords:
        closest_keywords = [w for w in closest_words if w not in stop_words] or [closest_words[-1]]

    # Sentiment-based adjective
    if score >= 7:
        adj = np.random.choice(["great", "nice", "awesome"])
    elif score >= 4:
        adj = np.random.choice(["okay", "fine", "decent"])
    else:
        adj = np.random.choice(["tough", "bad", "rough"])

    # Construct response with proper grammar
    subject = input_keywords[0]
    context = closest_keywords[0] if closest_keywords[0] != subject else (closest_keywords[1] if len(closest_keywords) > 1 else subject)
    
    if is_question:
        prefix = "Yes" if score >= 5 else "No"
        return f"{prefix}, {subject} can feel {adj} like {context}."
    else:
        return f"That’s a {adj} {subject} like {context}."

# Train model
threshold = 20
new_sentences = []
new_labels = []
recent_sentences = []
recent_scores = []
retrain_count = 0

if len(X_train) >= threshold:
    model = LinearRegression()
    model.fit(X_matrix, y_train)
    print(f"Model trained with {len(X_train)} examples and {X_matrix.shape[1]} features")

    # Prediction loop
    while True:
        try:
            new_sentence = input("Enter sentence for prediction (or press Enter to exit): ").strip()
            if not new_sentence:
                if new_sentences:
                    additional_data.extend([f"{label} {sentence}" for label, sentence in zip(new_labels, new_sentences)])
                    with open(additional_data_file, 'w') as f:
                        json.dump(additional_data, f)
                    print(f"Saved {len(new_sentences)} new training examples to {additional_data_file}")
                print("Prediction phase ended")
                break

            # Check known words
            words = new_sentence.lower().split()
            has_known_words = any(word in freq_words for word in words)

            if not has_known_words:
                print("This sentence contains few known words.")
                while True:
                    try:
                        rating = float(input("Please rate (1-10): "))
                        if 1 <= rating <= 10:
                            break
                        print("Rating must be between 1 and 10")
                    except ValueError:
                        print("Please enter a valid number")
                new_sentences.append(new_sentence)
                new_labels.append(rating)
                recent_sentences.append(new_sentence)
                recent_scores.append(rating)
                if len(recent_sentences) > 5:
                    recent_sentences.pop(0)
                    recent_scores.pop(0)
                print("Added to training data. Thank you!")
                print(f"Response: {generate_response(new_sentence, rating, X_train, y_train, recent_sentences, recent_scores)}")
            else:
                new_X = create_feature_matrix([new_sentence], freq_words)
                prediction = model.predict(new_X)[0]
                print(f"Prediction: {prediction:.2f}")
                print(f"Response: {generate_response(new_sentence, prediction, X_train, y_train, recent_sentences, recent_scores)}")
                feedback = input("Is this prediction accurate? (y/n, or enter to skip): ").lower()
                if feedback == 'n':
                    while True:
                        try:
                            rating = float(input("What would be a better rating (1-10)? "))
                            if 1 <= rating <= 10:
                                break
                            print("Rating must be between 1 and 10")
                        except ValueError:
                            print("Please enter a valid number")
                    new_sentences.append(new_sentence)
                    new_labels.append(rating)
                    recent_sentences.append(new_sentence)
                    recent_scores.append(rating)
                    if len(recent_sentences) > 5:
                        recent_sentences.pop(0)
                        recent_scores.pop(0)
                    print("Added to training data. Thank you!")
                    print(f"Updated Response: {generate_response(new_sentence, rating, X_train, y_train, recent_sentences, recent_scores)}")

            # Retrain dynamically every 5 new examples
            if len(new_sentences) >= 5 and len(new_sentences) % 5 == 0:
                print("Retraining model with new data...")
                X_train.extend(new_sentences[-5:])
                y_train.extend(new_labels[-5:])
                freq_words = build_features(X_train)
                X_matrix = create_feature_matrix(X_train, freq_words)
                model.fit(X_matrix, y_train)
                retrain_count += 1
                print(f"Model retrained ({retrain_count}) with {len(X_train)} examples and {X_matrix.shape[1]} features")

        except KeyboardInterrupt:
            if new_sentences:
                additional_data.extend([f"{label} {sentence}" for label, sentence in zip(new_labels, new_sentences)])
                with open(additional_data_file, 'w') as f:
                    json.dump(additional_data, f)
                print(f"Saved {len(new_sentences)} new training examples to {additional_data_file}")
            print("\nPrediction phase ended")
            break
        except Exception as e:
            print(f"Error: {e}")
else:
    print(f"Insufficient data: only {len(X_train)} examples found")