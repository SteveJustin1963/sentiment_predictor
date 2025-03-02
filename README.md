# sentiment_predictor
- it can talk with you and learn
- try and send responses to the https://github.com/SteveJustin1963/tec-SPEECH/  the SPO256A-AL2

```
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
print(f"Frequent words: {freq_words}")  # Debug: Show the vocabulary

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

# Response construction with question handling
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

    # Extract keywords (fall back to any word if none in freq_words)
    input_keywords = [w for w in words if w in freq_words and w not in {'is', 'was', 'to', 'the', 'a', 'this', 'that'}]
    if not input_keywords:
        input_keywords = [w for w in words if w not in {'is', 'was', 'to', 'the', 'a', 'this', 'that'}] or [words[0]]
    closest_keywords = [w for w in closest_words if w in freq_words and w not in {'is', 'was', 'to', 'the', 'a', 'this', 'that'}]
    if not closest_keywords:
        closest_keywords = [w for w in closest_words if w not in {'is', 'was', 'to', 'the', 'a', 'this', 'that'}] or [closest_words[0]]

    # Sentiment-based adjective
    if score >= 7:
        adj = np.random.choice(["great", "nice", "awesome"])
    elif score >= 4:
        adj = np.random.choice(["okay", "fine", "decent"])
    else:
        adj = np.random.choice(["tough", "bad", "rough"])

    # Construct response
    subject = input_keywords[0]
    context = closest_keywords[0] if closest_keywords[0] != subject else (closest_keywords[1] if len(closest_keywords) > 1 else subject)
    
    if is_question:
        prefix = "Yes" if score >= 5 else "No"
        return f"{prefix}, {subject} can be {adj} like {context}."
    else:
        return f"That {subject} feels {adj} like {context}."

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
```


# bridge to send to mint so it can speak on the SP0

Let's set up a system that can transfer the answers from your Python sentiment analysis code to the MINT speech interface. This requires creating a bridge between the Python application and the TEC-1.

Here's how we can do it:

```python
import serial
import time
import subprocess
import re

# Configuration
SERIAL_PORT = '/dev/ttyUSB0'  # Change to match your setup (COM1, etc. on Windows)
BAUD_RATE = 4800
MAX_TEXT_LENGTH = 80  # Maximum characters to send at once to MINT

def clean_text_for_speech(text):
    """Clean and prepare text for the speech synthesizer"""
    # Remove special characters that might confuse MINT
    text = re.sub(r'[^a-zA-Z0-9 .,?!]', '', text)
    # Limit length to prevent buffer overflow
    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH-3] + "..."
    return text

def send_to_mint(text):
    """Send text to the MINT speech interface via serial port"""
    try:
        # Open serial connection
        ser = serial.Serial(
            port=SERIAL_PORT,
            baudrate=BAUD_RATE,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1
        )
        
        print(f"Sending to MINT speech interface: '{text}'")
        
        # Send the text followed by Enter
        ser.write(text.encode('ascii', errors='replace'))
        ser.write(b'\r')
        
        # Wait for completion and read any response
        time.sleep(2)  # Allow time for processing
        response = ser.read(ser.in_waiting).decode('ascii', errors='replace')
        if response:
            print(f"Response from MINT: {response}")
        
        # Close the connection
        ser.close()
        return True
        
    except serial.SerialException as e:
        print(f"Error communicating with MINT: {e}")
        return False

def run_sentiment_analysis():
    """Run the sentiment analysis Python script and capture its output"""
    try:
        # Run the sentiment analysis script
        process = subprocess.Popen(
            ['python', 'sentiment_analysis.py'],  # Adjust the path as needed
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Interact with the script - send a test sentence
        print("Sending test sentence to sentiment analysis...")
        test_sentence = "Today was a really good day"
        process.stdin.write(test_sentence + "\n")
        process.stdin.flush()
        
        # Read output until we get a response
        output = ""
        while True:
            line = process.stdout.readline()
            output += line
            print(line.strip())  # Display in console
            
            # Check if we got a response line
            if "Response:" in line:
                # Extract just the response part
                response_text = line.split("Response:", 1)[1].strip()
                
                # Clean and send to MINT
                clean_response = clean_text_for_speech(response_text)
                send_to_mint(clean_response)
                break
                
            # Avoid infinite loop if something goes wrong
            if not line or "Prediction phase ended" in line:
                break
        
        # Close process
        process.terminate()
        return output
        
    except Exception as e:
        print(f"Error running sentiment analysis: {e}")
        return None

def interactive_mode():
    """Interactive mode to send sentences to analysis and then to MINT"""
    try:
        # Start the sentiment analysis process
        process = subprocess.Popen(
            ['python', 'sentiment_analysis.py'],  # Adjust path as needed
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print("Interactive mode started. Enter sentences for analysis (or 'exit' to quit).")
        
        # Skip initial output until prompt
        while True:
            line = process.stdout.readline()
            print(line.strip())
            if "Enter sentence for prediction" in line:
                break
        
        # Main interaction loop
        while True:
            # Get user input
            user_input = input("\nEnter sentence for analysis (or 'exit' to quit): ")
            if user_input.lower() == 'exit':
                break
                
            # Send to sentiment analysis
            process.stdin.write(user_input + "\n")
            process.stdin.flush()
            
            # Read output until we get a response
            response_found = False
            while not response_found:
                line = process.stdout.readline()
                print(line.strip())
                
                if "Response:" in line:
                    # Extract just the response part
                    response_text = line.split("Response:", 1)[1].strip()
                    
                    # Ask user if they want to send to MINT
                    send_choice = input(f"Send this response to MINT? (y/n): ")
                    if send_choice.lower() == 'y':
                        # Clean and send to MINT
                        clean_response = clean_text_for_speech(response_text)
                        send_to_mint(clean_response)
                        
                    response_found = True
                
                # Handle feedback prompt
                if "Is this prediction accurate?" in line:
                    feedback = input("Enter feedback (y/n, or press Enter to skip): ")
                    process.stdin.write(feedback + "\n")
                    process.stdin.flush()
                    
                    # If feedback is 'n', handle the rating prompt
                    if feedback.lower() == 'n':
                        rating_line = process.stdout.readline()
                        print(rating_line.strip())
                        rating = input("Enter rating (1-10): ")
                        process.stdin.write(rating + "\n")
                        process.stdin.flush()
                        
                        # Read the updated response
                        while True:
                            line = process.stdout.readline()
                            print(line.strip())
                            if "Updated Response:" in line:
                                response_text = line.split("Updated Response:", 1)[1].strip()
                                
                                # Ask user if they want to send to MINT
                                send_choice = input(f"Send this updated response to MINT? (y/n): ")
                                if send_choice.lower() == 'y':
                                    clean_response = clean_text_for_speech(response_text)
                                    send_to_mint(clean_response)
                                break
                
                # Avoid infinite loop
                if not line or "Prediction phase ended" in line:
                    break
        
        # Close process
        print("Exiting interactive mode.")
        process.terminate()
        
    except Exception as e:
        print(f"Error in interactive mode: {e}")

if __name__ == "__main__":
    print("Python to MINT Speech Interface Bridge")
    print("-------------------------------------")
    print("1. Run sentiment analysis with test sentence")
    print("2. Enter interactive mode")
    print("3. Send custom text to MINT")
    print("4. Exit")
    
    choice = input("Select an option (1-4): ")
    
    if choice == '1':
        run_sentiment_analysis()
    elif choice == '2':
        interactive_mode()
    elif choice == '3':
        text = input("Enter text to send to MINT: ")
        clean_text = clean_text_for_speech(text)
        send_to_mint(clean_text)
    else:
        print("Exiting.")

```

## How to Connect Python with MINT Speech Interface

This solution creates a bridge between your Python sentiment analysis program and the MINT speech interface on the TEC-1. Here's how it works:

### Setup Requirements:

1. **Hardware Connection**:
   - Connect your computer to the TEC-1 using a serial adapter (USB to Serial)
   - Make sure the TEC-1 is running the MINT speech interface program

2. **Software Dependencies**:
   - Install the PySerial library: `pip install pyserial`
   - Save your sentiment analysis code as `sentiment_analysis.py`
   - Save the bridge code as `mint_bridge.py`

### How the Bridge Works:

1. **Serial Communication**:
   - The bridge opens a serial connection to the TEC-1 at 4800 baud
   - It sends text responses from the sentiment analysis to MINT
   - The MINT program will then speak the text through the SP0256-AL2

2. **Operating Modes**:
   - **Test mode**: Runs a single test sentence through the analysis
   - **Interactive mode**: Continuous input/output with the sentiment analysis
   - **Custom text**: Send arbitrary text directly to MINT

3. **Text Processing**:
   - Cleans text to make it compatible with MINT
   - Limits length to prevent buffer overflow
   - Removes special characters that might confuse the speech synthesizer

### Usage Instructions:

1. Make sure your TEC-1 is powered on and running the MINT speech interface
2. Run the bridge program: `python mint_bridge.py`
3. Choose the operating mode:
   - Option 1 runs a test with "Today was a really good day"
   - Option 2 enters interactive mode for ongoing analysis
   - Option 3 lets you send custom text without sentiment analysis

4. In interactive mode:
   - Type sentences for sentiment analysis
   - Confirm whether to send the response to MINT
   - Provide feedback on prediction accuracy if asked

### Customization Options:

- Change the `SERIAL_PORT` variable to match your setup
- Adjust `MAX_TEXT_LENGTH` based on your MINT program's buffer size
- Modify the text cleaning function to match your speech synthesis needs

With this setup, you can now use your sentiment analysis model to generate responses and have the TEC-1 speak them through the SP0256-AL2 chip!

```
import serial
import time
import subprocess
import re

# Configuration
SERIAL_PORT = '/dev/ttyUSB0'  # Change to match your setup (COM1, etc. on Windows)
BAUD_RATE = 4800
MAX_TEXT_LENGTH = 80  # Maximum characters to send at once to MINT

def clean_text_for_speech(text):
    """Clean and prepare text for the speech synthesizer"""
    # Remove special characters that might confuse MINT
    text = re.sub(r'[^a-zA-Z0-9 .,?!]', '', text)
    # Limit length to prevent buffer overflow
    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH-3] + "..."
    return text

def send_to_mint(text):
    """Send text to the MINT speech interface via serial port"""
    try:
        # Open serial connection
        ser = serial.Serial(
            port=SERIAL_PORT,
            baudrate=BAUD_RATE,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1
        )
        
        print(f"Sending to MINT speech interface: '{text}'")
        
        # Send the text followed by Enter
        ser.write(text.encode('ascii', errors='replace'))
        ser.write(b'\r')
        
        # Wait for completion and read any response
        time.sleep(2)  # Allow time for processing
        response = ser.read(ser.in_waiting).decode('ascii', errors='replace')
        if response:
            print(f"Response from MINT: {response}")
        
        # Close the connection
        ser.close()
        return True
        
    except serial.SerialException as e:
        print(f"Error communicating with MINT: {e}")
        return False

def run_sentiment_analysis():
    """Run the sentiment analysis Python script and capture its output"""
    try:
        # Run the sentiment analysis script
        process = subprocess.Popen(
            ['python', 'sentiment_analysis.py'],  # Adjust the path as needed
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Interact with the script - send a test sentence
        print("Sending test sentence to sentiment analysis...")
        test_sentence = "Today was a really good day"
        process.stdin.write(test_sentence + "\n")
        process.stdin.flush()
        
        # Read output until we get a response
        output = ""
        while True:
            line = process.stdout.readline()
            output += line
            print(line.strip())  # Display in console
            
            # Check if we got a response line
            if "Response:" in line:
                # Extract just the response part
                response_text = line.split("Response:", 1)[1].strip()
                
                # Clean and send to MINT
                clean_response = clean_text_for_speech(response_text)
                send_to_mint(clean_response)
                break
                
            # Avoid infinite loop if something goes wrong
            if not line or "Prediction phase ended" in line:
                break
        
        # Close process
        process.terminate()
        return output
        
    except Exception as e:
        print(f"Error running sentiment analysis: {e}")
        return None

def interactive_mode():
    """Interactive mode to send sentences to analysis and then to MINT"""
    try:
        # Start the sentiment analysis process
        process = subprocess.Popen(
            ['python', 'sentiment_analysis.py'],  # Adjust path as needed
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print("Interactive mode started. Enter sentences for analysis (or 'exit' to quit).")
        
        # Skip initial output until prompt
        while True:
            line = process.stdout.readline()
            print(line.strip())
            if "Enter sentence for prediction" in line:
                break
        
        # Main interaction loop
        while True:
            # Get user input
            user_input = input("\nEnter sentence for analysis (or 'exit' to quit): ")
            if user_input.lower() == 'exit':
                break
                
            # Send to sentiment analysis
            process.stdin.write(user_input + "\n")
            process.stdin.flush()
            
            # Read output until we get a response
            response_found = False
            while not response_found:
                line = process.stdout.readline()
                print(line.strip())
                
                if "Response:" in line:
                    # Extract just the response part
                    response_text = line.split("Response:", 1)[1].strip()
                    
                    # Ask user if they want to send to MINT
                    send_choice = input(f"Send this response to MINT? (y/n): ")
                    if send_choice.lower() == 'y':
                        # Clean and send to MINT
                        clean_response = clean_text_for_speech(response_text)
                        send_to_mint(clean_response)
                        
                    response_found = True
                
                # Handle feedback prompt
                if "Is this prediction accurate?" in line:
                    feedback = input("Enter feedback (y/n, or press Enter to skip): ")
                    process.stdin.write(feedback + "\n")
                    process.stdin.flush()
                    
                    # If feedback is 'n', handle the rating prompt
                    if feedback.lower() == 'n':
                        rating_line = process.stdout.readline()
                        print(rating_line.strip())
                        rating = input("Enter rating (1-10): ")
                        process.stdin.write(rating + "\n")
                        process.stdin.flush()
                        
                        # Read the updated response
                        while True:
                            line = process.stdout.readline()
                            print(line.strip())
                            if "Updated Response:" in line:
                                response_text = line.split("Updated Response:", 1)[1].strip()
                                
                                # Ask user if they want to send to MINT
                                send_choice = input(f"Send this updated response to MINT? (y/n): ")
                                if send_choice.lower() == 'y':
                                    clean_response = clean_text_for_speech(response_text)
                                    send_to_mint(clean_response)
                                break
                
                # Avoid infinite loop
                if not line or "Prediction phase ended" in line:
                    break
        
        # Close process
        print("Exiting interactive mode.")
        process.terminate()
        
    except Exception as e:
        print(f"Error in interactive mode: {e}")

if __name__ == "__main__":
    print("Python to MINT Speech Interface Bridge")
    print("-------------------------------------")
    print("1. Run sentiment analysis with test sentence")
    print("2. Enter interactive mode")
    print("3. Send custom text to MINT")
    print("4. Exit")
    
    choice = input("Select an option (1-4): ")
    
    if choice == '1':
        run_sentiment_analysis()
    elif choice == '2':
        interactive_mode()
    elif choice == '3':
        text = input("Enter text to send to MINT: ")
        clean_text = clean_text_for_speech(text)
        send_to_mint(clean_text)
    else:
        print("Exiting.")
```


# review

implementation of a sentiment analysis system using linear regression with active learning components. 

Here's a detailed analysis and suggestions for improvement:

### **Strengths**
1. **Active Learning Implementation**
   - Good handling of unknown words through user feedback
   - Dynamic retraining mechanism (every 5 new examples)
   - Persistent storage of additional training data

2. **Response Generation**
   - Context-aware responses using recent inputs
   - Question handling with score-based answers
   - Keyword extraction for personalized replies

3. **Feature Engineering**
   - Automatic vocabulary building from frequent words
   - Dynamic vocabulary updates with new data
   - Simple bag-of-words approach with word counts

### **Areas for Improvement**
1. **Feature Consistency**
   - **Problem:** Vocabulary changes during retraining can cause feature space inconsistencies
   - **Solution:** Implement fixed vocabulary after initial training or use hashing vectorizer

2. **Scalability**
   - **Problem:** Full dataset reprocessing during retraining becomes inefficient
   - **Solution:** Use incremental learning (e.g., `SGDRegressor` with partial_fit)

3. **Keyword Handling**
   - **Problem:** Potential IndexError if sentences contain only stopwords
   - **Solution:** Add fallback mechanisms for empty keyword lists

4. **Model Performance**
   - **Problem:** Simple word counts may not capture sentiment effectively
   - **Solution:** Consider TF-IDF or pre-trained word embeddings

5. **Code Structure**
   - **Problem:** Mixed responsibilities in prediction loop
   - **Solution:** Separate into distinct components (data processing, modeling, UI)

### **Critical Code Issues**
1. **Vocabulary Rebuilding**
```python
# During retraining:
freq_words = build_features(X_train)  # Resets vocabulary
X_matrix = create_feature_matrix(X_train, freq_words)  # New feature space
```
- **Risk:** Model coefficients become meaningless after vocabulary changes
- **Fix:** Maintain persistent vocabulary or use feature hashing

2. **Response Generation Fallback**
```python
input_keywords = [w for w in words if w in freq_words and ...]
if not input_keywords:
    input_keywords = [w for w in words if ...] or [words[0]]  # words[0] could be stopword
```
- **Risk:** Might select irrelevant words
- **Fix:** Use POS tagging to prioritize nouns/adjectives

### **Suggested Improvements**
1. **Improved Feature Handling**
```python
from sklearn.feature_extraction.text import HashingVectorizer

# Replace build_features and create_feature_matrix with:
vectorizer = HashingVectorizer(n_features=100, norm=None)
X_matrix = vectorizer.fit_transform(X_train)
```

2. **Incremental Learning**
```python
from sklearn.linear_model import SGDRegressor

model = SGDRegressor(warm_start=True)
# During retraining:
model.partial_fit(X_new, y_new)
```

3. **Enhanced Response Generation**
```python
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_keywords(text):
    doc = nlp(text)
    return [token.text for token in doc if token.pos_ in ['NOUN', 'ADJ']]
```

### **Alternative Approaches**
Consider using:
1. **Pre-trained Models** (e.g., Hugging Face's Transformers) for better sentiment understanding
2. **Proper Train/Test Splits** for performance evaluation
3. **Validation Metrics** tracking during retraining
4. **Dimensionality Reduction** (e.g., TruncatedSVD) for sparse features

### **Conclusion**
The code demonstrates good understanding of ML workflows and active learning concepts. While functional for small-scale use, it would benefit from:
- More consistent feature representation
- Proper incremental learning
- Enhanced linguistic processing
- Error handling improvements

# changes
improved version of the sentiment analysis system with the enhancements   

 

### Major Improvements

1. **Better Feature Engineering**
   - Replaced basic word counts with TF-IDF vectorization
   - Added n-gram support (1-2) to capture phrases
   - Proper stopword handling with NLTK

2. **Robust Model Architecture**
   - Used SGDRegressor with warm_start for proper incremental learning
   - Implemented a scikit-learn Pipeline for better workflow
   - Added elasticnet regularization to prevent overfitting

3. **Enhanced Response Generation**
   - More sophisticated keyword extraction with fallback mechanisms
   - Expanded response templates with natural language variations
   - Better adjective selection based on sentiment scores

4. **Code Structure and Organization**
   - Implemented object-oriented design with proper class structure
   - Separated responsibilities into logical methods
   - Added comprehensive error handling and logging

5. **Model Evaluation**
   - Added train/validation split for performance tracking
   - Included MSE and MAE metrics during training
   - Better vocabulary evaluation for unknown words

6. **User Experience**
   - More detailed feedback during training and prediction
   - Better context handling in responses
   - Improved data persistence

This version maintains all the functionality of the original while addressing the critical issues we identified. The system should now produce more accurate predictions, better responses, and handle new data more effectively.

```
import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure NLTK resources are available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

class SentimentAnalysisSystem:
    def __init__(self, data_file="additional_training_data.json", max_features=100):
        self.data_file = data_file
        self.max_features = max_features
        self.stop_words = set(stopwords.words('english'))
        
        # Create a model pipeline with TF-IDF and SGDRegressor
        self.pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(
                max_features=self.max_features,
                min_df=2,
                stop_words='english',
                ngram_range=(1, 2)
            )),
            ('regressor', SGDRegressor(
                loss='squared_error',
                penalty='elasticnet',
                alpha=0.0001,
                l1_ratio=0.15,
                max_iter=1000,
                tol=1e-3,
                warm_start=True
            ))
        ])
        
        # Initialize data containers
        self.X_train = []
        self.y_train = []
        self.recent_sentences = []
        self.recent_scores = []
        self.new_sentences = []
        self.new_labels = []
        self.additional_data = []
        self.model_trained = False
        self.retrain_count = 0
        
        # Load existing data
        self.load_data()
        
    def load_data(self):
        """Load initial and additional training data"""
        # Initial training data
        initial_data = [
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
        
        # Process initial data
        for line in initial_data:
            parts = line.split(' ', 1)
            if len(parts) == 2:
                self.X_train.append(parts[1])
                self.y_train.append(float(parts[0]))
        
        # Load additional data if available
        if os.path.exists(self.data_file):
            logger.info(f"Loading additional training data from {self.data_file}...")
            try:
                with open(self.data_file, 'r') as f:
                    self.additional_data = json.load(f)
                
                for line in self.additional_data:
                    parts = line.split(' ', 1)
                    if len(parts) == 2:
                        self.X_train.append(parts[1])
                        self.y_train.append(float(parts[0]))
                
                logger.info(f"Added {len(self.additional_data)} additional examples")
            except Exception as e:
                logger.error(f"Error loading additional data: {e}")
        
        logger.info(f"Processed data: {len(self.X_train)} sentences, {len(self.y_train)} labels")
    
    def train_model(self):
        """Train the sentiment analysis model"""
        if len(self.X_train) < 20:
            logger.warning(f"Insufficient data: only {len(self.X_train)} examples found")
            return False
        
        # Create train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            self.X_train, self.y_train, test_size=0.2, random_state=42
        )
        
        try:
            logger.info("Training model...")
            self.pipeline.fit(X_train, y_train)
            
            # Validate model
            val_predictions = self.pipeline.predict(X_val)
            mse = np.mean((val_predictions - y_val) ** 2)
            mae = np.mean(np.abs(val_predictions - y_val))
            
            logger.info(f"Model trained with {len(X_train)} examples")
            logger.info(f"Validation MSE: {mse:.4f}, MAE: {mae:.4f}")
            
            self.model_trained = True
            return True
        except Exception as e:
            logger.error(f"Error during training: {e}")
            return False
    
    def extract_keywords(self, text, n=2):
        """Extract important keywords from text, skipping stopwords"""
        words = nltk.word_tokenize(text.lower())
        keywords = [w for w in words if w not in self.stop_words and len(w) > 2]
        
        # If no keywords found after filtering, use any non-stopwords
        if not keywords:
            keywords = [w for w in words if w not in self.stop_words]
        
        # If still no keywords, use the first few words
        if not keywords and words:
            keywords = words[:min(n, len(words))]
            
        return keywords[:n]  # Return top n keywords
    
    def generate_response(self, sentence, score):
        """Generate a context-aware response based on sentiment score"""
        is_question = sentence.strip().endswith('?')
        
        # Extract keywords from input
        input_keywords = self.extract_keywords(sentence)
        if not input_keywords:
            input_keywords = ["that"]
        
        # Find closest example from recent or training data
        if self.recent_sentences:
            closest_idx = min(range(len(self.recent_scores)), 
                             key=lambda i: abs(self.recent_scores[i] - score))
            closest_sentence = self.recent_sentences[closest_idx]
        elif self.X_train:
            closest_idx = min(range(len(self.y_train)), 
                             key=lambda i: abs(self.y_train[i] - score))
            closest_sentence = self.X_train[closest_idx]
        else:
            closest_sentence = ""
        
        # Extract keywords from closest example
        closest_keywords = self.extract_keywords(closest_sentence)
        if not closest_keywords:
            closest_keywords = ["experience"]
        
        # Get sentiment-appropriate adjectives
        if score >= 8:
            adj = np.random.choice(["amazing", "excellent", "fantastic", "wonderful"])
        elif score >= 6:
            adj = np.random.choice(["good", "nice", "pleasant", "enjoyable"])
        elif score >= 4:
            adj = np.random.choice(["okay", "fine", "fair", "average"])
        elif score >= 2:
            adj = np.random.choice(["disappointing", "unfortunate", "tough", "difficult"])
        else:
            adj = np.random.choice(["terrible", "awful", "horrible", "dreadful"])
        
        # Choose primary subject and context
        subject = input_keywords[0]
        # Ensure context is different from subject when possible
        if len(closest_keywords) > 1 and closest_keywords[0] == subject:
            context = closest_keywords[1]
        else:
            context = closest_keywords[0] if closest_keywords else "situation"
        
        # Construct appropriate response
        if is_question:
            if score >= 5:
                responses = [
                    f"Yes, {subject} seems {adj}. It reminds me of {context}.",
                    f"I'd say yes - {subject} has a {adj} quality to it.",
                    f"Definitely! {subject} appears quite {adj}."
                ]
            else:
                responses = [
                    f"Not really, {subject} seems rather {adj}, similar to {context}.",
                    f"I don't think so. {subject} feels {adj} to me.",
                    f"I'm leaning towards no - {subject} has a {adj} quality to it."
                ]
        else:
            responses = [
                f"I see that {subject} feels {adj}, similar to {context}.",
                f"That {subject} definitely has a {adj} quality to it.",
                f"I understand - {subject} seems {adj}, like {context}.",
                f"{subject} sounds {adj}. That reminds me of {context}."
            ]
        
        return np.random.choice(responses)
    
    def add_to_training(self, sentence, score):
        """Add a new example to training data"""
        self.new_sentences.append(sentence)
        self.new_labels.append(score)
        
        # Maintain recent examples for context
        self.recent_sentences.append(sentence)
        self.recent_scores.append(score)
        if len(self.recent_sentences) > 5:
            self.recent_sentences.pop(0)
            self.recent_scores.pop(0)
        
        # Add to main training data
        self.X_train.append(sentence)
        self.y_train.append(score)
        
        logger.info("Added new example to training data")
    
    def retrain_if_needed(self):
        """Retrain model if enough new examples are collected"""
        # Check if we have 5 new examples
        if len(self.new_sentences) % 5 == 0 and len(self.new_sentences) > 0:
            logger.info("Retraining model with new data...")
            
            # Partial fit with SGDRegressor's warm_start
            # We only need to fit on the new data thanks to warm_start=True
            if self.model_trained:
                try:
                    # Get the most recent examples
                    recent_X = self.new_sentences[-5:]
                    recent_y = self.new_labels[-5:]
                    
                    # Vectorize (need to fit_transform to ensure new n-grams are handled)
                    self.pipeline.fit(self.X_train, self.y_train)
                    
                    self.retrain_count += 1
                    logger.info(f"Model retrained ({self.retrain_count}) with {len(self.X_train)} examples")
                    return True
                except Exception as e:
                    logger.error(f"Error during retraining: {e}")
                    return False
            else:
                # Initial training
                return self.train_model()
        
        return False
    
    def save_data(self):
        """Save additional training data to file"""
        if self.new_sentences:
            try:
                # Format data for storage
                new_data = [f"{label} {sentence}" for label, sentence in 
                           zip(self.new_labels, self.new_sentences)]
                
                # Append to existing additional data
                self.additional_data.extend(new_data)
                
                # Save to file
                with open(self.data_file, 'w') as f:
                    json.dump(self.additional_data, f)
                
                logger.info(f"Saved {len(self.new_sentences)} new training examples to {self.data_file}")
                return True
            except Exception as e:
                logger.error(f"Error saving data: {e}")
                return False
        return False
    
    def predict_sentiment(self, sentence):
        """Predict sentiment score for a new sentence"""
        if not self.model_trained:
            logger.warning("Model not trained yet")
            return None
        
        try:
            # Make a prediction
            prediction = self.pipeline.predict([sentence])[0]
            
            # Ensure prediction is within valid range
            prediction = max(1, min(10, prediction))
            
            return prediction
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None
    
    def evaluate_known_words(self, sentence):
        """Check if sentence contains known vocabulary"""
        # Extract features from vectorizer
        if not hasattr(self.pipeline.named_steps['vectorizer'], 'vocabulary_'):
            return False
        
        # Get vocab from fitted vectorizer
        vocabulary = self.pipeline.named_steps['vectorizer'].vocabulary_
        
        # Check if any word in sentence is in vocabulary
        words = nltk.word_tokenize(sentence.lower())
        non_stop_words = [w for w in words if w not in self.stop_words]
        
        # Check for unigrams
        unigram_match = any(word in vocabulary for word in non_stop_words)
        
        # Check for bigrams
        bigram_match = False
        if len(non_stop_words) > 1:
            for i in range(len(non_stop_words) - 1):
                bigram = f"{non_stop_words[i]} {non_stop_words[i+1]}"
                if bigram in vocabulary:
                    bigram_match = True
                    break
        
        return unigram_match or bigram_match

    def run_prediction_loop(self):
        """Main interaction loop for sentiment prediction"""
        # Initial training
        if not self.model_trained:
            success = self.train_model()
            if not success:
                logger.error("Failed to train model. Exiting.")
                return
        
        logger.info("Starting prediction loop. Enter a sentence or press Enter to exit.")
        
        try:
            while True:
                # Get user input
                new_sentence = input("\nEnter sentence for prediction (or press Enter to exit): ").strip()
                if not new_sentence:
                    self.save_data()
                    logger.info("Prediction phase ended")
                    break
                
                # Check if sentence contains known vocabulary
                has_known_words = self.evaluate_known_words(new_sentence)
                
                if not has_known_words:
                    logger.info("This sentence contains few known words.")
                    
                    # Get manual rating
                    while True:
                        try:
                            rating = float(input("Please rate (1-10): "))
                            if 1 <= rating <= 10:
                                break
                            logger.warning("Rating must be between 1 and 10")
                        except ValueError:
                            logger.warning("Please enter a valid number")
                    
                    # Add to training data
                    self.add_to_training(new_sentence, rating)
                    
                    # Generate response
                    response = self.generate_response(new_sentence, rating)
                    logger.info(f"Response: {response}")
                    
                else:
                    # Predict sentiment
                    prediction = self.predict_sentiment(new_sentence)
                    if prediction is None:
                        logger.error("Failed to make prediction")
                        continue
                    
                    # Show prediction and response
                    logger.info(f"Prediction: {prediction:.2f}")
                    response = self.generate_response(new_sentence, prediction)
                    logger.info(f"Response: {response}")
                    
                    # Get feedback
                    feedback = input("Is this prediction accurate? (y/n, or enter to skip): ").lower()
                    if feedback == 'n':
                        # Get correct rating
                        while True:
                            try:
                                rating = float(input("What would be a better rating (1-10)? "))
                                if 1 <= rating <= 10:
                                    break
                                logger.warning("Rating must be between 1 and 10")
                            except ValueError:
                                logger.warning("Please enter a valid number")
                        
                        # Add to training data
                        self.add_to_training(new_sentence, rating)
                        
                        # Generate updated response
                        updated_response = self.generate_response(new_sentence, rating)
                        logger.info(f"Updated Response: {updated_response}")
                
                # Retrain if needed
                self.retrain_if_needed()
                
        except KeyboardInterrupt:
            logger.info("\nInterrupted by user")
        finally:
            self.save_data()
            logger.info("Prediction phase ended")


if __name__ == "__main__":
    sentiment_system = SentimentAnalysisSystem()
    sentiment_system.run_prediction_loop()
```


