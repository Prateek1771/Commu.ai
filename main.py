import cv2
import dlib
import mediapipe as mp
import speech_recognition as sr
import matplotlib.pyplot as plt
import numpy as np
import threading
import time

# Initialize components
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils
recognizer = sr.Recognizer()

# Global variable to store audio analysis results
audio_analysis = {"wpm": 0}

# Track eye movements
def detect_eyes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    eye_positions = []
    for face in faces:
        landmarks = predictor(gray, face)
        left_eye = (landmarks.part(36).x, landmarks.part(36).y)
        right_eye = (landmarks.part(45).x, landmarks.part(45).y)
        eye_positions.append((left_eye, right_eye))
    return eye_positions

# Track hand movements
def detect_hands(frame_rgb):
    results = hands.process(frame_rgb)
    hand_positions = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            hand_positions.append(hand_landmarks)
    return hand_positions

# Analyze audio
def analyze_audio(duration):
    global audio_analysis
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Recording audio for {} seconds...".format(duration))
        audio = recognizer.listen(source, timeout=duration, phrase_time_limit=duration)
        try:
            text = recognizer.recognize_google(audio)
            print("Transcription: ", text)
            # Calculate words per minute
            words = len(text.split())
            wpm = (words / duration) * 60
            print("Words per minute: ", wpm)
            audio_analysis = {"wpm": wpm}  # Only using WPM for analysis
        except sr.UnknownValueError:
            print("Audio not clear")
            audio_analysis = {"wpm": 0}
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))
            audio_analysis = {"wpm": 0}

# Feedback mechanism
def provide_feedback(eye_contact_counter, hand_gesture_counter, frame_count, audio_analysis):
    feedback = []
    if eye_contact_counter == 0:
        feedback.append("Improve eye contact.")
    if hand_gesture_counter == 0:
        feedback.append("Increase hand gestures.")
    if audio_analysis["wpm"] < 100:  # Threshold for WPM
        feedback.append("Increase speaking speed.")
    return feedback

# Display results
def display_results(eye_score, hand_score, audio_score):
    labels = ['Eye Contact', 'Hand Gestures', 'Speaking Speed']
    scores = [eye_score, hand_score, audio_score]
    
    fig, ax = plt.subplots()
    bars = ax.bar(labels, scores, color=['blue', 'green', 'red'])
    ax.set_ylim(0, 100)
    ax.set_ylabel('Score (%)')
    ax.set_title('Communication Skills Analysis')
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 2, round(yval, 2), ha='center', va='bottom')
    
    plt.show()

# Video recording function
def record_video(duration, frame_rate):
    global eye_contact_counter, hand_gesture_counter, frame_count
    eye_contact_counter = 0
    hand_gesture_counter = 0
    frame_count = 0
    
    cap = cv2.VideoCapture(0)
    start_time = time.time()

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        eye_positions = detect_eyes(frame)
        if eye_positions:
            eye_contact_counter += 1
        
        hand_positions = detect_hands(frame_rgb)
        if hand_positions:
            hand_gesture_counter += 1
        
        cv2.imshow('Communication Helper', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
        time.sleep(1 / frame_rate)  # Maintain the frame rate

    cap.release()
    cv2.destroyAllWindows()

# Main function to integrate all components
def main():
    duration = 20  # Duration for audio and video recording in seconds
    frame_rate = 30  # Frame rate for video recording
    
    print("Starting recording...")

    # Create threads for audio and video recording
    audio_thread = threading.Thread(target=analyze_audio, args=(duration,))
    video_thread = threading.Thread(target=record_video, args=(duration, frame_rate))

    # Start the threads
    audio_thread.start()
    video_thread.start()

    # Wait for both threads to complete
    audio_thread.join()
    video_thread.join()

    # Calculate scores
    eye_score = (eye_contact_counter / frame_count) * 100
    hand_score = (hand_gesture_counter / frame_count) * 100
    audio_score = (audio_analysis["wpm"] / 150) * 100  # Assuming 200 WPM is the ideal speaking speed
    
    # Provide feedback
    feedback = provide_feedback(eye_contact_counter, hand_gesture_counter, frame_count, audio_analysis)
    print("Feedback:", feedback)
    
    # Display results
    display_results(eye_score, hand_score, audio_score)

if __name__ == "__main__":
    main()