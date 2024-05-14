import pickle
import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: ' ', 27: 'O'
}

def open_camera():
    for i in range(5):  
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera opened successfully at index {i}")
            return cap
    print("Error: Failed to open webcam")
    return None

def capture_and_display(cap):
    recognized_letters = []
    sentence = ""

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame")
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  
                    hand_landmarks,  
                    mp_hands.HAND_CONNECTIONS,  
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []
                x_ = []
                y_ = []

                for landmark in hand_landmarks.landmark:
                    x = landmark.x
                    y = landmark.y
                    x_.append(x)
                    y_.append(y)

                for landmark in hand_landmarks.landmark:
                    x = landmark.x
                    y = landmark.y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                prediction = model.predict([np.asarray(data_aux)])
                current_letter = labels_dict[int(prediction[0])]

                letters_text = " ".join(recognized_letters)
                sentence_text = f"Final Sentence: {sentence}"
                cv2.putText(frame, f"Recognized Letters: {letters_text}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, sentence_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow('frame', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    recognized_letters.append(current_letter)
                    cv2.imshow('frame', np.zeros((H, W, 3), np.uint8))
                elif key == ord('q'):
                    break

        if len(recognized_letters) > 0 and all(char in labels_dict.values() for char in recognized_letters):
            sentence = "".join(recognized_letters)

        if len(recognized_letters) == 26 or key == ord('q'):
            break

    return sentence

def main():
    # Open webcam
    cap = open_camera()
    if not cap:
        return

    print("Get ready to show hand gestures. Press SPACEBAR to capture each letter.")

    while True:
        ret, frame = cap.read()
        cv2.putText(frame, "Press SPACEBAR to start", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

    final_sentence = capture_and_display(cap)
    print("Final Sentence:", final_sentence)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
