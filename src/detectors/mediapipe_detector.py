"""
MediaPipe-based hand detection
"""
import cv2
import mediapipe as mp
import numpy as np
from .hand_detector_base import HandDetectorBase
from ..core.config import MP_MODEL_COMPLEXITY, MP_MIN_DETECTION_CONFIDENCE, MP_MIN_TRACKING_CONFIDENCE

class MediaPipeDetector(HandDetectorBase):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            model_complexity=MP_MODEL_COMPLEXITY,
            min_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MP_MIN_TRACKING_CONFIDENCE,
            max_num_hands=1
        )
    
    def process_frame(self, frame):
        """Process frame using MediaPipe"""
        h, w = frame.shape[:2]
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        # Prepare output
        output = {
            'detected': False,
            'hand_x': 0.5,
            'hand_y': 0.5,
            'finger_count': 0,
            'annotated_frame': frame.copy()
        }
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw landmarks
            self.mp_drawing.draw_landmarks(
                output['annotated_frame'],
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Get thumb tip position (normalized 0-1)
            thumb_tip = hand_landmarks.landmark[4]
            output['hand_x'] = thumb_tip.x
            output['hand_y'] = thumb_tip.y
            
            # Draw cursor indicator
            cx = int(thumb_tip.x * w)
            cy = int(thumb_tip.y * h)
            cv2.circle(output['annotated_frame'], (cx, cy), 10, (0, 255, 0), -1)
            
            # Count extended fingers
            output['finger_count'] = self._count_fingers(hand_landmarks)
            output['detected'] = True
            
            # Add finger count text
            cv2.putText(output['annotated_frame'], f"Fingers: {output['finger_count']}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            cv2.putText(output['annotated_frame'], "Hand not detected", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return output
    
    def _count_fingers(self, hand_landmarks):
        finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
        thumb_tip = 4
        finger_count = 0
        
        # Check thumb (different logic)
        if hand_landmarks.landmark[thumb_tip].x < hand_landmarks.landmark[thumb_tip - 1].x:
            finger_count += 1
        
        # Check other fingers
        for tip_id in finger_tips:
            if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
                finger_count += 1
        
        return finger_count
    
    def detect_gesture(self, hand_landmarks):
        def distance(p1, p2):
            return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
        
        thumb = hand_landmarks.landmark[4]
        index = hand_landmarks.landmark[8]
        middle = hand_landmarks.landmark[12]
        ring = hand_landmarks.landmark[16]
        pinky = hand_landmarks.landmark[20]
        
        touch_threshold = 0.05
        
        thumb_index = distance(thumb, index) < touch_threshold
        thumb_middle = distance(thumb, middle) < touch_threshold
        thumb_ring = distance(thumb, ring) < touch_threshold
        thumb_pinky = distance(thumb, pinky) < touch_threshold
        index_middle = distance(index, middle) < touch_threshold
        middle_ring = distance(middle, ring) < touch_threshold
        
        if thumb_index:
            return "Draw"
        elif thumb_middle:
            return "Erase"
        elif thumb_ring:
            return "Cycle Color"
        elif thumb_pinky:
            return "Clear"
        elif index_middle:
            return "Increase Size"
        elif middle_ring:
            return "Decrease Size"
        else:
            return "None"
    
    def cleanup(self):
        """Release MediaPipe resources"""
        try:
            self.hands.close()
        except:
            pass  # Ignore errors during cleanup
