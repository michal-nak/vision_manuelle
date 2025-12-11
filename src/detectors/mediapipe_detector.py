"""
MediaPipe-based hand detection
"""
import cv2
import mediapipe as mp
import numpy as np
from .hand_detector_base import HandDetectorBase
from ..core.config import MP_MODEL_COMPLEXITY, MP_MIN_DETECTION_CONFIDENCE, MP_MIN_TRACKING_CONFIDENCE

class MediaPipeDetector(HandDetectorBase):
    def __init__(self, show_debug=False):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Use static_image_mode=True to avoid timestamp conflicts
        # This treats each frame independently which is more stable
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            model_complexity=MP_MODEL_COMPLEXITY,
            min_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
            max_num_hands=1
        )
        
        self.last_landmarks = None
        self.show_debug_overlay = show_debug
        self.last_result = None
    
    def process_frame(self, frame, use_palm_center=False):
        """Process frame using MediaPipe
        
        Args:
            frame: Input frame
            use_palm_center: If True, use palm center for hand_x/hand_y instead of thumb tip.
                           Used during calibration/optimization for better alignment with CV detector.
        """
        h, w = frame.shape[:2]
        
        # Convert to RGB for MediaPipe (writeable flag improves performance)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        try:
            results = self.hands.process(rgb_frame)
            rgb_frame.flags.writeable = True
        except Exception as e:
            # Handle timestamp mismatch errors by returning last known result or empty result
            error_msg = str(e).lower()
            if "timestamp" in error_msg or "invalid_argument" in error_msg:
                # Return last result if available, otherwise return empty detection
                if self.last_result is not None:
                    return self.last_result
                else:
                    # Return empty result if no previous frame available
                    return {
                        'detected': False,
                        'hand_x': 0.5,
                        'hand_y': 0.5,
                        'finger_count': 0,
                        'gesture': 'None',
                        'annotated_frame': frame.copy()
                    }
            # For other errors, re-raise
            raise
        
        # Prepare output
        output = {
            'detected': False,
            'hand_x': 0.5,
            'hand_y': 0.5,
            'finger_count': 0,
            'gesture': 'None',
            'annotated_frame': frame.copy()
        }
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            self.last_landmarks = hand_landmarks
            
            # Draw landmarks
            self.mp_drawing.draw_landmarks(
                output['annotated_frame'],
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Get hand position (normalized 0-1)
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            
            if use_palm_center:
                # Use palm center for calibration/optimization (better alignment with CV detector)
                wrist = hand_landmarks.landmark[0]
                middle_mcp = hand_landmarks.landmark[9]  # Middle finger base
                output['hand_x'] = (wrist.x + middle_mcp.x) / 2
                output['hand_y'] = (wrist.y + middle_mcp.y) / 2
                
                # Draw palm center indicator
                cx = int(output['hand_x'] * w)
                cy = int(output['hand_y'] * h)
                cv2.circle(output['annotated_frame'], (cx, cy), 15, (255, 165, 0), 3)  # Orange for palm
                cv2.circle(output['annotated_frame'], (cx, cy), 3, (255, 165, 0), -1)
            else:
                # Use thumb tip for drawing (default)
                output['hand_x'] = thumb_tip.x
                output['hand_y'] = thumb_tip.y
                
                # Draw large thumb cursor indicator
                cx = int(thumb_tip.x * w)
                cy = int(thumb_tip.y * h)
                cv2.circle(output['annotated_frame'], (cx, cy), 15, (0, 255, 0), 3)
                cv2.circle(output['annotated_frame'], (cx, cy), 3, (0, 255, 0), -1)
            
            # Draw index finger for reference
            ix = int(index_tip.x * w)
            iy = int(index_tip.y * h)
            cv2.circle(output['annotated_frame'], (ix, iy), 10, (255, 0, 255), 2)
            
            # Count extended fingers
            output['finger_count'] = self._count_fingers(hand_landmarks)
            
            # Detect gesture
            output['gesture'] = self.detect_gesture(hand_landmarks)
            
            output['detected'] = True
            
            # Add debug overlays if enabled
            if self.show_debug_overlay:
                # Confidence: MediaPipe doesn't provide per-frame confidence, so we show 100% when detected
                cv2.putText(output['annotated_frame'], "Detection: MediaPipe", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(output['annotated_frame'], f"Status: DETECTED", 
                           (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(output['annotated_frame'], f"Confidence: 100%", 
                           (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(output['annotated_frame'], f"Fingers: {output['finger_count']}", 
                           (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(output['annotated_frame'], f"Gesture: {output['gesture']}", 
                           (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                position_label = "Palm Center" if use_palm_center else "Thumb Tip"
                color_label = "Orange" if use_palm_center else "Green"
                cv2.putText(output['annotated_frame'], f"Hand Center: ({output['hand_x']:.2f}, {output['hand_y']:.2f})", 
                           (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.putText(output['annotated_frame'], f"{position_label} ({color_label} Circle)", 
                           (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            if self.show_debug_overlay:
                cv2.putText(output['annotated_frame'], "Detection: MediaPipe", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(output['annotated_frame'], "Status: NO HAND DETECTED", 
                           (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Store result for timestamp error recovery
        self.last_result = output
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
