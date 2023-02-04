import os
import cv2
import json
import time
import math
import copy
import numpy as np
import pandas as pd
import mediapipe as mp

def find_pose(img, pose, mp_pose):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    coords = {}
    
    if results.pose_landmarks:            
        # Knee
        right_knee_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x
        right_knee_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y
        left_knee_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x
        left_knee_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y
        
        right_knee_cor = right_knee_x, right_knee_y
        left_knee_cor = left_knee_x, left_knee_y
        
        coords.update({
            "knee coordinates": {
                "right knee": (right_knee_x, right_knee_y),
                "left knee": (left_knee_x, left_knee_y)
            }
        })
        
    imgBGR = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)
    return img, results, coords

def find_position(img, results):
    pose_list = np.array([[res.x, res.y] for res in results.pose_landmarks.landmark] if results.pose_landmarks else np.zeros((32, 2)))              
    return pose_list

def find_angle(img, p1, p2, p3, pose_list):
    
    # Get the landmarks
    x1, y1 = pose_list[p1]
    x2, y2 = pose_list[p2]
    x3, y3 = pose_list[p3]
    
    # Calculate the Angle
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360    
    return angle

def pose_detector(vid, num_frames=15):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    count_frames = 0
    cap = cv2.VideoCapture(vid)
    joint_angles = {
        "right knee": [],
        "left knee": []
                        }
    
    i = 0
    with pose:
        while True:
            success, image = cap.read()
            image = cv2.resize(image, (649, 400))
            if not success:
                print("Ignoring empty camera frame.")
                break             
            
            image, results, coords = find_pose(image, pose, mp_pose)
            pose_list = find_position(image, results)
           
            if len(pose_list) != 0:                
                # Right leg
                right_knee_angles = find_angle(image, 24, 26, 28, pose_list)           
                joint_angles["right knee"].append(int(right_knee_angles))
                
                # Left leg
                left_knee_angles = find_angle(image, 23, 25, 29, pose_list)
                joint_angles["left knee"].append(int(left_knee_angles))           
        
            count_frames += 1
            i += 1
            
            if num_frames == count_frames:                
                break            
            
            # Show to screen - Uncomment to see video
            # cv2.imshow('Feed', image)
            
            # Break - End video with letter q on keyboard - Uncomment if you are watching video
            # if cv2.waitKey(10) & 0xFF == ord('q'):
            #     break      
           
    
    cap.release()
    cv2.destroyAllWindows()
    
    return joint_angles, coords

x, y = pose_detector(f"videos/dribbling.mp4", )
print(x)
# def get_joint_angles_and_coords(path):    
#     players = {}
#     x_y_cors = {}
#     dir_list = os.listdir(path)
    
#     for i, vid in enumerate(dir_list):    
#         joint_angles, coords = pose_detector(f"{path}/{vid}", num_frames=num_frames)
#         players[f"joint_angles_{i+1}"] = joint_angles
#         x_y_cors[f"joint_cors_{i+1}"] = coords
        
#     return players, x_y_cors

# def get_joint_angle_sequences(joint_angles):    
#     players_joint_angles = []
    
#     for i, joint_angle in enumerate(joint_angles):
#         joint_angle = joint_angles[joint_angle]
#         player_joint_angles = []
   
#         for j in range(num_frames):            
#             player_joint_angles.extend([joint_angle['right knee'][j], joint_angle['left knee'][j]])
#         players_joint_angles.append(player_joint_angles) 
        
#     return np.array(players_joint_angles)

