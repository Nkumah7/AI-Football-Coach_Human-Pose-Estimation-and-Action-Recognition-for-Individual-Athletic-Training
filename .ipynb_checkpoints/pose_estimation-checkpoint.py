import cv2

def get_pose(vid):
    cap = cv2.VideoCapture(vid)    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
    
            cv2.imshow("Feed", frame)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
    