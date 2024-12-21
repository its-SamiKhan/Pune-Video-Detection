import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

class VideoProcessor:
    def __init__(self):
        # Initialize YOLO model for person detection
        self.model = YOLO('yolov8n.pt')
        self.storage_path = "static/heatmaps/"
        
    def process_stream(self, video_url, zones):
        cap = cv2.VideoCapture(video_url)
        
        # Check if the video capture opened successfully
        if not cap.isOpened():
            print(f"Error: Unable to open video stream {video_url}")
            return {'total_count': 0, 'zone_counts': {}}
        
        total_count = 0
        zone_counts = {}
        
        # Initialize zone counts
        for zone in zones:
            zone_counts[zone['zone_id']] = 0
            
        heatmap_data = []  # To store detection positions for heatmap
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video stream or error reading frame.")
                break
                
            # Run YOLO detection
            results = self.model(frame, classes=[0])  # 0 is the class ID for person
            
            # Process detections
            detections = results[0].boxes  # Access the boxes directly
            
            # Update counts
            total_count = len(detections)
            
            # Check zones and collect heatmap data
            for zone in zones:
                zone_count = self._count_in_zone(detections, zone['coordinates'])
                zone_counts[zone['zone_id']] = zone_count
            
            # Draw bounding boxes and collect heatmap data
            for det in detections:
                x1, y1, x2, y2 = map(int, det.xyxy[0])  # Get bounding box coordinates
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box
                heatmap_data.append(((x1 + x2) // 2, (y1 + y2) // 2))  # Collect center points for heatmap
            
            # Display footfall count on the frame
            cv2.putText(frame, f'Total Footfall: {total_count}', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Generate and overlay heatmap
            if heatmap_data:
                self.generate_heatmap(heatmap_data, frame.shape[1], frame.shape[0])
                heatmap_image = cv2.imread(self.storage_path + "latest_heatmap.png")
                heatmap_image = cv2.resize(heatmap_image, (frame.shape[1], frame.shape[0]))
                frame = cv2.addWeighted(frame, 0.6, heatmap_image, 0.4, 0)  # Overlay heatmap
            
            # Show the frame with footfall count and heatmap
            cv2.imshow('Footfall Detection', frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
        return {
            'total_count': total_count,
            'zone_counts': zone_counts
        }
        
    def _count_in_zone(self, detections, coordinates):
        count = 0
        for det in detections:
            x, y = det.xyxy[0][0], det.xyxy[0][1]  # center coordinates of detection
            if (coordinates['x_min'] <= x <= coordinates['x_max'] and
                coordinates['y_min'] <= y <= coordinates['y_max']):
                count += 1
        return count

    def generate_heatmap(self, heatmap_data, width, height):
        # Create a blank heatmap
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        # Populate the heatmap with detection points
        for x, y in heatmap_data:
            heatmap[y, x] += 1  # Increment the heatmap at the detection point
        
        # Normalize the heatmap
        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = np.uint8(heatmap)
        
        # Create a color map
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Save the heatmap
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"{self.storage_path}latest_heatmap.png", heatmap_color)