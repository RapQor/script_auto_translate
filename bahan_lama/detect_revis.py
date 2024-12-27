import cv2
import numpy as np

def draw_text_boxes(image_path):
    # Read the image
    image = cv2.imread(image_path)
    output = image.copy()
    
    # Define the coordinates for each text region
    # Format: [x, y, width, height]
    text_regions = {
        # Japanese title at the top
        'top_title': [(260, 90, 1370, 190)],
        
        # Japanese text on left side
        'left_text': [(100, 280, 130, 2000)],
        
        # Left speech bubble text
        'left_bubble': [(270, 1160, 490, 700)],
        
        # # Right speech bubble texts (2 separate boxes)
        'right_bubble': [
            (1760, 1130, 470, 650),
            (1455, 1500, 425, 620)
        ],
        
        # # Bottom speech bubbles (2 separate boxes)
        'bottom_bubbles': [
            (150, 2740, 595, 840),
            (1800, 2650, 525, 845),
            (1320, 2920, 490, 745),
        ]
    }
    
    # Colors for different regions
    colors = {
        'top_title': (255, 0, 0),      # Blue
        'left_text': (0, 255, 0),      # Green
        'left_bubble': (0, 0, 255),    # Red
        'right_bubble': (255, 255, 0),  # Cyan
        'bottom_bubbles': (255, 0, 255) # Magenta
    }
    
    # Draw rectangles for each region
    for region_name, boxes in text_regions.items():
        color = colors[region_name]
        for box in boxes:
            x, y, w, h = box
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 5)
            # Add label
            cv2.putText(output, region_name, (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    
    # Save the output image
    output_path = 'manga_text_detection.jpg'
    cv2.imwrite(output_path, output)
    return output_path

# Function to run the detection

# Usage
if __name__ == "__main__":
    image_path = "./temp_images/page_2.jpg"  # Replace with your image path
    draw_text_boxes(image_path)