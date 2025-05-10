import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import io
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
import requests
import os
import time

if not os.path.exists('history'):
    os.mkdir('history')

# Load the BlazeFace model for face detection
base_options = python.BaseOptions(model_asset_path="blaze_face_short_range.tflite")
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

# Placeholder for face detection and cropping
def detect_and_crop_face(image):
    """
    Detect faces in the image and crop the first detected face.
    
    Args:
        image: PIL Image object
        
    Returns:
        - cropped_face: The cropped face image or None if no face detected
        - face_detected: Boolean indicating if a face was detected
    """
    # save the image to history folder with current timestamp
    timestamp = int(time.time())
    image.save(f'history/{timestamp}.jpg')

    # Convert PIL Image to OpenCV format
    img_cv = np.array(image)


    if len(img_cv.shape) == 3 and img_cv.shape[2] == 4:  # RGBA
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2RGB)
    
    # Convert to MediaPipe Image format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_cv)
    
    # Face detection
    detection_result = detector.detect(mp_image)
    
    # Check if faces were detected
    face_detected = len(detection_result.detections) > 0
    
    if face_detected:
        detection = detection_result.detections[0]
        bbox = detection.bounding_box

        cropped_face = img_cv[bbox.origin_y:bbox.origin_y + bbox.height,
                              bbox.origin_x:bbox.origin_x + bbox.width]
    
        # Convert back to PIL for Gradio
        cropped_face = Image.fromarray(cropped_face)
    else:
        cropped_face = None
    
    return cropped_face, face_detected


# Placeholder for beauty score prediction
def predict_beauty_score(face_image):
    """
    Predict beauty score for the given face image.
    
    In a real implementation, you would replace this with your actual beauty score
    prediction model.
    
    Args:
        face_image: Image containing a face
        
    Returns:
        - score: Overall beauty score (1-5)
        - std_dev: Standard deviation of the score
        - probabilities: Probabilities for each score from 1-5
    """
    if isinstance(face_image, Image.Image):
        face_image_np = np.array(face_image)
        # Convert RGBA to RGB if needed
        if len(face_image_np.shape) == 3 and face_image_np.shape[2] == 4:
            face_image_np = cv2.cvtColor(face_image_np, cv2.COLOR_RGBA2RGB)
    else:
        # Assume it's already a numpy array
        face_image_np = face_image


    _, img_encoded = cv2.imencode('.jpg', face_image_np)

    # Convert to BytesIO object
    img_bytes = io.BytesIO(img_encoded.tobytes())

    # Prepare files dictionary for requests
    files = {'image': ('image.jpg', img_bytes, 'image/jpeg')}

    # Send request
    url = 'http://localhost:5001/predict'
    response = requests.post(url, files=files)
    prediction = np.array(response.json()['prediction'])
    
    # Calculate the weighted score (expected value)
    scores = np.array([1, 2, 3, 4, 5])
    weighted_score = np.sum(prediction * scores)
    
    # Calculate standard deviation
    std_dev = np.sqrt(np.sum(prediction * (scores - weighted_score) ** 2))
    
    return weighted_score, std_dev, prediction

# Create a bar plot for the beauty score probabilities
def create_probability_plot(probabilities):
    """
    Create a bar plot showing probabilities for each beauty score (1-5).
    
    Args:
        probabilities: Array of 5 probability values (should sum to 1)
        
    Returns:
        PIL Image containing the plot
    """
    plt.figure(figsize=(10, 5))
    
    # Create the bar chart
    bars = plt.bar(
        [1, 2, 3, 4, 5], 
        probabilities, 
        color='skyblue', 
        edgecolor='navy', 
        width=0.6
    )
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2., 
            height + 0.01,
            f'{height:.2f}', 
            ha='center', 
            va='bottom', 
            fontweight='bold'
        )
    
    # Add labels and formatting
    plt.xlabel('Beauty Score', fontsize=12, fontweight='bold')
    plt.ylabel('Probability', fontsize=12, fontweight='bold')
    plt.title('Beauty Score Distribution', fontsize=14, fontweight='bold')
    plt.ylim(0, max(probabilities) * 1.2)  # Add space for text labels
    plt.xticks([1, 2, 3, 4, 5], fontsize=11)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add a horizontal line at the expected score
    expected_score = np.sum(np.array([1, 2, 3, 4, 5]) * probabilities)
    plt.axvline(x=expected_score, color='red', linestyle='--', alpha=0.7)
    plt.text(
        expected_score + 0.1, 
        max(probabilities) * 0.5,
        f'Mean: {expected_score:.2f}', 
        color='red', 
        fontweight='bold'
    )
    
    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    # Convert to PIL Image
    plot_image = Image.open(buf)
    return plot_image

# Process the uploaded image
def process_image(image, force_prediction=False):
    """
    Main processing function for the uploaded image.
    
    Args:
        image: Input image (PIL Image)
        force_prediction: Boolean flag to force prediction even if no face is detected
    
    Returns:
        - cropped_face: Cropped face image or None
        - plot_image: Probability distribution plot or None
        - result_text: Text describing the results or warning
        - face_detected: Boolean indicating if a face was detected
    """
    if image is None:
        return None, None, "Please upload an image.", False
    
    # Step 1: Detect and crop face
    cropped_face, face_detected = detect_and_crop_face(image)
    
    if face_detected or force_prediction:
        # Step 2: Run beauty score prediction
        face_image = image if not face_detected else cropped_face
        score, std_dev, probabilities = predict_beauty_score(face_image)
        
        # Step 3: Create visualization
        plot_image = create_probability_plot(probabilities)
        
        # Step 4: Format result text
        result_text = f"Beauty Score: {score:.2f} (±{std_dev:.2f})"
        
        if not face_detected and force_prediction:
            result_text = "⚠️ WARNING: No face detected. Forced prediction.\n" + result_text
        
        return cropped_face, plot_image, result_text, face_detected
    else:
        # No face detected and not forcing prediction
        return None, None, "⚠️ No face detected in the image. Please try another image or use the 'Force Prediction' button.", False

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="Facial Beauty Score Analyzer") as demo:
        gr.Markdown("# 👤 Facial Beauty Score Analyzer")
        gr.Markdown("""
        Upload a photo to analyze facial beauty score. The model will:
        1. Detect and crop the face
        2. Analyze the beauty score (1-5 scale)
        3. Display the probability distribution and overall score
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input components
                input_image = gr.Image(label="Upload Photo", type="pil")
                with gr.Row():
                    submit_btn = gr.Button("Analyze", variant="primary")
                    clear_btn = gr.Button("Clear")
                
            with gr.Column(scale=2):
                # Output components
                with gr.Group(visible=False) as no_face_warning:
                    gr.Markdown("### ⚠️ No face detected!")
                    gr.Markdown("""
                    The photo you uploaded doesn't appear to contain a face that our model can detect.
                    Do you want to force the beauty prediction model to run anyway?
                    """)
                    force_pred_btn = gr.Button("Force Prediction Anyway", variant="secondary")
                
                with gr.Row():
                    cropped_face = gr.Image(label="Detected Face", visible=True)
                
                result_text = gr.Textbox(label="Results")
                prob_plot = gr.Image(label="Beauty Score Distribution")
        
        # Define process flow
        def on_analyze(image):
            if image is None:
                return gr.update(visible=False), None, "Please upload an image.", None
            
            cropped, plot, text, face_detected = process_image(image)
            
            if not face_detected:
                return gr.update(visible=True), None, text, None
            else:
                return gr.update(visible=False), cropped, text, plot
        
        def on_force_prediction(image):
            cropped, plot, text, _ = process_image(image, force_prediction=True)
            return gr.update(visible=False), image, text, plot  # Show original image since no face was cropped
        
        def on_clear():
            return None, gr.update(visible=False), None, None, None
        
        # Connect events
        submit_btn.click(
            fn=on_analyze,
            inputs=[input_image],
            outputs=[no_face_warning, cropped_face, result_text, prob_plot]
        )
        
        force_pred_btn.click(
            fn=on_force_prediction,
            inputs=[input_image],
            outputs=[no_face_warning, cropped_face, result_text, prob_plot]
        )
        
        clear_btn.click(
            fn=on_clear,
            inputs=[],
            outputs=[input_image, no_face_warning, cropped_face, result_text, prob_plot]
        )
    
    return demo

# Launch the Gradio app
if __name__ == "__main__":
    demo = create_interface()
    demo.launch()