import gradio as gr
from deepface import DeepFace
import numpy as np
import cv2
from PIL import Image
import base64
from io import BytesIO

def numpy_to_base64(img_array):
    """Convert numpy array to base64 string for HTML display"""
    if img_array is None:
        return None
    img = Image.fromarray(img_array.astype('uint8'))
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def detect_and_align_face(img, detector_backend='opencv'):
    """Detect and align face from image using DeepFace"""
    try:
        # Use DeepFace.extract_faces to detect and align
        face_objs = DeepFace.extract_faces(
            img_path=img,
            detector_backend=detector_backend,
            align=True,
            enforce_detection=True
        )
        
        if len(face_objs) == 0:
            return None, None, None
        
        # Get the first face
        face_obj = face_objs[0]
        detected_face = face_obj['face']  # Aligned face array
        facial_area = face_obj['facial_area']  # Face coordinates
        confidence = face_obj['confidence']
        
        # Convert detected face back to uint8 for visualization
        if detected_face.max() <= 1.0:
            detected_face = (detected_face * 255).astype('uint8')
        
        # Draw rectangle on original image
        img_with_box = img.copy()
        x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
        cv2.rectangle(img_with_box, (x, y), (x+w, y+h), (0, 255, 0), 3)
        
        # Add label
        label = f"Face Detected ({confidence:.1%})"
        cv2.putText(img_with_box, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return detected_face, img_with_box, facial_area
    except Exception as e:
        return None, None, str(e)

def preprocess_face(face_img, target_size=(224, 224)):
    """Preprocess face for model input"""
    try:
        # Ensure face_img is in proper format
        if face_img.max() <= 1.0:
            face_img = (face_img * 255).astype('uint8')
        
        # Resize
        resized = cv2.resize(face_img, target_size)
        
        # Normalize to [0, 1]
        normalized = resized.astype('float32') / 255.0
        
        # Convert to visualization (scale back for display)
        normalized_vis = (normalized * 255).astype('uint8')
        
        return normalized, normalized_vis
    except Exception as e:
        return None, None

def create_embedding_visualization(embedding, img_shape=(400, 400)):
    """Create a visual representation of the embedding vector"""
    try:
        # Reshape embedding for visualization
        emb_len = len(embedding)
        
        # Create a grid visualization
        grid_size = int(np.ceil(np.sqrt(emb_len)))
        grid = np.zeros((grid_size, grid_size))
        
        # Normalize embedding to [0, 1]
        emb_normalized = (embedding - embedding.min()) / (embedding.max() - embedding.min() + 1e-10)
        
        # Fill grid
        for i in range(min(emb_len, grid_size * grid_size)):
            row = i // grid_size
            col = i % grid_size
            grid[row, col] = emb_normalized[i]
        
        # Scale to image size
        grid_img = cv2.resize(grid, img_shape, interpolation=cv2.INTER_NEAREST)
        
        # Apply colormap
        grid_colored = cv2.applyColorMap((grid_img * 255).astype('uint8'), cv2.COLORMAP_JET)
        grid_colored = cv2.cvtColor(grid_colored, cv2.COLOR_BGR2RGB)
        
        return grid_colored
    except Exception as e:
        return None

def verify_faces_with_visualization(nid_image, webcam_image, model_name):
    """
    Verify faces with complete step-by-step visualization
    """
    try:
        # Check if images are provided
        if nid_image is None or webcam_image is None:
            return (
                "❌ Error: Please upload both images",
                None, None, None, None, None, None, None, None, None
            )
        
        # Initialize outputs
        step_status = "<h3>🔄 Processing...</h3>"
        
        # STEP 1: Face Detection - NID Image
        step_status += "<div style='padding: 10px; background-color: #fff3cd; border-radius: 5px; margin: 10px 0;'>"
        step_status += "<b>Step 1: Face Detection (NID Image)</b><br>"
        step_status += "Detecting face in NID card image using face detector...<br>"
        
        nid_face, nid_detected_img, nid_area = detect_and_align_face(nid_image, detector_backend='opencv')
        
        if nid_face is None:
            return (
                "❌ Error: No face detected in NID image. Please ensure the image contains a clear, visible face.",
                None, None, None, None, None, None, None, None, None
            )
        
        step_status += f"✅ Face detected at position: x={nid_area['x']}, y={nid_area['y']}, width={nid_area['w']}, height={nid_area['h']}</div>"
        
        # STEP 2: Face Detection - Webcam Image
        step_status += "<div style='padding: 10px; background-color: #fff3cd; border-radius: 5px; margin: 10px 0;'>"
        step_status += "<b>Step 2: Face Detection (Webcam Image)</b><br>"
        step_status += "Detecting face in webcam image...<br>"
        
        webcam_face, webcam_detected_img, webcam_area = detect_and_align_face(webcam_image, detector_backend='opencv')
        
        if webcam_face is None:
            return (
                "❌ Error: No face detected in webcam image. Please ensure the image contains a clear, visible face.",
                nid_detected_img, None, None, None, None, None, None, None, None
            )
        
        step_status += f"✅ Face detected at position: x={webcam_area['x']}, y={webcam_area['y']}, width={webcam_area['w']}, height={webcam_area['h']}</div>"
        
        # STEP 3: Face Alignment
        step_status += "<div style='padding: 10px; background-color: #d1ecf1; border-radius: 5px; margin: 10px 0;'>"
        step_status += "<b>Step 3: Face Alignment</b><br>"
        step_status += "Aligning faces to standard position using facial landmarks (eyes, nose, mouth)...<br>"
        step_status += "✅ Both faces aligned and cropped to focus on facial region</div>"
        
        aligned_nid = nid_face
        aligned_webcam = webcam_face
        
        # STEP 4: Preprocessing
        step_status += "<div style='padding: 10px; background-color: #f8d7da; border-radius: 5px; margin: 10px 0;'>"
        step_status += "<b>Step 4: Preprocessing</b><br>"
        step_status += f"Resizing images to {model_name} input size<br>"
        step_status += "Normalizing pixel values to range [0, 1]<br>"
        
        # Get target size based on model
        target_sizes = {
            "VGG-Face": (224, 224),
            "Facenet": (160, 160),
            "OpenFace": (96, 96),
            "ArcFace": (112, 112)
        }
        target_size = target_sizes.get(model_name, (224, 224))
        
        nid_preprocessed, nid_preprocessed_vis = preprocess_face(aligned_nid, target_size)
        webcam_preprocessed, webcam_preprocessed_vis = preprocess_face(aligned_webcam, target_size)
        
        step_status += f"✅ Images resized to {target_size[0]}x{target_size[1]} and normalized</div>"
        
        # STEP 5: Neural Network Processing
        step_status += "<div style='padding: 10px; background-color: #e7d4f5; border-radius: 5px; margin: 10px 0;'>"
        step_status += "<b>Step 5: Neural Network Processing</b><br>"
        step_status += f"Passing images through {model_name} deep neural network...<br>"
        step_status += f"Model: {model_name} (Convolutional Neural Network with multiple layers)<br>"
        
        # Get embeddings separately for visualization (Step 6)
        step_status += "✅ Processing images...</div>"
        
        step_status += "<div style='padding: 10px; background-color: #e0e0e0; border-radius: 5px; margin: 10px 0;'>"
        step_status += "<b>Step 6: Generate Embeddings (Feature Vectors)</b><br>"
        step_status += "Converting face images into numerical vectors that represent unique facial features...<br>"
        
        try:
            nid_embedding_result = DeepFace.represent(
                img_path=nid_image, 
                model_name=model_name, 
                enforce_detection=True,
                detector_backend='opencv'
            )
            webcam_embedding_result = DeepFace.represent(
                img_path=webcam_image, 
                model_name=model_name, 
                enforce_detection=True,
                detector_backend='opencv'
            )
            
            nid_embedding = nid_embedding_result[0]["embedding"]
            webcam_embedding = webcam_embedding_result[0]["embedding"]
            
            embedding_dim = len(nid_embedding)
            step_status += f"✅ Generated {embedding_dim}-dimensional embeddings for both faces<br>"
            step_status += f"NID Embedding sample: [{nid_embedding[0]:.4f}, {nid_embedding[1]:.4f}, ..., {nid_embedding[-1]:.4f}]<br>"
            step_status += f"Webcam Embedding sample: [{webcam_embedding[0]:.4f}, {webcam_embedding[1]:.4f}, ..., {webcam_embedding[-1]:.4f}]</div>"
            
            # Create embedding visualizations
            nid_emb_vis = create_embedding_visualization(np.array(nid_embedding))
            webcam_emb_vis = create_embedding_visualization(np.array(webcam_embedding))
        except Exception as e:
            step_status += f"⚠️ Embedding visualization unavailable: {str(e)}</div>"
            nid_emb_vis = None
            webcam_emb_vis = None
            embedding_dim = "N/A"
        
        # Perform actual verification
        result = DeepFace.verify(
            img1_path=nid_image,
            img2_path=webcam_image,
            model_name=model_name,
            enforce_detection=True,
            detector_backend='opencv'
        )
        
        # STEP 7: Calculate Distance
        distance = result["distance"]
        threshold = result["threshold"]
        metric = result["similarity_metric"]
        
        step_status += "<div style='padding: 10px; background-color: #fff8dc; border-radius: 5px; margin: 10px 0;'>"
        step_status += "<b>Step 7: Calculate Distance</b><br>"
        step_status += f"Computing {metric} distance between the two embeddings...<br>"
        step_status += f"Distance Metric: {metric}<br>"
        step_status += f"<b>Calculated Distance: {distance:.6f}</b><br>"
        step_status += "Lower distance = More similar faces</div>"
        
        # STEP 8: Threshold Comparison
        verified = result["verified"]
        
        step_status += "<div style='padding: 10px; background-color: #d1ecf1; border-radius: 5px; margin: 10px 0;'>"
        step_status += "<b>Step 8: Compare Distance to Threshold</b><br>"
        step_status += f"Model Threshold: {threshold:.6f}<br>"
        step_status += f"Calculated Distance: {distance:.6f}<br>"
        step_status += f"<b>Is Distance < Threshold? {distance:.6f} < {threshold:.6f}?</b><br>"
        
        if distance < threshold:
            step_status += f"✅ YES! {distance:.6f} < {threshold:.6f} → Faces MATCH</div>"
        else:
            step_status += f"❌ NO! {distance:.6f} ≥ {threshold:.6f} → Faces DO NOT MATCH</div>"
        
        # STEP 9: Final Result
        if verified:
            confidence = ((threshold - distance) / threshold) * 100
            status = "✅ MATCH"
            status_color = "green"
        else:
            confidence = ((distance - threshold) / distance) * 100 if distance > 0 else 0
            status = "❌ NO MATCH"
            status_color = "red"
        
        step_status += f"<div style='padding: 15px; background-color: {'#d4edda' if verified else '#f8d7da'}; border-radius: 5px; margin: 10px 0; border: 3px solid {status_color};'>"
        step_status += f"<h2 style='color: {status_color}; margin: 0;'><b>Step 9: Final Result - {status}</b></h2><br>"
        step_status += f"<b>Confidence: {confidence:.2f}%</b><br>"
        step_status += f"Model: {result['model']}<br>"
        step_status += f"Detector: {result['detector_backend']}</div>"
        
        # Create comprehensive summary
        summary = f"""
        <div style='padding: 20px; background-color: #f9f9f9; border-radius: 10px; border: 2px solid #ddd;'>
            <h2 style='text-align: center; color: {status_color};'>{status}</h2>
            <hr>
            <h3>📊 Final Metrics:</h3>
            <ul style='font-size: 16px; line-height: 1.8;'>
                <li><b>Verification Result:</b> <span style='color: {status_color};'>{verified}</span></li>
                <li><b>Confidence Score:</b> {confidence:.2f}%</li>
                <li><b>Distance Score:</b> {distance:.6f}</li>
                <li><b>Threshold:</b> {threshold:.6f}</li>
                <li><b>Difference:</b> {abs(distance - threshold):.6f}</li>
                <li><b>Model Used:</b> {result['model']}</li>
                <li><b>Face Detector:</b> {result['detector_backend']}</li>
                <li><b>Similarity Metric:</b> {metric}</li>
                <li><b>Embedding Dimension:</b> {embedding_dim}D vector</li>
            </ul>
            <hr>
            <div style='background-color: #fff3cd; padding: 10px; border-radius: 5px;'>
                <b>💡 Understanding the Process:</b><br>
                • The system extracted {embedding_dim} unique facial features from each image<br>
                • These features were compared using {metric} distance metric<br>
                • The distance of {distance:.6f} {'is below' if verified else 'exceeds'} the threshold of {threshold:.6f}<br>
                • Therefore, the faces {'are identified as the same person' if verified else 'are identified as different people'}
            </div>
        </div>
        """
        
        return (
            summary,
            nid_detected_img,      # Step 1 output
            webcam_detected_img,    # Step 2 output
            aligned_nid,            # Step 3 output (NID)
            aligned_webcam,         # Step 3 output (Webcam)
            nid_preprocessed_vis,   # Step 4 output (NID)
            webcam_preprocessed_vis, # Step 4 output (Webcam)
            nid_emb_vis,            # Step 6 output (NID embedding)
            webcam_emb_vis,         # Step 6 output (Webcam embedding)
            step_status             # All steps status
        )
        
    except Exception as e:
        error_msg = f"""
        <div style='padding: 20px; background-color: #f8d7da; border-radius: 10px; border: 2px solid #dc3545;'>
            <h3 style='color: #dc3545;'>❌ Error Occurred</h3>
            <p><b>Error:</b> {str(e)}</p>
            <p>Please ensure both images contain clear, visible faces and try again.</p>
        </div>
        """
        return (error_msg, None, None, None, None, None, None, None, None, None)

# Create Gradio Interface
with gr.Blocks(theme=gr.themes.Ocean()) as demo:
    
    gr.Markdown(
        """
        # 🔐 Face Verification System with Complete Process Visualization
        ### See Every Step: From Image Input to Final Verification Result
        
        This system provides **complete transparency** - visualizing exactly what happens at each step 
        of the face verification process, including intermediate images and numerical computations.
        """
    )
    
    with gr.Row():
        with gr.Column():
            nid_input = gr.Image(
                label="📇 NID Card Image",
                type="numpy",
                sources=["upload", "webcam"],
                height=300
            )
        
        with gr.Column():
            webcam_input = gr.Image(
                label="📷 Webcam/Live Image",
                type="numpy",
                sources=["upload", "webcam"],
                height=300
            )
    
    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=["ArcFace", "VGG-Face", "OpenFace", "Facenet"],
            value="ArcFace",
            label="🤖 Select Verification Model",
            info="ArcFace recommended for best accuracy"
        )
    
    verify_button = gr.Button("🔍 Verify Faces & Show Process", variant="primary", size="lg")
    
    gr.Markdown("---")
    gr.Markdown("## 📊 Verification Result")
    
    result_output = gr.HTML(label="Final Result Summary")
    
    gr.Markdown("---")
    gr.Markdown("## 🔍 Step-by-Step Process Visualization")
    
    process_output = gr.HTML(label="Detailed Process Steps")
    
    gr.Markdown("### Step 1-2: Face Detection")
    gr.Markdown("*Detecting and locating faces in both images*")
    
    with gr.Row():
        step1_output = gr.Image(label="Step 1: Face Detected in NID Image", type="numpy")
        step2_output = gr.Image(label="Step 2: Face Detected in Webcam Image", type="numpy")
    
    gr.Markdown("### Step 3: Face Alignment")
    gr.Markdown("*Cropped and aligned face regions*")
    
    with gr.Row():
        step3a_output = gr.Image(label="Step 3: Aligned NID Face", type="numpy")
        step3b_output = gr.Image(label="Step 3: Aligned Webcam Face", type="numpy")
    
    gr.Markdown("### Step 4: Preprocessing")
    gr.Markdown("*Resized and normalized images ready for neural network*")
    
    with gr.Row():
        step4a_output = gr.Image(label="Step 4: Preprocessed NID Face", type="numpy")
        step4b_output = gr.Image(label="Step 4: Preprocessed Webcam Face", type="numpy")
    
    gr.Markdown("### Step 5: Neural Network Processing")
    gr.Markdown("*Images processed through deep convolutional neural network layers*")
    gr.Info("Neural network processing happens internally - multiple convolutional, pooling, and dense layers extract features")
    
    gr.Markdown("### Step 6: Face Embeddings (Feature Vectors)")
    gr.Markdown("*Visual representation of high-dimensional embedding vectors*")
    gr.Markdown("Each color represents a value in the embedding vector - similar patterns indicate similar faces")
    
    with gr.Row():
        step6a_output = gr.Image(label="Step 6: NID Face Embedding Visualization", type="numpy")
        step6b_output = gr.Image(label="Step 6: Webcam Face Embedding Visualization", type="numpy")
    
    gr.Markdown("### Step 7-9: Distance Calculation, Threshold Comparison & Final Result")
    gr.Markdown("*Numerical comparison and decision-making process shown in the process steps above*")
    
    # Button click event
    verify_button.click(
        fn=verify_faces_with_visualization,
        inputs=[nid_input, webcam_input, model_dropdown],
        outputs=[
            result_output,
            step1_output,
            step2_output,
            step3a_output,
            step3b_output,
            step4a_output,
            step4b_output,
            step6a_output,
            step6b_output,
            process_output
        ]
    )
    
    gr.Markdown(
        """
        ---
        
        ## 📖 Complete Process Explanation
        
        ### 🔄 The 9-Step Face Verification Pipeline:
        
        | Step | Name | What Happens | Why It's Important |
        |------|------|--------------|-------------------|
        | **1-2** | **Face Detection** | Locates face in image using Haar Cascades or MTCNN | Must find face before processing |
        | **3** | **Face Alignment** | Rotates and crops face using eye positions as reference | Standardizes face orientation |
        | **4** | **Preprocessing** | Resizes to model input size, normalizes pixels to [0,1] | Prepares data for neural network |
        | **5** | **Neural Network** | Passes through CNN layers (conv → pool → dense) | Extracts hierarchical features |
        | **6** | **Embedding** | Final layer outputs N-dimensional vector (128-512D) | Compresses face to numerical representation |
        | **7** | **Distance Calc** | Computes Euclidean/Cosine distance between vectors | Measures similarity mathematically |
        | **8** | **Threshold** | Compares distance to model-specific threshold | Determines if similarity is high enough |
        | **9** | **Decision** | Returns Match/No Match based on comparison | Final verification result |
        
        ---
        
        ### 🧠 Understanding Each Step in Detail:
        
        #### Step 1-2: Face Detection
        - **Input**: Full image (NID card or webcam photo)
        - **Process**: Scans image with trained detector (OpenCV Haar Cascades, MTCNN, etc.)
        - **Output**: Bounding box coordinates (x, y, width, height) of detected face
        - **Visualization**: Green rectangle drawn around detected face with confidence score
        
        #### Step 3: Face Alignment
        - **Input**: Detected face region
        - **Process**: 
          - Detects facial landmarks (eyes, nose, mouth)
          - Calculates rotation angle to make eyes horizontal
          - Crops to standard size maintaining face center
        - **Output**: Aligned and cropped face image
        - **Why**: Ensures all faces have same orientation for consistent comparison
        
        #### Step 4: Preprocessing
        - **Input**: Aligned face image
        - **Process**:
          - Resize to model's required input size (e.g., 224×224, 160×160)
          - Convert pixel values from [0, 255] to [0, 1]
          - May apply mean subtraction and standardization
        - **Output**: Normalized face array ready for neural network
        - **Why**: Neural networks require specific input formats
        
        #### Step 5: Neural Network Processing
        - **Input**: Preprocessed face array
        - **Process**: 
          - Convolutional layers detect edges, textures, patterns
          - Pooling layers reduce spatial dimensions
          - Deeper layers detect complex features (eyes, nose shape, face structure)
          - Fully connected layers combine features
        - **Output**: Raw feature vector (pre-embedding)
        - **Architecture**: VGG-Face (16-22 layers), ResNet (50-100 layers), ArcFace (custom architecture)
        
        #### Step 6: Embedding Generation
        - **Input**: Neural network output
        - **Process**:
          - Final dense layer compresses features to fixed-length vector
          - Each dimension captures specific facial characteristic
          - L2 normalization ensures unit length
        - **Output**: N-dimensional embedding (128D, 512D, etc.)
        - **Visualization**: Shown as colored heatmap where each pixel represents one dimension
        - **Example**: [0.234, -0.456, 0.123, ..., 0.789] (512 numbers)
        
        #### Step 7: Calculate Distance
        - **Input**: Two embedding vectors (E1 and E2)
        - **Formulas**:
          - **Euclidean**: √(Σ(E1ᵢ - E2ᵢ)²)
          - **Cosine**: 1 - (E1·E2)/(||E1||×||E2||)
        - **Output**: Single distance value
        - **Interpretation**: 
          - Smaller distance = More similar faces
          - Larger distance = Less similar faces
        
        #### Step 8: Threshold Comparison
        - **Input**: Calculated distance and model threshold
        - **Process**: Simple comparison: `if distance < threshold`
        - **Thresholds** (examples):
          - ArcFace: ~0.68 (Cosine)
          - Facenet: ~0.40 (Euclidean)
          - VGG-Face: ~0.40 (Cosine)
        - **Output**: Boolean (True/False)
        
        #### Step 9: Final Decision
        - **Input**: Threshold comparison result
        - **Output**: 
          - **MATCH** if distance < threshold
          - **NO MATCH** if distance ≥ threshold
        - **Confidence**: Calculated as percentage based on distance from threshold
        
        ---
        
        ### 🎯 Model Comparison
        
        | Model | Embedding Size | Speed | Accuracy | Best For |
        |-------|---------------|-------|----------|----------|
        | **ArcFace** | 512D | Medium | 99.4%+ | High-security verification |
        | **VGG-Face** | 2622D | Slow | 98.9%+ | High accuracy needed |
        | **OpenFace** | 128D | Fast | 93%+ | Real-time applications |
        | **Facenet** | 128D | Fast | 99.2%+ | Resource-constrained systems |
        
        ---
        
        ### 💡 Key Concepts
        
        **What is an Embedding?**
        - A mathematical representation of a face in high-dimensional space
        - Similar faces have embeddings close together
        - Different faces have embeddings far apart
        - Think of it as "coordinates" for a face in 128-512 dimensional space
        
        **Distance Metrics:**
        - **Euclidean Distance**: Straight-line distance between two points
        - **Cosine Distance**: Measures angle between vectors (better for normalized embeddings)
        - **L2 Distance**: Similar to Euclidean, often used in face recognition
        
        **Why Use Deep Learning?**
        - Traditional methods (pixel comparison) fail with lighting, angle, expression changes
        - Deep learning learns invariant features that work across variations
        - Trained on millions of face images to recognize patterns humans can't describe
        
        ---
        
        ### ⚠️ Important Notes
        
        **For Best Results:**
        - ✅ Use clear, well-lit, front-facing photos
        - ✅ Ensure face is unobstructed
        - ✅ Similar lighting in both images helps
        - ✅ High resolution images (at least 640×480)
        
        **Limitations:**
        - ❌ Won't work with heavily occluded faces (masks, sunglasses)
        - ❌ Extreme angles or profiles reduce accuracy
        - ❌ Very low quality or blurry images may fail
        - ❌ Identical twins may be difficult to distinguish
        
        ---
        
        **🔬 Educational Purpose**: This tool demonstrates how modern face recognition works.
        For production systems, additional security measures and privacy protections are required.
        """
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True, debug=True)