import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Custom metric class (needed for model loading)
class RegressionAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='reg_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.round(tf.clip_by_value(tf.reshape(y_pred, [-1]), 0, 4))
        matches = tf.cast(tf.equal(y_true, y_pred), tf.float32)
        batch_acc = tf.reduce_mean(matches)
        self.total.assign_add(batch_acc)
        self.count.assign_add(1.0)

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)

    def reset_state(self):
        self.total.assign(0)
        self.count.assign(0)


def load_model(model_path='dr_model.h5'):
    """Load the trained DR model with custom metrics"""
    try:
        model = keras.models.load_model(
            model_path,
            custom_objects={'RegressionAccuracy': RegressionAccuracy},
            compile=False
        )
        # Recompile the model
        model.compile(
            loss='mse',
            metrics=['mae', RegressionAccuracy()]
        )
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def crop_image(img, tol=7):
    """Remove black borders from fundus image"""
    img = np.array(img)
    if len(img.shape) == 2:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    mask = gray > tol
    if not mask.any():
        return img
    return img[np.ix_(mask.any(1), mask.any(0))]


def circle_crop(img):
    """Apply circular crop to fundus image"""
    img = np.array(img)
    img = crop_image(img)
    h, w = img.shape[:2]
    side = max(h, w)
    
    # Resize to square
    img = cv2.resize(img, (side, side))
    
    # Create circular mask
    x, y = side // 2, side // 2
    r = min(x, y)
    
    mask = np.zeros((side, side), np.uint8)
    cv2.circle(mask, (x, y), r, 1, -1)
    
    mask = mask.astype(np.uint8)
    print(type(img), img.shape, img.dtype)
    print(type(mask), mask.shape, mask.dtype)
    img = cv2.bitwise_and(img, img, mask=mask)
    return crop_image(img)


def preprocess_image(image, target_size=(320, 320)):
    """
    Full preprocessing pipeline matching training:
    1. Circle crop
    2. Resize to target size
    3. Ben Graham's preprocessing (contrast enhancement)
    4. Normalize to [0, 1]
    """
    # Ensure RGB format
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Circle crop
    image = circle_crop(image)
    
    # Resize
    image = cv2.resize(image, target_size)
    
    # Ben Graham's preprocessing - subtract local average color
    image = cv2.addWeighted(
        image, 4, 
        cv2.GaussianBlur(image, (0, 0), 10), -4, 
        128
    )
    
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    return image


def make_gradcam_heatmap(img_array, model, last_conv_layer_name='top_conv', pred_index=None):
    """
    Generate GradCAM heatmap for model interpretation
    """
    # Create a model that maps the input image to the activations of the last conv layer
    grad_model = Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    # Compute the gradient of the predicted class with respect to the output feature map
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, 0]  # Regression output
    
    # Gradient of the output neuron with respect to the output feature map
    grads = tape.gradient(class_channel, conv_outputs)
    
    # Mean intensity of the gradient over each feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the channels by their importance
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def overlay_gradcam(image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlay GradCAM heatmap on original image
    """
    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Apply colormap
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    
    # Convert to RGB
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Ensure image is in correct format
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = np.uint8(255 * image)
    
    # Overlay heatmap on image
    superimposed = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    
    return superimposed


def predict_dr_stage(image, model):
    """
    Predict DR stage from preprocessed image
    Returns: stage_name, confidence, raw_prediction
    """
    # Add batch dimension
    img_array = np.expand_dims(image, axis=0)
    
    # Get prediction
    prediction = model.predict(img_array, verbose=0)
    raw_pred = prediction[0][0]
    
    # Round to nearest integer class
    stage = int(np.clip(np.round(raw_pred), 0, 4))
    
    # Calculate confidence (distance from nearest class boundary)
    distance_to_pred = abs(raw_pred - stage)
    confidence = 1 - min(distance_to_pred, 0.5) * 2  # Scale to [0.5, 1]
    
    # Stage names
    stage_names = {
        0: "No DR",
        1: "Mild Non-Proliferative DR",
        2: "Moderate Non-Proliferative DR",
        3: "Severe Non-Proliferative DR",
        4: "Proliferative DR"
    }
    
    return stage_names[stage], confidence, raw_pred


def get_stage_recommendations(stage_name):
    """Get clinical findings and recommendations based on stage"""
    stage_info = {
        "No DR": {
            "findings": "No signs of diabetic retinopathy detected. Retinal blood vessels appear normal.",
            "recommendations": [
                "Continue regular diabetes management",
                "Annual eye examinations recommended",
                "Maintain healthy HbA1c levels (below 7%)",
                "Monitor blood pressure and cholesterol",
                "Maintain healthy lifestyle habits"
            ],
            "severity": "low"
        },
        "Mild Non-Proliferative DR": {
            "findings": "Microaneurysms present in the retina. These are small bulges in blood vessels.",
            "recommendations": [
                "Monitor blood sugar levels closely",
                "Schedule eye exams every 6-12 months",
                "Maintain HbA1c below 7%",
                "Control blood pressure (target <140/90 mmHg)",
                "Consult with your ophthalmologist regularly"
            ],
            "severity": "low"
        },
        "Moderate Non-Proliferative DR": {
            "findings": "Multiple microaneurysms, dot/blot hemorrhages, and some blood vessel blockage detected.",
            "recommendations": [
                "Eye exams every 3-6 months required",
                "Strict glycemic control essential (HbA1c <7%)",
                "Consider laser treatment consultation",
                "Monitor for macular edema",
                "Control blood pressure and cholesterol strictly",
                "May require fluorescein angiography"
            ],
            "severity": "medium"
        },
        "Severe Non-Proliferative DR": {
            "findings": "Extensive hemorrhages, cotton wool spots, and venous beading observed. High risk of progression to proliferative DR.",
            "recommendations": [
                "⚠️ IMMEDIATE ophthalmologist consultation required",
                "Eye exams every 2-3 months mandatory",
                "Laser photocoagulation treatment strongly recommended",
                "Intensive blood sugar management critical",
                "Monitor closely for progression to PDR",
                "Consider panretinal photocoagulation (PRP)"
            ],
            "severity": "high"
        },
        "Proliferative DR": {
            "findings": "⚠️ CRITICAL: Abnormal new blood vessel growth (neovascularization) detected. These fragile vessels can bleed and cause severe vision loss.",
            "recommendations": [
                "🚨 URGENT: Immediate treatment required",
                "Panretinal photocoagulation (PRP) laser surgery needed",
                "Anti-VEGF injections may be necessary",
                "Monthly follow-up appointments mandatory",
                "High risk of vision loss - immediate action critical",
                "Consider vitrectomy if severe bleeding occurs",
                "Emergency contact: See ophthalmologist within 24-48 hours"
            ],
            "severity": "critical"
        }
    }
    
    return stage_info.get(stage_name, stage_info["No DR"])


def create_comparison_image(original, processed, gradcam):
    """Create a side-by-side comparison image"""
    # Ensure all images are the same size
    h, w = processed.shape[:2]
    
    if original.dtype == np.float32 or original.dtype == np.float64:
        original = np.uint8(255 * original)
    if processed.dtype == np.float32 or processed.dtype == np.float64:
        processed = np.uint8(255 * processed)
    
    original = cv2.resize(original, (w, h))
    
    # Create comparison
    comparison = np.hstack([original, processed, gradcam])
    
    return comparison
