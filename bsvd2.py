import cv2
import numpy as np

# Function to calculate PSNR
def calculate_psnr(original, denoised):
    mse = np.mean((original - denoised) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# Create VideoCapture object for webcam
cap = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Add more noticeable Gaussian noise to the original frame
    noise = np.random.normal(0, 50, frame.shape)
    noisy_frame = np.clip(frame + noise, 0, 255).astype(np.uint8)

    # Denoise the frame using Gaussian blur
    denoised_frame = cv2.GaussianBlur(noisy_frame, (5, 5), 0)

    # Calculate PSNR
    psnr_original = calculate_psnr(noisy_frame, frame)
    psnr_denoised = calculate_psnr(noisy_frame, denoised_frame)

    # Display the PSNR values in red for the original frame
    cv2.putText(noisy_frame, f"Noisy - PSNR: {psnr_original:.2f} dB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # Add label to the original window in red
    cv2.putText(noisy_frame, "Noisy", (10, noisy_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the PSNR values in green for the denoised frame
    cv2.putText(denoised_frame, f"Denoised - PSNR: {psnr_denoised:.2f} dB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # Add label to the denoised window in green
    cv2.putText(denoised_frame, "Denoised", (10, denoised_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Concatenate the original and denoised frames horizontally
    display_frame = np.concatenate((noisy_frame, denoised_frame), axis=1)

    # Display the original and denoised frames in a single window
    cv2.imshow('Noisy vs Denoised', display_frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()