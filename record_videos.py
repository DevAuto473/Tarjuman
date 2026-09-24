import os
import sys
import time
import cv2
import argparse

# Add src to path so we can import the project's SmartCamera (works on Pi and Laptop)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from tarjuman_core.camera_manager import SmartCamera, choose_camera_interactive

def crop_to_aspect_ratio(frame, target_width=640, target_height=480):
    """Locks the frame to exactly the target aspect ratio by center-cropping."""
    h, w = frame.shape[:2]
    target_ratio = target_width / target_height
    current_ratio = w / h
    
    if current_ratio > target_ratio + 0.01:
        # Frame is too wide (e.g. 16:9), crop the sides
        new_w = int(h * target_ratio)
        x_offset = (w - new_w) // 2
        frame = frame[:, x_offset:x_offset + new_w]
    elif current_ratio < target_ratio - 0.01:
        # Frame is too tall, crop the top and bottom
        new_h = int(w / target_ratio)
        y_offset = (h - new_h) // 2
        frame = frame[y_offset:y_offset + new_h, :]
        
    return cv2.resize(frame, (target_width, target_height))

def main():
    parser = argparse.ArgumentParser(description="Raw Video Recording Tool")
    parser.add_argument("--camera", type=str, default="auto", help="Camera source (e.g., auto, picamera2, picam_subprocess)")
    args = parser.parse_args()

    # Allow interactive camera selection if not provided
    camera_source = args.camera
    if camera_source == "auto":
        camera_source = choose_camera_interactive()

    print("=" * 50)
    print(" Raw Video Recording Tool (No MediaPipe)")
    print("=" * 50)
    
    # Setup target dimensions
    width, height = 640, 480
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Use SmartCamera to ensure compatibility with Raspberry Pi
    print("Starting camera... (This only happens once)")
    cam = SmartCamera(source=camera_source, width=width, height=height, fps=fps)
    cam.start()
    cam.start_grabber()
    
    if not cam.is_running:
        print("Failed to open the camera.")
        return

    try:
        while True:
            # 1. Ask for details
            print("-" * 50)
            sign_name = input("Enter the sign name you want to record (or press Enter to quit): ").strip()
            if not sign_name:
                print("Exiting...")
                break
                
            save_dir = os.path.join("data", "DATA", sign_name)
            os.makedirs(save_dir, exist_ok=True)
            
            existing_mp4s = [f for f in os.listdir(save_dir) if f.endswith('.mp4')]
            
            offset = 0
            is_replacing = False
            target_replace_num = -1
            num_videos_to_record = 1
            
            if existing_mp4s:
                max_idx = 0
                for f in existing_mp4s:
                    try:
                        # e.g., hello_003.mp4 -> 3
                        idx_str = f.split('_')[-1].split('.')[0]
                        max_idx = max(max_idx, int(idx_str))
                    except:
                        pass
                
                print(f"\nFolder already exists and contains {len(existing_mp4s)} videos (Highest index: {max_idx}).")
                choice = input("Do you want to add (N)ew videos or (R)eplace an existing video? [N/r]: ").strip().lower()
                
                if choice == 'r':
                    is_replacing = True
                    try:
                        target_replace_num = int(input(f"Which video number do you want to replace? (1 to {max_idx}): ").strip())
                    except ValueError:
                        print("Invalid number. Cancelling.")
                        continue
                        
                    num_videos_to_record = 1
                    offset = target_replace_num - 1
                else:
                    offset = max_idx
                    try:
                        num_videos_input = input("How many new videos do you want to record? (default 1): ").strip()
                        num_videos_to_record = int(num_videos_input) if num_videos_input else 1
                    except ValueError:
                        print("Invalid number. Please try again.")
                        continue
            else:
                try:
                    num_videos_input = input("How many videos do you want to record? (default 1): ").strip()
                    num_videos_to_record = int(num_videos_input) if num_videos_input else 1
                except ValueError:
                    print("Invalid number. Please try again.")
                    continue

            target_count = offset + num_videos_to_record
            video_count = offset
            
            print("\nInstructions:")
            print(" - Press 'r' to start recording a new video.")
            print(" - Press 's' to stop recording and save the current video.")
            print(" - Press 'q' to abort this sign and choose a new one.\n")

            is_recording = False
            out = None
            
            window_name = f"Recording: {sign_name}"
            cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO | cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, width, height)

            # 2. Record the videos for this sign
            while video_count < target_count or is_recording:
                ret, frame = cam.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue
                
                # Enforce exactly 4:3 aspect ratio (640x480)
                frame = crop_to_aspect_ratio(frame, width, height)
                
                display_frame = frame.copy()
                
                # Draw overlay texts
                if is_replacing:
                    status_text = f"Replacing Video {target_replace_num} - "
                    status_text += "RECORDING" if is_recording else "READY (Press 'r')"
                else:
                    progress = video_count - offset
                    if not is_recording:
                        progress += 1
                        
                    if progress > num_videos_to_record:
                        progress = num_videos_to_record
                        
                    status_text = f"New: {progress}/{num_videos_to_record} (File: {video_count+1 if not is_recording else video_count}) - "
                    status_text += "RECORDING" if is_recording else "READY (Press 'r')"

                color = (0, 0, 255) if is_recording else (0, 255, 0)
                
                cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                if is_recording:
                    cv2.circle(display_frame, (width - 30, 30), 10, (0, 0, 255), -1)
                    
                cv2.imshow(window_name, display_frame)
                
                if is_recording and out is not None:
                    out.write(frame) # Save original frame without text overlays
                    
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print(f"Aborted recording for sign '{sign_name}'.")
                    break
                elif key == ord('r') and not is_recording:
                    # Start new recording
                    video_count += 1
                    filename = os.path.join(save_dir, f"{sign_name}_{video_count:03d}.mp4")
                    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
                    is_recording = True
                    print(f"Started recording video {video_count}...")
                elif key == ord('s') and is_recording:
                    # Stop recording
                    is_recording = False
                    if out:
                        out.release()
                        out = None
                    print(f"Saved video {video_count} to {filename}")
                    
            # 3. Clean up the window before asking for the next sign
            if out:
                out.release()
            cv2.destroyWindow(window_name)
            cv2.waitKey(1) # process window destruction event
            
            print(f"\nDone with sign '{sign_name}'.")

    finally:
        cam.stop_grabber()
        cam.release()
        cv2.destroyAllWindows()
        print("\nCamera closed.")

if __name__ == "__main__":
    main()
