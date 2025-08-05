import cv2
import numpy as np
import time
import os
import argparse
from enum import Enum
import json

class BackgroundMode(Enum):
    STATIC = 0
    VIDEO = 1
    BLUR = 2
    NEIGHBOR = 3
    CHROMA = 4

def parse_arguments():
    """Parse command line arguments with enhanced options"""
    parser = argparse.ArgumentParser(description='Ultimate Invisibility Cloak Simulation')
    parser.add_argument('--color', type=str, default='red', 
                       choices=['red', 'green', 'blue', 'custom'],
                       help='Color of the invisibility cloak')
    parser.add_argument('--hsv-range', type=str, default=None,
                       help='Custom HSV range as "H1-H2,S1-S2,V1-V2"')
    parser.add_argument('--video', type=str, default='video.mp4',
                       help='Path to background video file')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to static background image file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path to save result')
    parser.add_argument('--no-smoothing', action='store_true',
                       help='Disable mask smoothing for better performance')
    parser.add_argument('--debug', action='store_true',
                       help='Show debug views of the masking process')
    parser.add_argument('--no-mirror', action='store_true',
                       help='Disable mirror effect (shows true camera view)')
    parser.add_argument('--save-config', type=str, default=None,
                       help='Save current settings to config file')
    parser.add_argument('--load-config', type=str, default=None,
                       help='Load settings from config file')
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='Color detection threshold (0.1-1.0)')
    parser.add_argument('--edge-blend', type=int, default=5,
                       help='Edge blending width in pixels (0-20)')
    return parser.parse_args()

def load_config(config_path):
    """Load settings from JSON config file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Failed to load config: {e}")
        return None

def save_config(config_path, settings):
    """Save settings to JSON config file"""
    try:
        with open(config_path, 'w') as f:
            json.dump(settings, f, indent=4)
        print(f"✓ Config saved to {config_path}")
    except Exception as e:
        print(f"⚠️ Failed to save config: {e}")

def get_color_range(color, hsv_range=None, threshold=0.7):
    """Return HSV color ranges with adjustable threshold"""
    threshold = np.clip(threshold, 0.1, 1.0)
    
    if hsv_range:
        try:
            h_range, s_range, v_range = hsv_range.split(',')
            h1, h2 = map(int, h_range.split('-'))
            s1, s2 = map(int, s_range.split('-'))
            v1, v2 = map(int, v_range.split('-'))
            return [(np.array([h1, s1, v1]), np.array([h2, s2, v2]))]
        except:
            print("⚠️ Invalid HSV range format. Using default.")
    
    if color == 'red':
        lower1 = np.array([0, int(120*threshold), int(70*threshold)])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([170, int(120*threshold), int(70*threshold)])
        upper2 = np.array([180, 255, 255])
        return [(lower1, upper1), (lower2, upper2)]
    elif color == 'green':
        lower = np.array([35, int(100*threshold), int(100*threshold)])
        upper = np.array([85, 255, 255])
        return [(lower, upper)]
    elif color == 'blue':
        lower = np.array([90, int(100*threshold), int(100*threshold)])
        upper = np.array([130, 255, 255])
        return [(lower, upper)]
    else:  # custom
        return [(np.array([0, int(120*threshold), int(70*threshold)]), 
                np.array([180, 255, 255]))]

def setup_background(video_path, image_path, frame_size):
    """Setup multiple background options"""
    sources = {}
    
    # Video background
    if os.path.exists(video_path):
        bg_video = cv2.VideoCapture(video_path)
        if bg_video.isOpened():
            ret, frame = bg_video.read()
            if ret:
                sources['video'] = (bg_video, cv2.resize(frame, frame_size))
    
    # Image background
    if image_path and os.path.exists(image_path):
        img = cv2.imread(image_path)
        if img is not None:
            sources['image'] = (None, cv2.resize(img, frame_size))
    
    # Default black background
    if not sources:
        print("ℹ️ Using default black background")
        sources['black'] = (None, np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8))
    
    return sources

def create_edge_blend_mask(mask, width):
    """Create smooth transition at mask edges"""
    if width <= 0:
        return mask
    
    kernel = np.ones((width, width), np.float32) / (width*width)
    blurred = cv2.filter2D(mask.astype(np.float32), -1, kernel)
    
    # Combine original mask with blurred edges
    result = np.where(mask == 255, 255, 
                     np.where(blurred > 0, blurred, 0))
    
    return result.astype(np.uint8)

def apply_chroma_key(frame, bg_frame, mask, edge_blend=0):
    """Enhanced chroma key with edge blending and spill suppression"""
    # Create blended mask
    blended_mask = create_edge_blend_mask(mask, edge_blend) / 255.0
    
    # Convert to float for high quality blending
    frame_f = frame.astype(np.float32) / 255
    bg_f = bg_frame.astype(np.float32) / 255
    
    # Apply the blended mask
    result = frame_f * (1 - blended_mask[..., None]) + bg_f * blended_mask[..., None]
    
    # Convert back to 8-bit
    return (result * 255).astype(np.uint8)

def process_frame(frame, bg_frame, color_ranges, bg_mode, no_smoothing=False, 
                 debug=False, edge_blend=0):
    """Process frame with multiple background modes"""
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create combined mask for all color ranges
    combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for lower, upper in color_ranges:
        mask = cv2.inRange(hsv, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # Clean up the mask
    if not no_smoothing:
        combined_mask = cv2.medianBlur(combined_mask, 5)
        kernel = np.ones((5,5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.dilate(combined_mask, None, iterations=2)
    
    # Handle different background modes
    if bg_mode == BackgroundMode.BLUR:
        bg_frame = cv2.GaussianBlur(frame, (51,51), 0)
    elif bg_mode == BackgroundMode.NEIGHBOR:
        # Simple neighbor fill (could be improved with inpainting)
        bg_frame = cv2.medianBlur(frame, 11)
    
    # Apply chroma key with edge blending
    final_output = apply_chroma_key(frame, bg_frame, combined_mask, edge_blend)
    
    if debug:
        debug_views = {
            'Original': frame,
            'HSV': cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR),
            'Mask': cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR),
            'Background': bg_frame,
            'Result': final_output
        }
        return final_output, debug_views
    
    return final_output, None

def main():
    args = parse_arguments()
    
    # Load config if specified
    if args.load_config:
        config = load_config(args.load_config)
        if config:
            for key, value in config.items():
                if hasattr(args, key):
                    setattr(args, key, value)
    
    print(f"""
    🧙 Ultimate Invisibility Cloak 2.0
    � Color: {args.color.upper()} (Threshold: {args.threshold})
    🖼️ Background: {'Video' if os.path.exists(args.video) else 'Image' if args.image else 'None'}
    📷 Camera: {'Mirrored' if not args.no_mirror else 'True View'}
    🛠️ Edge Blend: {args.edge_blend}px | Smoothing: {'ON' if not args.no_smoothing else 'OFF'}
    
    Controls:
    ESC: Exit | SPACE: Cycle Backgrounds | M: Toggle Mirror
    D: Debug View | S: Save Settings | +/-: Adjust Threshold
    """)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Webcam not accessible.")
        exit()

    # Get webcam frame size
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Failed to read frame from webcam.")
        exit()

    frame_size = (frame.shape[1], frame.shape[0])
    color_ranges = get_color_range(args.color, args.hsv_range, args.threshold)
    
    # Setup background sources
    bg_sources = setup_background(args.video, args.image, frame_size)
    bg_modes = list(BackgroundMode)
    current_bg_mode = BackgroundMode.VIDEO if 'video' in bg_sources else BackgroundMode.STATIC
    bg_index = bg_modes.index(current_bg_mode)
    
    # Get initial background
    bg_source = None
    if current_bg_mode == BackgroundMode.VIDEO:
        bg_source, bg_frame = bg_sources['video']
    elif current_bg_mode == BackgroundMode.STATIC and 'image' in bg_sources:
        bg_source, bg_frame = bg_sources['image']
    else:
        bg_frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
    
    # Setup output writer if requested
    output_writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_writer = cv2.VideoWriter(args.output, fourcc, 20.0, frame_size)
    
    # State variables
    show_debug = args.debug
    prev_time = time.time()
    fps_counter = 0
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame.")
            break
        
        # Mirror the frame by default (more natural like a mirror)
        if not args.no_mirror:
            frame = cv2.flip(frame, 1)
        
        # Get current background
        if current_bg_mode == BackgroundMode.VIDEO and bg_source:
            ret_bg, new_bg_frame = bg_source.read()
            if ret_bg:
                bg_frame = cv2.resize(new_bg_frame, frame_size)
            else:
                bg_source.set(cv2.CAP_PROP_POS_FRAMES, 0)
        elif current_bg_mode == BackgroundMode.STATIC and 'image' in bg_sources:
            bg_frame = bg_sources['image'][1]
        
        # Process frame
        final_output, debug_views = process_frame(
            frame, bg_frame, color_ranges, current_bg_mode, 
            args.no_smoothing, show_debug, args.edge_blend)
        
        # Calculate FPS (smoothed)
        fps_counter += 1
        if fps_counter % 10 == 0:
            curr_time = time.time()
            fps = 10 / (curr_time - prev_time)
            prev_time = curr_time
        
        # Add overlay information
        overlay_text = [
            f"FPS: {int(fps)}",
            f"Color: {args.color} (T:{args.threshold:.1f})",
            f"BG: {current_bg_mode.name}",
            f"Mirror: {'ON' if not args.no_mirror else 'OFF'}",
            f"Blend: {args.edge_blend}px",
            "Controls: SPACE=BG, M=Mirror, D=Debug, S=Save"
        ]
        
        for i, text in enumerate(overlay_text):
            cv2.putText(final_output, text, (10, 30 + i*25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        # Show debug views if enabled
        if show_debug and debug_views:
            debug_output = np.vstack([
                np.hstack([debug_views['Original'], debug_views['HSV']]),
                np.hstack([debug_views['Mask'], debug_views['Background']]),
                np.hstack([debug_views['Result'], 
                          np.zeros_like(debug_views['Result'])])
            ])
            cv2.imshow("Debug Views", debug_output)
        
        # Display the output
        cv2.imshow("🧙 Ultimate Invisibility Cloak 2.0", final_output)
        
        # Save to output if requested
        if output_writer:
            output_writer.write(final_output)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord(' '):  # SPACE cycles backgrounds
            bg_index = (bg_index + 1) % len(bg_modes)
            current_bg_mode = bg_modes[bg_index]
        elif key == ord('d'):  # Toggle debug views
            show_debug = not show_debug
            if not show_debug:
                cv2.destroyWindow("Debug Views")
        elif key == ord('m'):  # Toggle mirror effect
            args.no_mirror = not args.no_mirror
        elif key == ord('s'):  # Save settings
            if args.save_config:
                settings = {
                    'color': args.color,
                    'threshold': args.threshold,
                    'edge_blend': args.edge_blend,
                    'no_mirror': args.no_mirror,
                    'no_smoothing': args.no_smoothing
                }
                save_config(args.save_config, settings)
        elif key == ord('+') and args.threshold < 1.0:  # Increase threshold
            args.threshold = min(1.0, args.threshold + 0.05)
            color_ranges = get_color_range(args.color, args.hsv_range, args.threshold)
        elif key == ord('-') and args.threshold > 0.1:  # Decrease threshold
            args.threshold = max(0.1, args.threshold - 0.05)
            color_ranges = get_color_range(args.color, args.hsv_range, args.threshold)
        elif key == ord('[') and args.edge_blend > 0:  # Decrease edge blend
            args.edge_blend -= 1
        elif key == ord(']') and args.edge_blend < 20:  # Increase edge blend
            args.edge_blend += 1
    
    # Release resources
    cap.release()
    for source in bg_sources.values():
        if source[0]:  # Close video sources
            source[0].release()
    if output_writer:
        output_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()