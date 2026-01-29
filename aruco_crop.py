import cv2
import numpy as np
from pathlib import Path


def detect_markers_robust(gray, expected_ids=None):
    """
    Try multiple detection strategies to find markers.
    Returns corners, ids.
    """
    if expected_ids is None:
        expected_ids = {0, 1, 2, 3}
    
    configs = [
        {
            'minDistanceToBorder': 0,
            'errorCorrectionRate': 1.0,
            'adaptiveThreshWinSizeMin': 3,
            'adaptiveThreshWinSizeMax': 153,
            'adaptiveThreshWinSizeStep': 10,
        },
        {
            'minDistanceToBorder': 0,
            'errorCorrectionRate': 1.0,
            'adaptiveThreshWinSizeMin': 3,
            'adaptiveThreshWinSizeMax': 253,
            'adaptiveThreshWinSizeStep': 20,
        },
        {
            'minDistanceToBorder': 0,
            'errorCorrectionRate': 1.0,
            'adaptiveThreshWinSizeMin': 3,
            'adaptiveThreshWinSizeMax': 153,
            'adaptiveThreshWinSizeStep': 4,
        },
    ]
    
    preprocessors = [
        ("original", lambda g: g),
        ("clahe", lambda g: cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(g)),
        ("clahe_strong", lambda g: cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16,16)).apply(g)),
    ]
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    
    best_corners = None
    best_ids = None
    best_count = 0
    
    for preprocess_name, preprocess_fn in preprocessors:
        try:
            processed = preprocess_fn(gray)
        except:
            continue
        
        for config in configs:
            parameters = cv2.aruco.DetectorParameters()
            for key, value in config.items():
                setattr(parameters, key, value)
            
            detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
            corners, ids, rejected = detector.detectMarkers(processed)
            
            if ids is not None:
                valid_corners = []
                valid_ids = []
                for corner, marker_id in zip(corners, ids):
                    if marker_id[0] in expected_ids:
                        valid_corners.append(corner)
                        valid_ids.append(marker_id)
                
                if len(valid_ids) > best_count:
                    best_corners = valid_corners
                    best_ids = valid_ids
                    best_count = len(valid_ids)
                
                if best_count >= 4:
                    return best_corners, np.array(best_ids)
    
    if best_ids:
        return best_corners, np.array(best_ids)
    return None, None


def estimate_missing_marker(marker_data, missing_id):
    """
    Estimate the position of a missing marker based on the 3 detected ones.
    Uses the assumption that markers form a rectangle.
    """
    # Expected positions: 0=top-left, 1=top-right, 2=bottom-right, 3=bottom-left
    positions = {m['id']: m for m in marker_data}
    
    detected_ids = set(positions.keys())
    
    # Get the three detected markers
    if missing_id == 0:  # Missing top-left
        # Estimate from 1 (TR), 2 (BR), 3 (BL)
        # TL = BL + (TR - BR)
        tr, br, bl = positions[1], positions[2], positions[3]
        estimated_center = bl['center'] + (tr['center'] - br['center'])
        estimated_corners = bl['corners'] + (tr['corners'] - br['corners'])
    elif missing_id == 1:  # Missing top-right
        # Estimate from 0 (TL), 2 (BR), 3 (BL)
        # TR = TL + (BR - BL)
        tl, br, bl = positions[0], positions[2], positions[3]
        estimated_center = tl['center'] + (br['center'] - bl['center'])
        estimated_corners = tl['corners'] + (br['corners'] - bl['corners'])
    elif missing_id == 2:  # Missing bottom-right
        # Estimate from 0 (TL), 1 (TR), 3 (BL)
        # BR = TR + (BL - TL)
        tl, tr, bl = positions[0], positions[1], positions[3]
        estimated_center = tr['center'] + (bl['center'] - tl['center'])
        estimated_corners = tr['corners'] + (bl['corners'] - tl['corners'])
    elif missing_id == 3:  # Missing bottom-left
        # Estimate from 0 (TL), 1 (TR), 2 (BR)
        # BL = TL + (BR - TR)
        tl, tr, br = positions[0], positions[1], positions[2]
        estimated_center = tl['center'] + (br['center'] - tr['center'])
        estimated_corners = tl['corners'] + (br['corners'] - tr['corners'])
    
    return {
        'id': missing_id,
        'corners': estimated_corners,
        'center': estimated_center,
        'estimated': True
    }


def detect_and_crop_aruco(image_path, output_path=None, padding=5, expected_ids=None, allow_estimation=True):
    """
    Detect ArUco markers and crop with custom boundaries.
    
    If only 3 markers are found and allow_estimation=True, estimates the 4th marker position.
    """
    if expected_ids is None:
        expected_ids = {0, 1, 2, 3}
    
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    corners, ids = detect_markers_robust(gray, expected_ids)
    
    found_count = len(ids) if ids is not None else 0
    
    if found_count < 3:
        found_ids = [id[0] for id in ids] if ids is not None else []
        raise ValueError(f"Need at least 3 markers, found: {found_count} (IDs: {found_ids})")
    
    # Organize markers by position
    marker_data = []
    for corner, marker_id in zip(corners, ids):
        center = corner[0].mean(axis=0)
        marker_data.append({
            'id': marker_id[0],
            'corners': corner[0],
            'center': center,
            'estimated': False
        })
    
    # Check if we need to estimate a missing marker
    found_ids = {m['id'] for m in marker_data}
    missing_ids = expected_ids - found_ids
    
    if len(missing_ids) == 1 and allow_estimation:
        missing_id = list(missing_ids)[0]
        # Need to first sort markers to identify their positions
        marker_data.sort(key=lambda m: m['center'][1])
        top_markers = sorted(marker_data[:2] if len(marker_data) >= 2 else marker_data, key=lambda m: m['center'][0])
        bottom_markers = sorted(marker_data[2:] if len(marker_data) > 2 else [], key=lambda m: m['center'][0])
        
        # Re-identify markers by position
        position_map = {}
        if len(top_markers) >= 1:
            position_map[top_markers[0]['id']] = 'tl'
        if len(top_markers) >= 2:
            position_map[top_markers[1]['id']] = 'tr'
        if len(bottom_markers) >= 1:
            position_map[bottom_markers[0]['id']] = 'bl'
        if len(bottom_markers) >= 2:
            position_map[bottom_markers[1]['id']] = 'br'
        
        estimated = estimate_missing_marker(marker_data, missing_id)
        marker_data.append(estimated)
        print(f"  (Estimated missing marker ID {missing_id})")
    elif len(missing_ids) > 1:
        raise ValueError(f"Need 4 markers, found: {found_count} (IDs: {sorted(found_ids)})")
    
    # Sort by Y to get top vs bottom, then by X to get left vs right
    marker_data.sort(key=lambda m: m['center'][1])
    top_markers = sorted(marker_data[:2], key=lambda m: m['center'][0])
    bottom_markers = sorted(marker_data[2:], key=lambda m: m['center'][0])
    
    top_left = top_markers[0]
    top_right = top_markers[1]
    bottom_left = bottom_markers[0]
    bottom_right = bottom_markers[1]
    
    def get_marker_bounds(marker):
        c = marker['corners']
        return {
            'left': c[:, 0].min(),
            'right': c[:, 0].max(),
            'top': c[:, 1].min(),
            'bottom': c[:, 1].max()
        }
    
    tl = get_marker_bounds(top_left)
    tr = get_marker_bounds(top_right)
    bl = get_marker_bounds(bottom_left)
    br = get_marker_bounds(bottom_right)
    
    # Calculate crop boundaries
    left_boundary = max(tl['right'], bl['right']) + padding
    right_boundary = max(tr['right'], br['right']) - padding
    top_boundary = max(tl['bottom'], tr['bottom']) + padding
    bottom_boundary = min(bl['top'], br['top']) - padding
    
    x1 = int(max(0, left_boundary))
    x2 = int(min(image.shape[1], right_boundary))
    y1 = int(max(0, top_boundary))
    y2 = int(min(image.shape[0], bottom_boundary))
    
    cropped = image[y1:y2, x1:x2]
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), cropped)
        
        # Visualization
        viz = image.copy()
        # Draw detected markers
        detected_corners = [np.array([m['corners']]) for m in marker_data if not m.get('estimated', False)]
        detected_ids = np.array([[m['id']] for m in marker_data if not m.get('estimated', False)])
        if len(detected_corners) > 0:
            cv2.aruco.drawDetectedMarkers(viz, detected_corners, detected_ids)
        
        # Draw estimated markers in different color
        for m in marker_data:
            if m.get('estimated', False):
                pts = m['corners'].astype(int)
                cv2.polylines(viz, [pts], True, (0, 165, 255), 2)  # Orange for estimated
                cv2.putText(viz, f"ID={m['id']}(est)", tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        
        cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 0), 3)
        viz_path = output_path.parent / f"{output_path.stem}_viz{output_path.suffix}"
        cv2.imwrite(str(viz_path), viz)
    
    return cropped


def process_batch(input_dir, output_dir, padding=5):
    """Process all images in a directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    images = sorted([f for f in input_dir.iterdir() if f.suffix in extensions])
    
    print(f"Found {len(images)} images to process")
    
    success = 0
    failed = 0
    failed_files = []
    
    for img_path in images:
        print(f"\nProcessing: {img_path.name}")
        try:
            output_path = output_dir / f"{img_path.stem}_cropped{img_path.suffix}"
            detect_and_crop_aruco(img_path, output_path, padding=padding)
            print(f"  ✓ Success")
            success += 1
        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed += 1
            failed_files.append(img_path.name)
    
    print(f"\n{'='*60}")
    print(f"Successful: {success}/{len(images)}")
    print(f"Failed: {failed}/{len(images)}")
    if failed_files:
        print(f"Failed files: {failed_files}")
    

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) >= 2:
        input_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) >= 3 else "cropped_output.png"
        
        if Path(input_path).is_dir():
            process_batch(input_path, output_path)
        else:
            cropped = detect_and_crop_aruco(input_path, output_path, padding=5)
            print(f"Saved cropped image: {cropped.shape[:2]}")
    else:
        print("Usage:")
        print("  Single image: python aruco_crop.py input.jpg output.png")
        print("  Batch:        python aruco_crop.py input_dir/ output_dir/")
