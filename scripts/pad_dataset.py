import csv
import os
import sys

def migrate_dataset():
    old_file = r"c:\Users\HP\Desktop\pojects tree\TarjumanV1 (Copy)\data\dynamic_gestures_v5.csv"
    new_file = r"c:\Users\HP\Desktop\pojects tree\TarjumanV1 (Copy)\data\dynamic_gestures_v5_migrated.csv"

    if not os.path.exists(old_file):
        print(f"File {old_file} not found.")
        return

    # Old dimensions
    OLD_VALS_PER_FRAME = 140
    NEW_VALS_PER_FRAME = 144
    SEQUENCE_LENGTH = 30
    GLOBAL_FEATURES = 12

    # New headers
    headers = ["label"]
    for f in range(SEQUENCE_LENGTH):
        for v in range(NEW_VALS_PER_FRAME):
            headers.append(f"f{f}_v{v}")
            
    # Global features at the end (using names from feature_extractor)
    global_names = [
        "duration_s", "mean_speed", "peak_speed", "speed_variance",
        "path_length", "net_dx", "net_dy", "range_x", "range_y",
        "hands_used", "mean_openness", "openness_change"
    ]
    headers.extend(global_names)

    count = 0
    with open(old_file, 'r', encoding='utf-8') as f_in, \
         open(new_file, 'w', encoding='utf-8', newline='') as f_out:
        
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)
        
        # Write new header
        writer.writerow(headers)
        
        # Skip old header
        next(reader, None)
        
        for row in reader:
            if not row or len(row) < 2: continue
            label = row[0]
            data = row[1:]
            
            # Sanity check old length
            expected_old_len = (OLD_VALS_PER_FRAME * SEQUENCE_LENGTH) + GLOBAL_FEATURES
            if len(data) != expected_old_len:
                print(f"Skipping row for {label} - expected {expected_old_len} values, got {len(data)}")
                continue
                
            new_row = [label]
            
            # Process frames
            for f in range(SEQUENCE_LENGTH):
                start = f * OLD_VALS_PER_FRAME
                end = start + OLD_VALS_PER_FRAME
                frame_data = data[start:end]
                
                # Append to new row (old data + 4 zeros for new face features)
                new_row.extend(frame_data)
                new_row.extend(["0.0", "0.0", "0.0", "0.0"])  # head_yaw, head_pitch, mouth_width, mouth_pitch
                
            # Process globals (last 12 values)
            globals_data = data[-GLOBAL_FEATURES:]
            new_row.extend(globals_data)
            
            writer.writerow(new_row)
            count += 1

    print(f"Successfully migrated {count} samples to {new_file}")
    print("Backing up old file and replacing with new...")
    
    os.rename(old_file, old_file + ".old_v1")
    os.rename(new_file, old_file)
    print("Done! The dataset is now ready with face features.")

if __name__ == "__main__":
    migrate_dataset()
