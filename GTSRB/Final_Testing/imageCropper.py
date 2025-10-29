import pandas as pd
import cv2
import os

# Directory containing the filtered CSV and images
base_image_dir = "images"
# Directory to save cropped images
output_dir = "Cropped_Images"
# Classes to process (redundant check since CSV is already filtered, but included for robustness)
target_classes = [12, 13, 14]

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Path to the filtered CSV file
csv_file = os.path.join(base_image_dir, "filtered_classes.csv")

# Check if the CSV file exists
if not os.path.exists(csv_file):
    print(f"CSV file not found: {csv_file}")
    exit()

# Read the CSV file
try:
    df = pd.read_csv(csv_file, sep=";")
except Exception as e:
    print(f"Error reading CSV {csv_file}: {e}")
    exit()

# Filter for target classes (redundant since CSV is already filtered, but ensures robustness)
df = df[df["ClassId"].isin(target_classes)]
if df.empty:
    print(f"No data for classes {target_classes} in {csv_file}")
    exit()

# Group images by ClassId
grouped = df.groupby("ClassId")

# Process each class in the CSV
for class_id, group in grouped:
    # Create output directory for this class
    class_dir = os.path.join(output_dir, str(class_id))
    os.makedirs(class_dir, exist_ok=True)

    # Process each image in the group
    for index, row in group.iterrows():
        filename = row["Filename"]
        x1, y1 = row["Roi.X1"], row["Roi.Y1"]
        x2, y2 = row["Roi.X2"], row["Roi.Y2"]

        # Construct the full path to the image (e.g., selected_classes/00000.ppm)
        image_path = os.path.join(base_image_dir, filename)

        # Load the image
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                continue

            # Validate ROI coordinates
            if (
                x1 >= x2
                or y1 >= y2
                or x1 < 0
                or y1 < 0
                or x2 > row["Width"]
                or y2 > row["Height"]
            ):
                print(f"Invalid ROI for {filename}: ({x1}, {y1}, {x2}, {y2})")
                continue

            # Crop the image using the ROI coordinates
            cropped_image = image[y1:y2, x1:x2]

            # Save the cropped image
            output_filename = os.path.join(
                class_dir, filename.replace(".ppm", "_cropped.png")
            )
            cv2.imwrite(output_filename, cropped_image)
            print(f"Saved cropped image: {output_filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

print("Cropping complete!")
