import cv2
import numpy as np
import csv
import random
import os
import math
from tqdm import tqdm

# Constants
IMAGE_SIZE = (500, 500)
ARROW_LENGTH = 300
ARROW_CANVAS_SIZE = int(math.ceil(math.sqrt(2) * ARROW_LENGTH))  # Diagonal of the bounding box
OUTPUT_DIR = "arrows"
CSV_FILE = "arrow_data_big.csv"
NUM_ARROWS = 200_000  # Set the number of arrows you want to generate here
FINAL_SIZE = (56, 56)

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# Create a blank image
def create_blank_image(size=IMAGE_SIZE):
    return np.ones(size + (3,), np.uint8) * 255


# Create an arrow image
def create_arrow_image():
    img = create_blank_image((ARROW_CANVAS_SIZE, ARROW_CANVAS_SIZE))
    pt1 = (ARROW_CANVAS_SIZE // 2, ARROW_CANVAS_SIZE // 2 - ARROW_LENGTH // 2)
    pt2 = (ARROW_CANVAS_SIZE // 2, ARROW_CANVAS_SIZE // 2 + ARROW_LENGTH // 2)
    cv2.arrowedLine(img, pt1, pt2, (0, 0, 0), 50, tipLength=0.5)
    return img


# Rotate the image
def rotate_image(image, angle):
    center = (float(ARROW_CANVAS_SIZE) // 2, float(ARROW_CANVAS_SIZE) // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rot_mat, (ARROW_CANVAS_SIZE, ARROW_CANVAS_SIZE), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return rotated


# Place the rotated arrow on the main canvas
def place_on_canvas(rotated_arrow):
    canvas = create_blank_image()
    x_offset = (IMAGE_SIZE[0] - ARROW_CANVAS_SIZE) // 2
    y_offset = (IMAGE_SIZE[1] - ARROW_CANVAS_SIZE) // 2
    canvas[y_offset:y_offset + rotated_arrow.shape[0], x_offset:x_offset + rotated_arrow.shape[1]] = rotated_arrow
    return canvas


# Determine direction
def determine_direction(angle):
    if 22.5 <= angle < 67.5:
        return "South East"
    elif 67.5 <= angle < 112.5:
        return "East"
    elif 112.5 <= angle < 157.5:
        return "North East"
    elif 157.5 <= angle < 202.5:
        return "North"
    elif 202.5 <= angle < 247.5:
        return "North West"
    elif 247.5 <= angle < 292.5:
        return "West"
    elif 292.5 <= angle < 337.5:
        return "South West"
    else:
        return "South"


# Main function
def main():
    for arrow_count in tqdm(range(1, NUM_ARROWS + 1), desc="Generating arrows"):  # Using tqdm for progress bar
        angle = random.randint(0, 359)
        direction = determine_direction(angle)

        arrow = create_arrow_image()
        rotated_arrow = rotate_image(arrow, angle)
        final_image = place_on_canvas(rotated_arrow)

        # Scale down the final image to 28x28
        scaled_image = cv2.resize(final_image, FINAL_SIZE, interpolation=cv2.INTER_AREA)

        filename = os.path.join(OUTPUT_DIR, f"arrow_{arrow_count}.png")
        cv2.imwrite(filename, scaled_image)

        with open(CSV_FILE, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            if arrow_count == 1:
                csvwriter.writerow(["Arrow Number", "Angle", "Direction"])
            csvwriter.writerow([f"arrow_{arrow_count}.png", angle, direction])


if __name__ == "__main__":
    main()
