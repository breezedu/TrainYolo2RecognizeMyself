import os
import xml.etree.ElementTree as ET

# Paths
xml_dir = "C:\\Users\\duj\\TriangleAI_Presentation\\TrainYoloWithImages\\labels"  # Directory with XML files from LabelImg
output_dir = "C:\\Users\\duj\\TriangleAI_Presentation\\TrainYoloWithImages\\labels"   # Output directory for YOLO format labels
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to convert Pascal VOC XML to YOLO format
def convert_to_yolo(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Get image dimensions
    size = root.find("size")
    img_width = int(size.find("width").text)
    img_height = int(size.find("height").text)

    # Prepare YOLO format output
    yolo_lines = []

    # Iterate over each object in the XML
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        class_id = 0  # Assuming one class (yourself), you can adjust this

        # Get bounding box information
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        # Convert to YOLO format (normalized values)
        x_center = (xmin + xmax) / 2.0 / img_width
        y_center = (ymin + ymax) / 2.0 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        # Create YOLO format line
        yolo_lines.append(f"{class_id} {x_center} {y_center} {width} {height}\n")

    # Return the YOLO formatted label
    return yolo_lines

# Process each XML file
for xml_file in os.listdir(xml_dir):
    if xml_file.endswith(".xml"):
        xml_path = os.path.join(xml_dir, xml_file)
        yolo_labels = convert_to_yolo(xml_path)

        # Save YOLO labels to txt file
        txt_filename = os.path.splitext(xml_file)[0] + ".txt"
        txt_path = os.path.join(output_dir, txt_filename)
        with open(txt_path, "w") as f:
            f.writelines(yolo_labels)

        print(f"Converted {xml_file} to {txt_filename}")
