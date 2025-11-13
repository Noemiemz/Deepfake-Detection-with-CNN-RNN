
# extract faces from videos and save images and annotations
py src/extract_faces.py \
   --input_folder data/videos \
   --output_images_dir data/images \
   --output_annotations_dir data/annotations \
   --metadata_csv data/metadata.csv \
   --frames_per_second 2