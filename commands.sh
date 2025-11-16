
# extract faces from videos and save images and annotations
py src/extract_faces.py --input_folder data/videos/original --output_images_dir data/images --frames_per_second 2 --nb_frames_per_time_window 1 


py src/extract_faces.py --input_folder data/videos/FaceSwap --output_images_dir data/images --frames_per_second 2 --nb_frames_per_time_window 1 