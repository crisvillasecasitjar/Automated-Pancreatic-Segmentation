import os
import nibabel as nib

def remove_corrupted_nifti_files(directory):
    nifti_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.nii') or f.endswith('.nii.gz')]
    print(f"Found {len(nifti_files)} NIfTI files in {directory}.")
    for nifti_file in nifti_files:
        try:
            print(f"Inspecting file: {nifti_file}")
            img = nib.load(nifti_file)
            img.get_fdata()  # sI E FILE ESTÀ CORRUPTED NO PODRÉ ACCEDIR A LA IMATGE
            print(f"File {nifti_file} is valid.")
        except Exception as e:
            print(f"Corrupted file detected: {nifti_file} - Error: {e}")
            try:
                os.remove(nifti_file)
                print(f"Removed corrupted file: {nifti_file}")
            except Exception as remove_error:
                print(f"Failed to remove corrupted file: {nifti_file} - Error: {remove_error}")

remove_corrupted_nifti_files('/data/PANORAMA/cvillaseca/panorama_batch1/batch_1')
