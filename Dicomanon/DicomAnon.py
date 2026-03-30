import pydicom
import os

def anonymize_dicom(input_path, output_path):
    """
    Anonymizes a single DICOM file by clearing or modifying standard PHI tags.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' does not exist.")
        return

    try:
        # 1. Load the DICOM dataset
        dataset = pydicom.dcmread(input_path)

        # 2. Define the tags to anonymize
        # This list covers the most common PHI (Protected Health Information) fields.
        tags_to_anonymize =[
            'PatientName',
            'PatientID',
            'PatientBirthDate',
            'PatientSex',
            'PatientAge',
            'PatientWeight',
            'PatientAddress',
            'InstitutionName',
            'InstitutionAddress',
            'ReferringPhysicianName',
            'PerformingPhysicianName',
            'StudyID',
            'AccessionNumber',
            'OtherPatientIDs',
            'StudyDescription',
            'SeriesDescription'
        ]

        # 3. Iterate through the tags and clear/replace their values
        for tag in tags_to_anonymize:
            if tag in dataset:
                # 'PN' stands for Person Name. We replace names with 'ANONYMOUS'
                if dataset.data_element(tag).VR == 'PN':
                    setattr(dataset, tag, 'ANONYMOUS')
                # For other Value Representations (VR), we set an empty string
                else:
                    setattr(dataset, tag, '')

        # 4. Remove private tags
        # Medical equipment vendors often store proprietary or identifying data in private tags
        dataset.remove_private_tags()

        # 5. Save the anonymized file
        dataset.save_as(output_path)
        print(f"Success! Anonymized file saved to: {output_path}")

    except Exception as e:
        print(f"An error occurred while processing the DICOM file: {e}")

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # Replace these paths with your actual file paths
    input_dicom = "/home/babayaga/Téléchargements/c8711fb4-5f10-4396-acb7-34f7aa42ba97/DICOM/IMG0"
    output_dicom = "/home/babayaga/Documents/projet/Pipeline_dcm_analyzer/dicom_input/IMG_0.dcm"
    
    anonymize_dicom(input_dicom, output_dicom)