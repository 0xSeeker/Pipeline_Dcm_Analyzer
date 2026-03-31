import pydicom
import os
import argparse

def anonymize_dicom(input_path, output_path, verbose=False):
    """
    Anonymizes a single DICOM file by clearing or modifying standard PHI tags.
    """
    if not os.path.exists(input_path):
        print(f"❌ Error: Input file '{input_path}' does not exist.")
        return

    try:
        if verbose:
            print(f"📂 Loading DICOM: {input_path}")
            
        # 1. Load the DICOM dataset
        dataset = pydicom.dcmread(input_path)

        # 2. Define the tags to anonymize
        tags_to_anonymize = [
            'PatientName', 'PatientID', 'PatientBirthDate', 'PatientSex',
            'PatientAge', 'PatientWeight', 'PatientAddress', 'InstitutionName',
            'InstitutionAddress', 'ReferringPhysicianName', 'PerformingPhysicianName',
            'StudyID', 'AccessionNumber', 'OtherPatientIDs', 'StudyDescription',
            'SeriesDescription'
        ]

        # 3. Iterate through the tags and clear/replace their values
        for tag in tags_to_anonymize:
            if tag in dataset:
                # 'PN' stands for Person Name. We replace names with 'ANONYMOUS'
                if dataset.data_element(tag).VR == 'PN':
                    setattr(dataset, tag, 'ANONYMOUS')
                    if verbose: print(f"  ➜ Anonymized {tag} to 'ANONYMOUS'")
                # For other Value Representations (VR), we set an empty string
                else:
                    setattr(dataset, tag, '')
                    if verbose: print(f"  ➜ Cleared {tag}")

        # 4. Remove private tags
        if verbose:
            print("🔍 Removing private manufacturer tags...")
        dataset.remove_private_tags()

        # 5. Save the anonymized file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        dataset.save_as(output_path)
        print(f"✅ Success! Anonymized file saved to: {output_path}")

    except Exception as e:
        print(f"❌ An error occurred while processing the DICOM file: {e}")

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anonymize a single DICOM file by removing PHI.")
    parser.add_argument("-i", "--input", required=True, help="Path to the original input DICOM file")
    parser.add_argument("-o", "--output", required=True, help="Path to save the anonymized DICOM file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()
    
    anonymize_dicom(args.input, args.output, args.verbose)
