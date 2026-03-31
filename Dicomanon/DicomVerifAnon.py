import pydicom
import os
import argparse

def verify_anonymization(original_path, anonymized_path, verbose=False):
    """
    Compares the original and anonymized DICOM files to verify PHI removal.
    """
    if not os.path.exists(original_path):
        print(f"❌ Error: Original file '{original_path}' not found.")
        return
    if not os.path.exists(anonymized_path):
        print(f"❌ Error: Anonymized file '{anonymized_path}' not found.")
        return

    if verbose:
        print(f"📂 Loading original: {original_path}")
        print(f"📂 Loading anonymized: {anonymized_path}")

    # Load datasets
    try:
        ds_orig = pydicom.dcmread(original_path)
        ds_anon = pydicom.dcmread(anonymized_path)
    except Exception as e:
        print(f"❌ Error reading DICOM files: {e}")
        return

    tags_to_check = [
        'PatientName', 'PatientID', 'PatientBirthDate', 'PatientSex',
        'PatientAge', 'PatientWeight', 'PatientAddress', 'InstitutionName',
        'InstitutionAddress', 'ReferringPhysicianName', 'PerformingPhysicianName',
        'StudyID', 'AccessionNumber', 'OtherPatientIDs', 'StudyDescription', 
        'SeriesDescription'
    ]

    print("\n" + "="*85)
    print(f"{'DICOM Tag':<25} | {'Original Value':<25} | {'Anonymized Value':<25}")
    print("="*85)

    all_passed = True

    # 1. Check Standard PHI Tags
    for tag_name in tags_to_check:
        # Get values (convert to string, replace empty with '<Empty>')
        orig_val = str(getattr(ds_orig, tag_name, '<Not Present>'))
        anon_val = str(getattr(ds_anon, tag_name, '<Not Present>'))
        
        # Clean up empty strings for display
        if orig_val == '': orig_val = '<Empty>'
        if anon_val == '': anon_val = '<Empty>'

        # Truncate long strings for neat table formatting
        orig_val_disp = (orig_val[:22] + '...') if len(orig_val) > 25 else orig_val
        anon_val_disp = (anon_val[:22] + '...') if len(anon_val) > 25 else anon_val

        print(f"{tag_name:<25} | {orig_val_disp:<25} | {anon_val_disp:<25}")

        # Basic logic to flag if a tag was not cleared
        if orig_val not in ['<Not Present>', '<Empty>']:
            if anon_val not in ['<Not Present>', '<Empty>', 'ANONYMOUS']:
                all_passed = False
                if verbose:
                    print(f"  ⚠️ Warning: {tag_name} does not appear completely anonymized.")

    print("-" * 85)

    # 2. Check Private Tags
    def has_private_tags(dataset):
        for element in dataset:
            if element.tag.is_private:
                return True
        return False

    orig_has_private = has_private_tags(ds_orig)
    anon_has_private = has_private_tags(ds_anon)

    print(f"{'Contains Private Tags?':<25} | {str(orig_has_private):<25} | {str(anon_has_private):<25}")
    
    if anon_has_private:
        all_passed = False
        if verbose:
            print("  ⚠️ Warning: Anonymized file still contains private tags.")

    print("=" * 85)

    # 3. Final Conclusion
    print("\nVERIFICATION RESULT:")
    if all_passed:
        print("✅ SUCCESS: All targeted PHI fields have been cleared/anonymized, and no private tags remain.")
    else:
        print("❌ FAILED: Some PHI fields or private tags were not properly anonymized. Check the table above.")


# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify DICOM anonymization by comparing files.")
    parser.add_argument("-orig", "--original", required=True, help="Path to the original DICOM file")
    parser.add_argument("-anon", "--anonymized", required=True, help="Path to the anonymized DICOM file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()
    
    verify_anonymization(args.original, args.anonymized, args.verbose)
