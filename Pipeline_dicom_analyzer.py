import os
import re
import numpy as np
import pydicom
from PIL import Image
from pathlib import Path
from pydicom.pixel_data_handlers.util import apply_voi_lut
from google import genai
from google.genai import types

class DicomConverter:
    """Handles the conversion of DICOM files to PNG format."""
    
    def __init__(self, input_dir: str | Path, output_dir: str | Path):
        self.input_path = Path(input_dir)
        self.output_path = Path(output_dir)

    def process_batch(self) -> bool:
        """Walks through the directory and converts all .dcm files."""
        if not self.input_path.exists():
            print(f"❌ Error: Input directory '{self.input_path}' not found.")
            return False

        dicom_files = list(self.input_path.rglob("*.dcm"))
        
        if not dicom_files:
            print("⚠️ No .dcm files found. Skipping conversion.")
            return False

        print(f"🔍 Found {len(dicom_files)} DICOM files. Starting conversion...")
        
        for dcm_file in dicom_files:
            relative_path = dcm_file.parent.relative_to(self.input_path)
            target_folder = self.output_path / relative_path
            self._convert_dicom_to_png(dcm_file, target_folder)
            
        return True

    def _convert_dicom_to_png(self, dicom_path: Path, output_folder: Path) -> None:
        try:
            ds = pydicom.dcmread(dicom_path)
        except Exception as e:
            print(f"❌ Failed to read {dicom_path}: {e}")
            return

        if 'PixelData' not in ds:
            print(f"⏭️ Skipping {dicom_path.name}: No image pixels found (likely a Structured Report or DICOMDIR).")
            return

        try:
            pixel_array = ds.pixel_array
        except Exception as e:
            print(f"❌ Failed to extract pixel data from {dicom_path.name}: {e}")
            return 

        try:
            data = apply_voi_lut(pixel_array, ds)
        except Exception as e:
            # If VOI LUT fails, we just use the raw pixels
            data = pixel_array 

        if ds.get('PhotometricInterpretation') == "MONOCHROME1":
            data = np.amax(data) - data

        data = data.astype(float)
        if data.max() - data.min() != 0:
            data = (data - data.min()) / (data.max() - data.min()) * 255.0
        else:
            data = np.zeros(data.shape)

        data = data.astype(np.uint8)
        output_folder.mkdir(parents=True, exist_ok=True)
        base_name = dicom_path.stem

        if data.ndim == 2:
            self._save_image(data, output_folder / f"{base_name}.png")
        elif data.ndim == 3:
            print(f"📂 Detected 3D volume ({data.shape[0]} frames) in {base_name}")
            for i, frame in enumerate(data):
                self._save_image(frame, output_folder / f"{base_name}_frame_{i:03d}.png")

    def _save_image(self, array: np.ndarray, output_path: Path) -> None:
        """Helper to save numpy array as image."""
        image = Image.fromarray(array)
        image.save(output_path)
        print(f"✅ Saved Image: {output_path}")


class MedicalAnalysisAgent:
    """Handles communicating with Gemini to analyze medical images."""
    
    def __init__(self, api_key: str, model_id: str, input_dir: str | Path, output_dir: str | Path):
        self.client = genai.Client(api_key=api_key)
        self.model_id = model_id
        self.input_path = Path(input_dir)
        self.output_path = Path(output_dir)

    def run_analysis(self) -> None:
        """Main execution flow for the AI agent."""
        self._setup_folders()
        image_groups = self._group_images_by_id()
        
        if not image_groups:
            print(f"⚠️ No PNG files found in '{self.input_path}'.")
            return

        print(f"🧠 Found {len(image_groups)} cases. Starting AI analysis...")

        for group_id, files in image_groups.items():
            response = self._analyze_group(group_id, files)
            if response:
                self._save_report(group_id, response, files)

    def _setup_folders(self) -> None:
        """Ensures the output folder exists."""
        self.output_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Reports will be saved to: {self.output_path.resolve()}")

    def _group_images_by_id(self) -> dict[str, list[Path]]:
        """
        Groups PNG files by their original DICOM filename.
        Removes '_frame_XXX' suffixes to keep multi-frame DICOMs together.
        """
        groups = {}
        for img_file in self.input_path.rglob("*.png"):
            group_id = re.sub(r'_frame_\d+$', '', img_file.stem)
            
            if group_id not in groups:
                groups[group_id] = []
            groups[group_id].append(img_file)
        return groups

    def _analyze_group(self, group_id: str, file_paths: list[Path]):
        """Sends a group of images to Gemini for comparative analysis using a strict template."""
        print(f"\n⚙️ Processing Case: {group_id}...")
        
        prompt_text = (
            f"You are an expert radiologist AI analyzing Case {group_id}. "
            "Compare all provided views for abnormalities.\n\n"
            "You MUST output your final response STRICTLY following the exact Markdown structure below. "
            "Do not add any introductory or concluding text outside of this template.\n\n"
            "## 📝 Analysis\n"
            "**Analysis of Case:** [Provide a brief high-level overview]\n"
            "**Date of Exam:**[Extract if visible, otherwise state 'Not provided']\n"
            "**Modality:**[e.g., X-Ray, CT, MRI, Ultrasound]\n"
            "**Views Provided:** [List the specific views provided]\n\n"
            "**Step-by-step Visual Findings:**\n"
            "[Describe each provided image one by one in detail]\n\n"
            "**Comparison of Views and Abnormality Analysis:**\n"
            "[Compare the views and detail any abnormalities found]\n\n"
            "## Final report\n"
            "**Clinical information:** [Detail if apparent, otherwise state 'Not provided']\n"
            "**Findings:** [Provide formal detailed radiological findings]\n"
            "**Impression:**[Provide the clinical impression here. CRITICAL: This specific section must be written in plain language readable by a patient without any prior medical background.]\n"
            "**Recommendations:** [Suggested next actionable steps or 'None']\n"
            "**Exam assessment:** [Overall assessment of exam quality and findings]"
        )
        
        prompt_parts =[prompt_text]
        
        for path in file_paths:
            with open(path, "rb") as f:
                image_bytes = f.read()
                prompt_parts.append(
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png")
                )

        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt_parts,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(include_thoughts=True)
                )
            )
            return response
        except Exception as e:
            print(f"❌ API Error for Case {group_id}: {e}")
            return None

    def _save_report(self, group_id: str, response, file_list: list[Path]) -> None:
        """Saves the thinking and response to a Markdown file."""
        safe_id = re.sub(r'[^\w\-_\.]', '_', group_id)
        filename = f"report_{safe_id}.md"
        filepath = self.output_path / filename        
        thoughts = ""
        final_text = ""
        
        candidate = response.candidates[0]
        for part in candidate.content.parts:
            if getattr(part, 'thought', False):
                thoughts += part.text
            else:
                final_text += part.text

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# Analysis Report: Case {group_id}\n\n")
            f.write("> ⚠️ **DISCLAIMER: This report is generated by an Artificial Intelligence model for informational and research purposes only. It does NOT constitute professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified healthcare provider with any questions you may have regarding a medical condition.**\n\n")
            f.write(f"**Source Images:** {', '.join([p.name for p in file_list])}\n")
            f.write("\n---\n")
            f.write("## 💭 Internal Thinking Process\n")
            f.write(f"> {thoughts.strip() if thoughts else 'No internal reasoning returned.'}\n")
            f.write("\n---\n")
            f.write(final_text.strip())
        
        print(f"✅ Saved Report: {filename}")


class ImagingPipeline:
    """Orchestrates the conversion and analysis workflows."""
    
    def __init__(self, dicom_dir: str, png_dir: str, report_dir: str, api_key: str, model_id: str):
        self.converter = DicomConverter(input_dir=dicom_dir, output_dir=png_dir)
        self.agent = MedicalAnalysisAgent(
            api_key=api_key, 
            model_id=model_id, 
            input_dir=png_dir, 
            output_dir=report_dir
        )

    def run(self):
        print("🚀 Starting Medical Imaging Pipeline...")
        print("-" * 40)
        
        # Step 1: Convert DICOMs to PNGs
        self.converter.process_batch()
        
        print("-" * 40)
        
        # Step 2: Analyze PNGs with Gemini
        self.agent.run_analysis()
        
        print("-" * 40)
        print("🎉 Pipeline Complete!")


if __name__ == "__main__":
    # Security Tip: Use os.environ.get("GEMINI_API_KEY") for production!
    API_KEY = ""
    MODEL_ID = "" # Recommended for thinking config
    
    DICOM_INPUT_FOLDER = "Pipeline_dcm_analyzer/Dicom_Input"
    PNG_OUTPUT_FOLDER = "Pipeline_dcm_analyzer/Png_Output"
    REPORTS_FOLDER = "Pipeline_dcm_analyzer/Reports"

    pipeline = ImagingPipeline(
        dicom_dir=DICOM_INPUT_FOLDER,
        png_dir=PNG_OUTPUT_FOLDER,
        report_dir=REPORTS_FOLDER,
        api_key=API_KEY,
        model_id=MODEL_ID
    )
    
    pipeline.run()
