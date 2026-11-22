import os
import glob
import re
import pandas as pd
import numpy as np
import pytesseract
from PIL import Image

# Setup base directory
BASE_DIR = '/Users/jaymukati/Project_CBRI/Code/knet'
OUTPUT_CSV = '/Users/jaymukati/Project_CBRI/Code/clean_kiknet_dataset.csv'
AUDIT_LOG = '/Users/jaymukati/Project_CBRI/Code/extraction_audit_log.txt'

# Fallback/Base dictionary for Lithology
LITHOLOGY_MAP = {
    'F': 'Surface soil',
    'Fl': 'Fill soil',
    'G': 'Gravel',
    'GF': 'Gravelly soil',
    'S': 'Sand',
    'SF': 'Sandy soil',
    'M': 'Silt',
    'C': 'Clay',
    'O': 'Organic soil',
    'V': 'Volcanic ash clay',
    'Pt': 'Peat',
    'R': 'Rock'
}

def run_ocr_for_legend(image_path):
    """
    Uses OCR on the provided PNG to extract the soil column legend
    and updates the LITHOLOGY_MAP.
    """
    try:
        img = Image.open(image_path)
        # Crop the right side of the image where the legend typically is
        width, height = img.size
        legend_crop = img.crop((int(width * 0.65), int(height * 0.15), width, height))
        text = pytesseract.image_to_string(legend_crop)
        
        # Look for patterns like "F: Surface soil"
        lines = text.split('\n')
        for line in lines:
            match = re.search(r'([A-Za-z]+)\s*:\s*([A-Za-z\s]+)', line)
            if match:
                abbr = match.group(1).strip()
                desc = match.group(2).strip()
                # Clean up desc
                desc = re.sub(r'[^A-Za-z\s]', '', desc).strip()
                if abbr and desc and len(abbr) <= 2:
                    LITHOLOGY_MAP[abbr] = desc
    except Exception as e:
        print(f"OCR Failed for {image_path}: {e}")

def parse_txt_file(txt_path, station_id):
    """
    Parses a single KiK-net station text file and returns a list of dictionaries.
    """
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    data = []
    intervals = []
    
    # First pass: Extract soil column intervals
    for line in lines:
        interval_match = re.search(r'(\d+\.\d+)m\s*-\s*(\d+\.\d+)m\s+([A-Za-z0-9]+)', line)
        if interval_match:
            start_depth = float(interval_match.group(1))
            end_depth = float(interval_match.group(2))
            lith = interval_match.group(3)
            intervals.append((start_depth, end_depth, lith))
            
    # Second pass: Extract tabular data
    for line in lines:
        # Match lines starting with a number followed by 'm'
        row_match = re.search(r'^\s*(\d+)m\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+\.\d+)', line)
        if row_match:
            depth = float(row_match.group(1))
            n_val = float(row_match.group(2))
            vp = float(row_match.group(3))
            vs = float(row_match.group(4))
            density = float(row_match.group(5))
            
            # Missing value handling
            if n_val == 999: n_val = np.nan
            if vp == 9999 or vp == 0: vp = np.nan
            if vs == 9999 or vs == 0: vs = np.nan
            
            # Assign lithology based on intervals
            current_lith = 'Unknown'
            for start_d, end_d, lith in intervals:
                # The tabular depth is at the bottom of the 1m interval (e.g. 1m represents 0-1m)
                # But physically, we can just check if depth is within (start, end]
                if start_d < depth <= end_d or (depth == 1.0 and start_d == 0.0):
                    current_lith = lith
                    break
            
            full_lith = LITHOLOGY_MAP.get(current_lith, current_lith)
            
            data.append({
                'Station_ID': station_id,
                'Depth_m': depth,
                'N_Value': n_val,
                'Vp_m_s': vp,
                'Vs_m_s': vs,
                'Density_g_cm3': density,
                'Lithology_Code': current_lith,
                'Lithology': full_lith
            })
            
    return data

def main():
    print("Scanning directories...")
    station_dirs = glob.glob(os.path.join(BASE_DIR, '*'))
    
    # Run OCR on the first available PNG to verify/supplement LITHOLOGY_MAP
    png_files = glob.glob(os.path.join(BASE_DIR, '*', '*.png'))
    if png_files:
        print(f"Running OCR on sample image {png_files[0]} to verify lithology categories...")
        run_ocr_for_legend(png_files[0])
        print("Updated Lithology Map based on OCR & Fallback:")
        for k, v in LITHOLOGY_MAP.items():
            print(f"  {k}: {v}")
            
    all_data = []
    parsed_count = 0
    missing_txt_count = 0
    
    print(f"Processing {len(station_dirs)} station directories...")
    for s_dir in station_dirs:
        if not os.path.isdir(s_dir):
            continue
            
        station_id = os.path.basename(s_dir)
        txt_path = os.path.join(s_dir, f"{station_id}_soil_image.txt")
        
        if os.path.exists(txt_path):
            station_data = parse_txt_file(txt_path, station_id)
            if station_data:
                all_data.extend(station_data)
                parsed_count += 1
        else:
            missing_txt_count += 1
            
    df = pd.DataFrame(all_data)
    
    # Drop rows where target (Vs) is NaN
    initial_len = len(df)
    df.dropna(subset=['Vs_m_s'], inplace=True)
    target_clean_len = len(df)
    
    # Save master dataset
    df.to_csv(OUTPUT_CSV, index=False)
    
    # Audit Log
    audit_text = f"KiK-net Data Extraction Audit Log\n"
    audit_text += "="*40 + "\n"
    audit_text += f"Total Directories Scanned: {len(station_dirs)}\n"
    audit_text += f"Successfully Parsed Stations: {parsed_count}\n"
    audit_text += f"Stations Missing TXT Files: {missing_txt_count}\n"
    audit_text += "-"*40 + "\n"
    audit_text += f"Total Extracted Rows: {initial_len}\n"
    audit_text += f"Rows Dropped (Missing Target Vs): {initial_len - target_clean_len}\n"
    audit_text += f"Final Clean Dataset Rows: {target_clean_len}\n"
    audit_text += "-"*40 + "\n"
    audit_text += "Missing Values by Feature:\n"
    audit_text += df.isnull().sum().to_string() + "\n"
    
    with open(AUDIT_LOG, 'w') as f:
        f.write(audit_text)
        
    print(f"Extraction complete! Saved to {OUTPUT_CSV}")
    print(f"Audit log saved to {AUDIT_LOG}")
    print(audit_text)

if __name__ == "__main__":
    main()
