import os
import ssl
import urllib.request
import zipfile
import io

def download_and_extract_dataset():
    # Bypass SSL verification to avoid proxy/local network blockages
    ssl._create_default_https_context = ssl._create_unverified_context
    
    zip_url = "https://github.com/HarshaVardhanM08/FUTURE_ML_03/archive/refs/heads/main.zip"
    dest_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(dest_dir, exist_ok=True)
    
    print("Step 1: Downloading repository ZIP containing compressed dataset...")
    try:
        response = urllib.request.urlopen(zip_url)
        zip_bytes = response.read()
        print(f"Successfully downloaded {len(zip_bytes) / (1024*1024):.2f} MB.")
    except Exception as e:
        print(f"Failed to download repository: {e}")
        return False
        
    print("Step 2: Locating and extracting Resume.csv.zip from the repository file...")
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z_repo:
            # Find the path of Resume.csv.zip inside the repo
            csv_zip_path = None
            for name in z_repo.namelist():
                if name.endswith("Resume.csv.zip"):
                    csv_zip_path = name
                    break
                    
            if not csv_zip_path:
                print("Could not find Resume.csv.zip inside the repo zip file.")
                return False
                
            print(f"Found dataset at {csv_zip_path}. Extracting...")
            csv_zip_bytes = z_repo.read(csv_zip_path)
            
            print("Step 3: Extracting Resume.csv from its nested zip file...")
            with zipfile.ZipFile(io.BytesIO(csv_zip_bytes)) as z_csv:
                # Get the name of the CSV file inside this zip
                csv_filename = None
                for name in z_csv.namelist():
                    if name.endswith(".csv"):
                        csv_filename = name
                        break
                        
                if not csv_filename:
                    print("Could not find any CSV file inside the Resume.csv.zip container.")
                    return False
                    
                csv_content = z_csv.read(csv_filename)
                target_csv_path = os.path.join(dest_dir, "Resume.csv")
                
                with open(target_csv_path, "wb") as f_out:
                    f_out.write(csv_content)
                
                print(f"Dataset successfully extracted and saved to: {target_csv_path}")
                print(f"File size: {len(csv_content) / (1024*1024):.2f} MB")
                return True
                
    except Exception as e:
        print(f"Error during extraction process: {e}")
        return False

if __name__ == "__main__":
    download_and_extract_dataset()
