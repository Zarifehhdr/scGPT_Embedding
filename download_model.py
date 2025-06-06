import os
import gdown

def download_files():
    # Create a directory for the model files if it doesn't exist
    os.makedirs('model_files', exist_ok=True)
    
    # The folder ID from the Google Drive link
    folder_id = "1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y"
    
    print("Downloading files from Google Drive folder...")
    gdown.download_folder(id=folder_id, output="model_files", quiet=False)
    print("Download completed!")

if __name__ == "__main__":
    download_files() 