## Dataset name: 0604dataset (Roboflow)

- Number of images: 1,745
- Format: YOLOv5
- Classes: bicyle, car, crosswalk,kickboard, manhole, motorcycle, road, stairs, stop, straight

**Note:**  
Due to a typo during dataset preparation, the class "bicycle" is labeled as **"bicyle"** in both annotations and training.  
Please be aware that the model uses "bicyle" as the class name internally.

- Download: [Roboflow Project Page](https://app.roboflow.com/ossprojobjectdetectionpractice/0604dataset/4)

## Training Code

The model was trained using Google Colab with the YOLOv5s architecture.  
Key training settings are as follows:

- Image size: 640x640  
- Batch size: 64  
- Epochs: 300  
- Model: YOLOv5s  
- Data augmentation: Not applied

## YOLOv5 Training Code (Google Colab)

```python
# Clone YOLOv5 repository
!git clone https://github.com/ultralytics/yolov5

# Move into the yolov5 directory
%cd yolov5

# Install required dependencies
!pip install -r requirements.txt

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install Roboflow
!pip install roboflow

# Import Roboflow and download dataset
from roboflow import Roboflow
rf = Roboflow(api_key="")  # Insert your API key
project = rf.workspace("").project("")  # Insert workspace and project name
dataset = project.version().download("yolov5")  # Use correct format version

# Train YOLOv5
!python train.py --img 640 --batch 64 --epochs 300 --data data.yaml --weights yolov5s.pt```
**Final output:** `best.pt` was generated as the final trained model.

