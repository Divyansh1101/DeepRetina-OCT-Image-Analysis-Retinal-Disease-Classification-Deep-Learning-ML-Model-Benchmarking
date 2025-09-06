# DeepRetina

DeepRetina is an intelligent OCT (Optical Coherence Tomography) image analysis platform for **retinal disease classification, benchmarking, and visualization**.  
It provides a clean API and web-based interface to explore OCT datasets, evaluate multiple ML/DL models, and visualize results in a clinically relevant manner.

## 🚀 Features
- **Disease Classification**: Deep learning (CNN, Vision Transformer) and ML (SVM, Random Forest) models.  
- **Benchmarking**: Compare model accuracy, AUC, and inference speed.  
- **Preprocessing & Quality Check**: Automated OCT image preprocessing and quality assessment.  
- **Visualization**: Advanced visualization of predictions, heatmaps, and retinal biomarkers.  
- **Web Interface**: User-friendly Flask-based web app with Docker support.

## 📂 Project Structure
```
Deep-Retina/
  app.py                  # Flask entry point
  requirements.txt        # Dependencies
  Dockerfile              # Docker setup
  utils/                  # Core modules
    preprocessing.py
    model_loader.py
    benchmarking.py
    quality_assessment.py
    visualization.py
  templates/              # HTML frontend
  static/                 # CSS, JS, Images
```

## 🔧 Installation

1. Clone this repo:
   ```bash
   git clone https://github.com/your-username/deepretina.git
   cd deepretina
   ```

2. Create a virtual environment & install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   python app.py
   ```

4. Visit in browser: `http://127.0.0.1:5000`

## 🧪 Usage
- Upload OCT images to test disease classification.  
- Compare multiple models side by side with benchmarking results.  
- Visualize quality assessment and retinal disease heatmaps.  

## 📊 Example Outputs
- Accuracy & AUC comparison across CNN, ViT, and ML models.  
- Quality assessment flags for noisy/poor-quality OCT scans.  
- Heatmap overlays highlighting disease-relevant retinal regions.  

## 🐳 Docker Support
```bash
docker build -t deepretina .
docker run -p 5000:5000 deepretina
```

## 🤝 Contributing
Pull requests are welcome! Please open an issue first to discuss major changes.

## 📜 License
This project is licensed under the [MIT License](LICENSE).
