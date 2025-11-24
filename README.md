## 🎨 Wardrobe AI - ML-Powered Outfit Recommendation App
Android app that uses Machine Learning to classify clothing, suggest outfits based on weather, and track your wardrobe.

# 🚀 Features (Planned)
📸 Clothing classification using deep learning
🎭 Automatic background removal from clothing images
🤖 ML-based outfit recommendations
☁️ Weather-aware outfit suggestions
📊 Wardrobe analytics and insights
📱 Modern Android UI with Jetpack Compose

# 🛠️ Tech Stack
Machine Learning
Frameworks: TensorFlow, Keras, PyTorch
Models: MobileNetV2 (classification), U2-Net (segmentation)
Datasets: DeepFashion
Mobile Development
Language: Kotlin
UI: Jetpack Compose
ML Integration: TensorFlow Lite
APIs: OpenWeatherMap

# 📁 Project Structure
wardrobe-ai-android/
├── ml-training/              # Python ML training pipeline
│   ├── notebooks/            # Jupyter notebooks for experiments
│   ├── scripts/              # Production Python scripts
│   ├── models/               # Saved trained models (.h5, .pth)
│   └── datasets/             # Training datasets
│       ├── deepfashion/      # Real-world clothing images
├── android-app/              # Android application (Week 5+)
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md

# 📅 8-Week Development Timeline
Phase 1: ML Foundation (Weeks 1-4)
Week 1: Python, TensorFlow basics, Fashion-MNIST
Week 2: Clothing classification with transfer learning
Week 3: Background removal (U2-Net segmentation)
Week 4: Outfit recommendation system (Siamese networks)
Phase 2: Integration & Polish (Weeks 5-8)
Week 5: Weather API, advanced recommendation rules
Week 6: Model optimization, TensorFlow Lite conversion
Week 7: Android app development (Jetpack Compose)
Week 8: Testing, polish, deployment

# 🏃 Quick Start
Prerequisites
Python 3.8+
Git
Jupyter Notebook
10GB free disk space (for datasets)
Setup Development Environment
bash

# Clone repository
git clone https://github.com/YOUR_USERNAME/wardrobe-ai-android.git
cd wardrobe-ai-android

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook
Dataset Setup
bash

# DeepFashion (manual download required)
# 1. Visit: http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html
# 2. Download Category and Attribute Prediction Benchmark (~7GB)
# 3. Extract to: ml-training/datasets/deepfashion/

## 📊 Development Progress
# ✅ Day 1 (November 24, 2025)
 Repository created and initialized
 Project structure setup
 Python virtual environment configured
 Dependencies installed (NumPy, TensorFlow, Keras, Matplotlib, Pillow)
 Fashion-MNIST dataset download initiated
 DeepFashion dataset extraction in progress
 First commit pushed to GitHub
# Dataset Status:
DeepFashion: ⏳ Extracting (~7GB, 50,000+ images)

# 🔜 Day 2 (Planned)
 Complete dataset extraction and organization
 Explore DeepFashion dataset structure
 Create data visualization notebook
 Build first neural network (Fashion-MNIST baseline)
 Train simple classifier and evaluate accuracy
🎯 Upcoming Milestones
 Week 1 Goal: Achieve 85%+ accuracy on Fashion-MNIST
 Week 2 Goal: Production clothing classifier (90%+ accuracy)
 Week 3 Goal: Working background removal pipeline
 Week 4 Goal: ML outfit compatibility model trained
 
# 🧪 Current Status
Component	Status	Details
Environment	✅ Complete	Python 3.x, TensorFlow, Keras installed
Dataset	⏳ In Progress	Fashion-MNIST ready, DeepFashion extracting
ML Model	⏸️ Not Started	Training begins Day 2
Background Removal	⏸️ Not Started	Week 3
Recommendation	⏸️ Not Started	Week 4
Android App	⏸️ Not Started	Week 7

# 📚 Learning Resources
Currently Studying
Neural Networks fundamentals (3Blue1Brown series)
TensorFlow & Keras documentation
Computer Vision basics (Stanford CS231n)
Transfer Learning techniques

# References
TensorFlow Tutorials
Keras Documentation
DeepFashion Dataset

# 🎓 Skills Development
Technical Skills Gained (Week 1)
 Python environment setup
 Git & GitHub workflow
 NumPy array operations
 Image preprocessing
 Neural network architecture
 Model training & evaluation
Soft Skills
Daily commit discipline
Technical documentation
Progress tracking
Time management (6-8 hours/day commitment)
📝 Development Log
Detailed daily progress tracked in: docs/progress_log.md

# 🤝 Contributing
This is a personal learning project, but suggestions and feedback are welcome! Feel free to open issues or reach out.

# 📄 License
MIT License - see LICENSE file for details

# 👨‍💻 Author
Prachi

GitHub: @PrachiDhiman5
LinkedIn: [(https://www.linkedin.com/in/prachi-dhiman05/)]
Email: prachidhiman362@gmail.com
🙏 Acknowledgments
DeepFashion dataset by CUHK

Open-source ML community
Project Start Date: November 24, 2025
Expected Completion: January 19, 2026 (8 weeks)
Current Phase: Week 1 - Foundation & Dataset Preparation


🔥 Commitment
Working Schedule: 6-8 hours/day, 6 days/week
Next Update: Tomorrow (Day 2) - First ML model training

"Building something amazing, one commit at a time." 🚀

## 📋 Quick Commands
bash
# Daily workflow
git pull                    # Get latest changes
# ... do your work ...
git add .                   # Stage changes
git commit -m "Day X: ..."  # Commit with clear message
git push                    # Push to GitHub

# Activate environment
venv\Scripts\activate       # Windows

# Launch notebook
jupyter notebook

# Install new package
pip install package-name
pip freeze > requirements.txt  # Update dependencies


Next Steps:
Let dataset extraction complete
Tomorrow: Create first Jupyter notebook
Start training your first ML model!
