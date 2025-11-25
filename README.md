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
deepfashion/
├── Img/
│   ├── img/
│   │   ├── Blouses_Shirts/
│   │   │   ├── img_00000001.jpg
│   │   │   ├── img_00000002.jpg
│   │   │   └── ...
│   │   ├── Cardigans/
│   │   ├── Dresses/
│   │   ├── Graphic_Tees/
│   │   ├── Jackets_Coats/
│   │   ├── Jeans/
│   │   ├── Pants/
│   │   ├── Rompers_Jumpsuits/
│   │   ├── Shorts/
│   │   ├── Skirts/
│   │   ├── Sweaters/
│   │   ├── Sweatshirts_Hoodies/
│   │   ├── Tees_Tanks/
│   │   └── Vests/
│   └── (img-002 contents merged here)
│
├── Anno/
│   ├── list_attr_cloth.txt
│   ├── list_attr_img.txt
│   ├── list_bbox_cloth.txt
│   ├── list_bbox_inshop.txt
│   ├── list_category_cloth.txt
│   ├── list_category_img.txt
│   └── (other annotation files)
│
└── Eval/
    └── list_eval_partition.txt

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
git clone https://github.com/PrachiDhiman5/wardrobe-ai-android.git
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

🎨 Wardrobe AI - ML-Powered Outfit Recommendation App
Android app that uses Machine Learning to classify clothing, suggest outfits based on weather, and track your wardrobe intelligently.

🌟 Project Overview
Wardrobe AI is a comprehensive Android application that combines computer vision, machine learning, and smart recommendations to help users manage their wardrobe and get outfit suggestions. The app uses real-world fashion datasets and state-of-the-art ML models to provide personalized fashion assistance.

🚀 Features
Core Features (Implemented/In Progress)

📸 Clothing Classification - ML-powered identification of clothing types
🎭 Background Removal - Clean catalog images with segmentation
🤖 ML-based Outfit Recommendations - Smart outfit pairing using Siamese networks
☁️ Weather-aware Suggestions - Context-based outfit recommendations
📊 Wardrobe Analytics - Track your clothing usage and patterns
🌍 Multi-language Support - Localization for global users
🎨 Modern UI - Built with Jetpack Compose

Advanced Features (Planned)

🔄 Outfit History Tracking - Never repeat the same look
💰 Cost per Wear Analysis - Smart shopping decisions
🎒 Packing List Generator - AI-powered travel packing
🔐 Privacy-focused - All ML processing on-device

🛠️ Tech Stack
Machine Learning

Framework: TensorFlow, Keras, PyTorch
Models:

MobileNetV2 (Transfer Learning for Classification)
U2-Net (Background Removal/Segmentation)
Siamese Network (Outfit Compatibility)


Deployment: TensorFlow Lite for on-device inference

Mobile Development

Language: Kotlin
UI Framework: Jetpack Compose
Architecture: MVVM (Model-View-ViewModel)
ML Integration: TensorFlow Lite, ML Kit

APIs & Services

Weather: OpenWeatherMap API
Storage: On-device (privacy-first)

Dataset

DeepFashion: Category and Attribute Prediction Benchmark

551,410+ high-quality fashion images
5,000+ detailed clothing categories
Professional photography with annotations



📁 Project Structure
wardrobe-ai-android/
├── ml-training/              # Python ML training code
│   ├── notebooks/            # Jupyter notebooks for exploration
│   │   ├── 01_deepfashion_exploration.ipynb
│   │   ├── 02_classification_model.ipynb
│   │   ├── 03_segmentation.ipynb
│   │   └── 04_recommendations.ipynb
│   ├── scripts/              # Python training scripts
│   │   ├── train_classifier.py
│   │   ├── train_segmentation.py
│   │   └── train_recommendations.py
│   ├── models/               # Trained models (.h5, .tflite)
│   │   ├── clothing_classifier.tflite
│   │   ├── background_removal.tflite
│   │   └── outfit_compatibility.tflite
│   └── datasets/             # Training data
│       └── deepfashion/
│           ├── Img/          # 551,410 fashion images
│           ├── Anno/         # Category & attribute annotations
│           └── Eval/         # Train/val/test splits
├── android-app/              # Android Kotlin application
│   ├── app/
│   │   ├── src/
│   │   │   ├── main/
│   │   │   │   ├── java/com/wardrobe/ai/
│   │   │   │   │   ├── ui/          # Jetpack Compose screens
│   │   │   │   │   ├── ml/          # ML model integration
│   │   │   │   │   ├── data/        # Data layer
│   │   │   │   │   └── utils/       # Utilities
│   │   │   │   ├── res/             # Resources
│   │   │   │   └── assets/          # TFLite models
│   │   └── build.gradle
├── docs/                     # Documentation & logs
│   ├── progress_log.md       # Daily development log
│   ├── architecture.md       # System architecture
│   └── screenshots/          # App screenshots
├── README.md
├── requirements.txt          # Python dependencies
└── .gitignore
🏃 Quick Start
Prerequisites

Python: 3.8+ (for ML training)
Android Studio: Latest version (for Android app)
Git: For version control
Jupyter Notebook: For exploration and training

Setup Python Environment
bash# Clone repository
git clone https://github.com/YOUR_USERNAME/wardrobe-ai-android.git
cd wardrobe-ai-android

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter for ML training
jupyter notebook
Setup Android Development
bash# Open Android Studio
# File → Open → Select android-app/ folder

# Sync Gradle dependencies
# Build → Make Project

# Run on emulator or device
# Run → Run 'app'
📊 Dataset Information
DeepFashion - Category and Attribute Prediction Benchmark

Total Images: 551,410
Categories: 5,000+ detailed clothing types
Size: ~4.85 GB
Format: JPG (varying sizes, standardized to 224x224 for training)
Splits: Pre-divided into train/validation/test sets
Annotations:

Category labels
Attribute annotations (color, pattern, style)
Bounding boxes
Fine-grained attributes



Sample Categories

Blouses & Shirts
Cardigans
Dresses
Graphic Tees
Jackets & Coats
Jeans
Pants
Rompers & Jumpsuits
Shorts
Skirts
Sweaters
Sweatshirts & Hoodies
Tees & Tanks
Vests

🎓 Learning Path
This project follows a structured 8-week learning path covering:
Week 1: Foundation & Dataset

ML fundamentals
Python for image processing
Dataset exploration
Data quality analysis

Week 2: Classification Model

Transfer learning with MobileNetV2
Data augmentation
Model training and evaluation
TensorFlow Lite conversion

Week 3: Background Removal

Image segmentation concepts
U2-Net implementation
Batch processing pipeline
Model optimization

Week 4: Outfit Recommendations

Recommendation systems theory
Feature extraction (color, style, texture)
Siamese network architecture
Compatibility scoring

Week 5: Weather Integration

OpenWeatherMap API integration
Context-aware filtering
Color theory rules
Style matching algorithms

Week 6: Android Integration

TensorFlow Lite in Android
Camera integration
On-device inference
UI/UX with Jetpack Compose

Week 7: Navigation & State

Compose Navigation
State management
Multi-screen flows
Data persistence

Week 8: Localization & Polish

Multi-language support
Permissions handling
Testing and debugging
Documentation

📈 Development Progress
✅ Completed

 Project structure setup
 GitHub repository initialized
 Python environment configured
 DeepFashion dataset downloaded (551,410 images)
 Dataset exploration and analysis
 Category distribution analysis
 Image properties examination

🔄 In Progress

 Data preprocessing pipeline
 MobileNetV2 transfer learning
 Background removal with U2-Net
 Outfit recommendation model

🔜 Upcoming

 Android app skeleton
 Camera integration
 ML model deployment to Android
 Weather API integration
 UI/UX implementation
 Multi-language support
 Testing and optimization


Fork the repository
Create a feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
👨‍💻 Author
Prachi - Student Developer
📧 prachidhiman362@gmail.com
🔗 [Linkedin](https://www.linkedin.com/in/prachi-dhiman05/)
🐱 [GitHub](https://github.com/PrachiDhiman5)

🙏 Acknowledgments
DeepFashion Dataset - For providing comprehensive fashion image data
TensorFlow Team - For excellent ML frameworks and documentation
Android Developer Community - For Jetpack Compose resources
Fashion-MNIST - For initial learning and prototyping
Claude AI - For guidance and mentorship throughout development

📚 Resources & References
ML & Deep Learning

TensorFlow Documentation
Keras Applications
DeepFashion Dataset
Fashion-MNIST

Android Development

Jetpack Compose
Android ML Kit
TensorFlow Lite for Android
Kotlin Documentation

APIs

OpenWeatherMap API

📊 Project Statistics

Lines of Code: Growing daily
ML Models: 3 (Classification, Segmentation, Recommendation)
Dataset Size: 551,410 images (4.85 GB)
Target Platforms: Android 8.0+ (API 26+)
Development Time: 8 weeks (intensive learning)

🚀 Future Enhancements

 Cloud sync for wardrobe data
 Social features (outfit sharing)
 AR try-on integration
 Shopping integration
 Style trends analysis
 Sustainability metrics
 Capsule wardrobe suggestions
 Seasonal wardrobe rotation
