
# 🧠 Real-Time Face Recognition System

A hybrid deep learning and machine learning face recognition project that identifies known individuals in real-time using facial embeddings and a K-Nearest Neighbors (KNN) classifier.

## 👨‍💻 Project Overview

This system:
- Detects faces using **MTCNN**
- Extracts high-dimensional facial embeddings using **InceptionResnetV1** (FaceNet)
- Classifies the face using a **KNN classifier** trained on a custom celebrity dataset
- Returns **"Unknown"** if the face is not recognized
- Visualizes face embeddings using **PCA** and **t-SNE**

## 📸 Dataset

The model is trained on images of the following five celebrities:
- John Cena
- Kobe Bryant
- Maria Sharapova
- Virat Kohli
- Cristiano Ronaldo

When an image containing any of these celebrities is passed, the system correctly identifies them. For other faces, it returns `Unknown`.

## 🧠 How It Works

1. **Face Detection**  
   Faces are detected from input images using [MTCNN](https://github.com/timesler/facenet-pytorch#mtcnn-multitask-cascaded-convolutional-networks).

2. **Face Embedding**  
   Each detected face is passed through **InceptionResnetV1** to extract a 512-dimensional feature vector representing the face.

3. **Classification with KNN**  
   These embeddings are classified using a **KNN** model. If the distance to the nearest known embedding is above a set threshold (e.g., `0.5`), it is marked as **Unknown**.

4. **Visualization**  
   PCA and t-SNE are used to visualize the clustering of embeddings to ensure separation between different identities.

## 🛠️ Tech Stack

- **Deep Learning:** MTCNN, InceptionResnetV1 (FaceNet)
- **Machine Learning:** K-Nearest Neighbors (KNN)
- **Visualization:** PCA, t-SNE
- **Libraries:**  
  `PyTorch`, `facenet-pytorch`, `scikit-learn`, `NumPy`, `Matplotlib`, `joblib`

## 📂 Project Structure

```
face-recognition/
├── dataset/
│   └── [celebrity images]
├── Video of code 
├── Image classification.ipynb
└── README.md
```

## 🚀 How to Run

1. Clone the repository:

```bash
git clone https://github.com/yourusername/face-recognition.git
cd face-recognition
```

2. Install dependencies:

```bash
pip install facenet-pytorch scikit-learn matplotlib numpy joblib
```

3. Run the notebook or script to test with new images.

## 🎥 Demo

Check out the demo video [📹 here](#) *(add your demo video link)*

## 📈 Results

- Accurately identifies known faces
- Returns `"Unknown"` for new/unseen faces
- Embeddings are cleanly clustered in 2D space using t-SNE

## 📌 TODO

- Add support for real-time webcam detection
- Improve generalization with more diverse training data
- Deploy as a web or mobile app

## 💬 Let's Connect!

If you're curious about the project or have any suggestions, feel free to reach out. Always happy to chat! 😊

---

**#MachineLearning #DeepLearning #FaceRecognition #ComputerVision #AI #PCA #tSNE #KNN #MTCNN #Facenet #Python #PyTorch #DataScience #AIProjects**
