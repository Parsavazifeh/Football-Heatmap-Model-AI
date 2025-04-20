# 🏟️ Football Heatmap Position Classification

This project performs classification of football player positions based on heatmap images. Using computer vision techniques and a deep learning model (EfficientNet-B0), it processes heatmaps to predict a player's role on the field.

---

## 📌 Project Highlights

- 🖼️ Image preprocessing and color transformations to enhance heatmaps
- 🧪 Custom PyTorch Dataset & DataLoader for structured testing
- 🔁 Training with data augmentation and ImageNet normalization
- 🧠 Transfer learning using EfficientNet-B0
- 📈 Evaluation on validation and test datasets
- 📁 Outputs a CSV submission with predicted positions

---

## 🛠 Technologies Used

- Python 3.x
- PyTorch
- torchvision
- OpenCV
- PIL
- NumPy, pandas
- Matplotlib
- tqdm (progress bars)
- Google Colab (recommended for execution)

---

## 📂 Folder Structure

```
.
├── Football.ipynb               # Main notebook
├── /data/
│   └── train/                   # Subfolders for each position label
│       └── <position_name>/
│           └── *.jpg
├── /test/                       # Unlabeled test heatmaps
└── submission.csv               # Output predictions
```

---

## 🧠 Model Architecture

The project uses **EfficientNet-B0**, pre-trained on ImageNet. The classifier head is replaced with a dropout + linear layer suited to the number of football positions.

```python
model.classifier[1] = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(in_features, num_classes)
)
```

---

## 🧼 Image Preprocessing

Custom functions are used to:
- Crop the top arrow of the heatmap
- Convert yellow highlights to red
- Preserve white pitch lines
- Replace background with green

Minimal and clean preprocessing helps emphasize player activity zones.

---

## 🔁 Data Augmentation

A mild augmentation pipeline is applied:
- Random rotations (±5°)
- Color jitter
- Horizontal flips (50% chance)

This helps prevent overfitting during training.

---

## 🧪 Test Dataset

A custom `TestDataset` class handles test images without labels, returning filenames along with image tensors for final predictions.

---

## 🚀 Training Instructions

1. Mount Google Drive (for Colab users)
```python
from google.colab import drive
drive.mount('/content/drive')
```

2. Load and Split Dataset  
The training dataset is split into 80% training and 20% validation sets.

3. Train the Model  
EfficientNet is fine-tuned for 20 epochs using Adam optimizer and a learning rate scheduler.

4. Save the Model
```python
torch.save(model.state_dict(), "trained_model_efficientnet_minimal.pth")
```

---

## 📊 Evaluation

An `evaluate()` function calculates:
- Average loss
- Accuracy on validation set

Progress is monitored using `tqdm` during training.

---

## 📁 Generating Submission

Predictions on test images are written to a CSV file:
```python
submission = pd.DataFrame({
    "filename": filenames,
    "position": [class_labels[p] for p in predictions]
})
submission.to_csv("submission.csv", index=False)
```

---

## ✅ Sample Result Format

| filename | position |
|----------|----------|
| 001.jpg  | striker  |
| 002.jpg  | defender |

---

## 📌 Requirements

Install dependencies with:
```bash
pip install torch torchvision opencv-python pillow matplotlib pandas tqdm
```

Or use the provided Colab environment for ready-to-run execution.

---

## 📜 License

This project is released under the MIT License. See the LICENSE file for details.

---

## 🙌 Acknowledgements

- JetBrains Academy & Football dataset contributors
- PyTorch & torchvision community
- OpenCV for powerful image transformations

---

## 📬 Contact

For questions or suggestions, feel free to open an issue or submit a PR.

---