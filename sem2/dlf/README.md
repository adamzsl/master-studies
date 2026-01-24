# Cross-Modal Verification System - CNN + LSTM

Projekt realizuje weryfikację krzyżowo-modalną (obraz + tekst) przy użyciu klasycznej architektury CNN + LSTM (Ścieżka A).

## 📋 Cel zadania

System klasyfikacji binarnej sprawdzający, czy opis tekstowy jest zgodny z zawartością obrazu.

**Wejście:**
- Obraz I
- Tekst T

**Wyjście:**
- 1 - opis zgodny z obrazem
- 0 - opis niezgodny z obrazem

## 🏗️ Architektura

### Komponenty modelu:

1. **Image Encoder (CNN)**
   - Backbone: ResNet-50 (pretrenowany na ImageNet)
   - Transfer learning dla ekstrakcji cech wizualnych
   - Projekcja do 512-wymiarowej przestrzeni

2. **Text Encoder (LSTM)**
   - Dwukierunkowy LSTM (2 warstwy)
   - Word embeddings (256 wymiarów)
   - Projekcja do wspólnej przestrzeni reprezentacji

3. **Fusion Module**
   - Konkatenacja + mnożenie elementów (Hadamard product)
   - Wielowarstwowa sieć neuronowa (MLP)
   - Funkcja aktywacji: Sigmoid (wyjście 0-1)

### Architektura graficznie:

```
Image (224x224x3)          Text (słowa)
       ↓                        ↓
   ResNet-50              Word Embedding
       ↓                        ↓
  CNN Features          Bidirectional LSTM
       ↓                        ↓
  Projection             Projection
       ↓                        ↓
  Image Emb (256)        Text Emb (256)
       ↓                        ↓
       └────────┬───────────────┘
                ↓
    Fusion (concat + multiply)
                ↓
           MLP Layers
                ↓
          Sigmoid(1)
```

## 📊 Dane treningowe

### Źródła danych:
- **Flickr8k** (zalecane) - 8,000 obrazów, 5 opisów każdy
- **COCO Captions** - większy zbiór danych

### Generowanie negatywów:

**Hard Negatives (70% negatywów):**
- Zmiana kolorów: "czerwony samochód" → "niebieski samochód"
- Zmiana liczb: "trzy ptaki" → "dwa ptaki"
- Podmiana obiektów: wykorzystanie podobnych opisów

**Easy Negatives (30% negatywów):**
- Losowe pary (obraz + opis z innego obrazu)

### Podział danych:
- Training: 80%
- Validation: 20%

## 🚀 Instalacja i przygotowanie

### 1. Instalacja zależności

```bash
cd sem2/dlf
pip install -r requirements.txt
```

### 2. Przygotowanie danych

Pobierz dataset Flickr8k:
- https://www.kaggle.com/datasets/adityajn105/flickr8k

Struktura katalogów:
```
data/
  images/
    image1.jpg
    image2.jpg
    ...
  captions.json  # lub Flickr8k.token.txt
```

Przygotuj dane:
```bash
python prepare_data.py ./data
```

Dla testów można utworzyć minimalny dataset:
```bash
python prepare_data.py ./data
# Wybierz opcję utworzenia minimalnego datasetu
```

## 🎯 Trening modelu

### Podstawowy trening:

```bash
python train.py
```

### Konfiguracja treningu

Edytuj `config.py` aby zmienić parametry:

```python
# Model
IMAGE_EMBEDDING_DIM = 512
TEXT_EMBEDDING_DIM = 256
HIDDEN_DIM = 256

# Training
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_EPOCHS = 20

# Data
TRAIN_SPLIT = 0.8
NEGATIVE_RATIO = 0.5
HARD_NEGATIVE_RATIO = 0.7
```

### Monitoring treningu:

Trening generuje:
- Checkpointy co 5 epoch (`checkpoints/`)
- Best model (`checkpoints/best_model.pth`)
- Wykresy treningu (`training_curves.png`)
- Metryki: loss, accuracy, precision, recall, F1

## 📦 Przygotowanie zgłoszenia

### 1. Sprawdź format zgłoszenia:

```bash
python test_submission.py
```

### 2. Utwórz pliki zgłoszeniowe:

Po zakończeniu treningu:
- `model.py` - zawiera klasę SubmissionModel
- `weights.pth` - wagi wytrenowanego modelu

### 3. Utwórz ZIP:

```bash
python test_submission.py
# Opcjonalnie ręcznie:
zip submission.zip model.py weights.pth vocab.json
```

## 📝 Wymagania submission system

### Klasa SubmissionModel:

```python
class SubmissionModel(nn.Module):
    def predict(self, image_tensor, text_string):
        """
        Args:
            image_tensor: torch.Tensor [3, 224, 224]
            text_string: str
        Returns:
            float: 0.0 - 1.0
        """
        # ... implementacja
        return score  # Używaj .item() dla tensora
```

### Opcjonalna funkcja get_transform():

```python
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(...)
    ])
```

## 🎓 Ocenianie

| Ocena | Wymóg |
|-------|-------|
| 3.0   | >65% accuracy (zestaw 1) |
| 3.5   | >70% accuracy (zestaw 1) |
| 4.0   | >75% accuracy (zestawy 1, 2) |
| 4.5   | >80% accuracy (zestawy 1, 2, 3) |
| 5.0   | >85% accuracy (zestawy 1, 2, 3) |

## 📂 Struktura projektu

```
sem2/dlf/
├── model.py              # Model SubmissionModel
├── config.py             # Konfiguracja
├── dataset.py            # Dataset i hard negatives
├── train.py              # Skrypt treningu
├── prepare_data.py       # Przygotowanie danych
├── test_submission.py    # Walidacja zgłoszenia
├── requirements.txt      # Zależności
├── README.md            # Ten plik
├── RAPORT.md            # Raport (do wypełnienia)
├── data/                # Dane treningowe
├── checkpoints/         # Checkpointy modelu
├── weights.pth          # Finalne wagi
└── vocab.json           # Słownik
```

## 🔧 Rozwiązywanie problemów

### Problem: Brak danych
```bash
python prepare_data.py
# Utwórz minimalny dataset dla testów
```

### Problem: Mała dokładność
- Zwiększ liczbę epoch
- Dostosuj learning rate
- Zwiększ hard negative ratio
- Użyj data augmentation
- Użyj większego modelu (ResNet-101)

### Problem: Overfitting
- Zwiększ dropout
- Użyj weight decay
- Zmniejsz rozmiar modelu
- Użyj więcej danych

### Problem: Model nie uczy się
- Sprawdź balans klas (pozytywne/negatywne)
- Zmniejsz learning rate
- Sprawdź dane (czy negatywy są rzeczywiście trudne?)

## 📚 Źródła i inspiracje

- **Flickr8k Dataset**: https://www.kaggle.com/datasets/adityajn105/flickr8k
- **COCO Captions**: https://cocodataset.org/
- **ResNet Paper**: Deep Residual Learning for Image Recognition
- **LSTM**: Long Short-Term Memory Networks

## 🤝 Dodatkowe materiały

- Submission system: https://github.com/iis-siium/DLF_winter_2025_shared
- PyTorch docs: https://pytorch.org/docs/
- Transfer learning tutorial: https://pytorch.org/tutorials/

## ✅ Checklist przed submission

- [ ] Model trenuje się poprawnie
- [ ] Validation accuracy > 70%
- [ ] `test_submission.py` przechodzi wszystkie testy
- [ ] `model.py` i `weights.pth` gotowe
- [ ] `submission.zip` utworzony
- [ ] Raport wypełniony (RAPORT.md)
- [ ] Kod skomentowany i czytelny

---

**Autor:** Deep Learning Fundamentals - Zadanie 1  
**Ścieżka:** A - Klasyczna (CNN + LSTM)  
**Data:** 2025
