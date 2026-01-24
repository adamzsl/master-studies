# Krok po Kroku - Implementacja Systemu Weryfikacji Krzyżowo-Modalnej

## 📚 Przegląd Zadania

Zaimplementowano kompletny system weryfikacji krzyżowo-modalnej (obraz + tekst) używając klasycznej architektury CNN + LSTM zgodnie z wymogami zadania.

---

## 🎯 Krok 1: Zrozumienie Zadania

### Cel
Stworzyć model klasyfikacji binarnej, który odpowiada na pytanie:
> "Czy podany opis tekstowy jest zgodny z treścią obrazu?"

### Wejście/Wyjście
- **Wejście:** Obraz (224×224 RGB) + Tekst (ciąg słów)
- **Wyjście:** Wartość 0.0-1.0 (0 = niezgodny, 1 = zgodny)

### Wybrana Ścieżka
**Ścieżka A - Klasyczna:**
- ✅ CNN (ResNet-50) dla obrazów
- ✅ LSTM (Bidirectional) dla tekstu
- ❌ Zakaz Transformers i CLIP

---

## 🏗️ Krok 2: Architektura Modelu

### 2.1 Komponenty

#### Image Encoder (CNN)
```python
ResNet-50 (pretrained ImageNet)
  ↓
Features (2048-d)
  ↓
Projection (2048 → 512-d)
  ↓
Final Projection (512 → 256-d)
```

**Implementacja:**
- Wykorzystanie transfer learning z ImageNet
- Usunięcie ostatniej warstwy klasyfikacji
- Dodanie warstw projekcji z Dropout (0.3)

#### Text Encoder (LSTM)
```python
Word Embeddings (256-d)
  ↓
Bidirectional LSTM (2 layers, 256 hidden)
  ↓
Last Hidden State (512-d)
  ↓
Projection (512 → 256-d)
```

**Implementacja:**
- Słownik ~10,000 słów
- Maksymalna długość sekwencji: 50 tokenów
- Dwukierunkowy LSTM dla lepszego zrozumienia kontekstu

#### Fusion Module
```python
Image Emb (256-d) + Text Emb (256-d)
  ↓
[Concatenation | Element-wise Multiply]
  ↓
MLP (768 → 256 → 128 → 1)
  ↓
Sigmoid → [0, 1]
```

**Strategie fuzji:**
1. Konkatenacja: łączy oba embeddingi
2. Hadamard product: element-wise multiplication
3. Połączenie obu dla lepszej reprezentacji

### 2.2 Parametry Modelu
- **Całkowite parametry:** 27,935,297
- **Rozmiar modelu:** ~107 MB
- **Transfer learning:** ResNet-50 pretrained

---

## 📊 Krok 3: Przygotowanie Danych

### 3.1 Wybór Datasetu

**Opcje:**
1. **Flickr8k** (zalecane dla rozpoczęcia)
   - 8,000 obrazów
   - 5 opisów na obraz
   - ~40,000 par pozytywnych

2. **COCO Captions** (dla lepszych wyników)
   - 118,287 obrazów treningowych
   - 5+ opisów na obraz
   - Większa różnorodność

### 3.2 Struktura Danych

```
data/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── captions.json  # {"image_id": ["caption1", "caption2", ...]}
```

### 3.3 Generowanie Negatywów - KLUCZOWE!

#### Hard Negatives (70% negatywów)

**Dlaczego są ważne?**
System testowy zawiera "adwersarialne" przykłady - subtelne różnice, które wykrywają czy model naprawdę rozumie relacje obraz-tekst.

**Strategia 1: Podmiana kolorów**
```python
Original: "a red car parked on the street"
Negative: "a blue car parked on the street"
```

**Strategia 2: Podmiana liczb**
```python
Original: "three dogs playing in the park"
Negative: "two dogs playing in the park"
```

**Strategia 3: Podmiana obiektów**
```python
Original: "a golden retriever running"
Negative: "a german shepherd running"
```

#### Easy Negatives (30% negatywów)

Kompletnie losowe pary:
```python
Image: [pies w parku]
Text: "a stack of delicious pancakes"
```

### 3.4 Implementacja w `dataset.py`

```python
def generate_hard_negative_caption(correct_caption):
    colors = ['red', 'blue', 'green', 'yellow', ...]
    numbers = ['one', 'two', 'three', ...]
    
    tokens = correct_caption.split()
    for i, token in enumerate(tokens):
        if token in colors:
            tokens[i] = random.choice([c for c in colors if c != token])
            return ' '.join(tokens)
    # ... similar for numbers
```

---

## 🎓 Krok 4: Proces Treningu

### 4.1 Hiperparametry (config.py)

```python
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_EPOCHS = 20
WEIGHT_DECAY = 1e-5

IMAGE_EMBEDDING_DIM = 512
TEXT_EMBEDDING_DIM = 256
HIDDEN_DIM = 256
DROPOUT = 0.3
```

### 4.2 Funkcja Straty

```python
criterion = nn.BCELoss()  # Binary Cross-Entropy
```

Odpowiednia dla klasyfikacji binarnej z wyjściem [0, 1].

### 4.3 Optymalizacja

```python
optimizer = Adam(lr=0.0001, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(mode='max', factor=0.5, patience=2)
```

**ReduceLROnPlateau:**
- Redukuje learning rate gdy accuracy przestaje rosnąć
- Pomaga w fine-tuningu modelu

### 4.4 Proces Treningu

```bash
cd sem2/dlf
python train.py
```

**Etapy:**
1. Wczytanie danych (train/val split 80/20)
2. Budowa vocabulary z captions
3. Generowanie przykładów (positive + negative)
4. Training loop:
   - Forward pass
   - Compute loss
   - Backward pass
   - Update weights
5. Validation po każdej epoce
6. Zapisanie best model

### 4.5 Metryki

Śledzone metryki:
- **Loss** (train & validation)
- **Accuracy** (główna metryka)
- **Precision** (czy predykcje "match" są poprawne)
- **Recall** (czy znajdujemy wszystkie "matches")
- **F1-Score** (harmonic mean precision/recall)

### 4.6 Monitoring

Generowane pliki:
- `training_curves.png` - wykresy loss i accuracy
- `checkpoints/best_model.pth` - najlepszy model
- `checkpoints/checkpoint_epoch_X.pth` - checkpointy co 5 epoch

---

## 📦 Krok 5: Przygotowanie Submission

### 5.1 Wymagania Submission System

**Pliki wymagane:**
1. `model.py` - zawiera klasę SubmissionModel
2. `weights.pth` - wytrenowane wagi

**Opcjonalne:**
- `vocab.json` - słownik (jeśli używasz)
- `config.py` - konfiguracja
- Własne moduły/tokeniery

### 5.2 Klasa SubmissionModel

**Wymagania:**
```python
class SubmissionModel(nn.Module):
    def __init__(self):
        # Inicjalizacja modelu
        
    def predict(self, image_tensor, text_string):
        """
        Args:
            image_tensor: torch.Tensor [3, 224, 224]
            text_string: str
        Returns:
            float: 0.0 - 1.0
        """
        # MUSI zwrócić float (użyj .item())
        return score
```

**Opcjonalnie:**
```python
def get_transform():
    """Własne transformacje obrazu"""
    return transforms.Compose([...])
```

### 5.3 Eksport Wag

Po treningu:
```python
# W train.py automatycznie:
torch.save(model.state_dict(), 'weights.pth')
```

### 5.4 Testowanie Submission

```bash
python test_submission.py
```

**Sprawdza:**
- ✓ Czy model.py istnieje
- ✓ Czy klasa SubmissionModel istnieje
- ✓ Czy metoda predict() działa
- ✓ Czy zwraca float w zakresie [0, 1]
- ✓ Czy można załadować weights.pth

### 5.5 Utworzenie ZIP

```bash
zip submission.zip model.py weights.pth vocab.json
```

---

## 🧪 Krok 6: Walidacja i Debugging

### 6.1 Unit Tests

```bash
python test_model.py
```

Testuje:
- Image encoder
- Text encoder
- Fusion module
- Pełny model
- Metoda predict()
- Tokenizacja

**Status:** ✅ 11/11 testów przechodzi

### 6.2 Typowe Problemy i Rozwiązania

#### Problem: Niska accuracy (~50%)

**Przyczyny:**
1. Model nie uczy się, tylko zgaduje
2. Zbyt łatwe negatywy
3. Learning rate za duży/mały

**Rozwiązania:**
- Zwiększ proporcję hard negatives (70% → 90%)
- Zmniejsz learning rate (0.0001 → 0.00001)
- Trenuj dłużej (20 → 50 epoch)
- Sprawdź balans klas

#### Problem: Overfitting

**Objawy:**
- Train accuracy 95%, Val accuracy 65%

**Rozwiązania:**
- Zwiększ dropout (0.3 → 0.5)
- Dodaj weight decay (1e-5 → 1e-4)
- Użyj więcej danych
- Data augmentation

#### Problem: Model ignoruje jedną modalność

**Test:**
```python
# Daj tylko obraz (losowy tekst)
score1 = model.predict(image, "random words here")

# Daj tylko tekst (losowy obraz)  
score2 = model.predict(random_image, correct_text)

# Oba powinny być niskie!
```

**Rozwiązanie:**
- Upewnij się, że hard negatives wymuszają użycie obu modalności
- Sprawdź fusion layer - czy nie faworyzuje jednej modalności

---

## 📈 Krok 7: Interpretacja Wyników

### 7.1 Wykresy Treningu

**Training Curves:**
```
Loss powinien:
- Maleć stopniowo
- Nie oscylować zbyt mocno
- Val loss nie powinien rosnąć (overfitting)

Accuracy powinien:
- Rosnąć stopniowo
- Train i Val powinny być podobne
- Osiągnąć plateau
```

### 7.2 Kryteria Oceny

| Ocena | Wymagania |
|-------|-----------|
| 3.0 | >65% accuracy (test set 1) |
| 3.5 | >70% accuracy (test set 1) |
| 4.0 | >75% accuracy (test sets 1, 2) |
| 4.5 | >80% accuracy (test sets 1, 2, 3) |
| 5.0 | >85% accuracy (test sets 1, 2, 3) |

**Uwaga:** Dla modeli Transformer progi +5%

### 7.3 Benchmark

**Baseline (losowe zgadywanie):** ~50%
**Prosty CNN+LSTM:** 65-70%
**Dobry CNN+LSTM z hard negatives:** 75-80%
**State-of-art (CLIP):** 85-90%

---

## 🚀 Krok 8: Submission do Systemu

### 8.1 Przygotowanie

1. ✅ Sprawdź submission format:
```bash
python test_submission.py
```

2. ✅ Utwórz ZIP:
```bash
zip submission.zip model.py weights.pth vocab.json
```

3. ✅ Sprawdź rozmiar:
```bash
ls -lh submission.zip
# Powinno być <100MB (wagi ResNet-50 ~100MB)
```

### 8.2 Logowanie do Systemu

```
Username: [twój GitHub username]
Password: [hasło z repozytorium]
```

### 8.3 Upload

1. Wejdź na submission system URL
2. Zaloguj się
3. Upload submission.zip
4. Czekaj na weryfikację (zestaw weryfikacyjny)
5. Czekaj na testy (~12-24h między zgłoszeniami)

### 8.4 Wyniki

System testuje:
1. **Zestaw weryfikacyjny** - test techniczny (nie liczony)
2. **Test set 1** - podstawowe testy (300 par)
3. **Test set 2** - trudniejsze testy (300 par)
4. **Test set 3** - najtrudniejsze (300 par)

---

## 📝 Krok 9: Raport

### 9.1 Wypełnij RAPORT.md

**Sekcje do wypełnienia:**
1. Parametry modelu (liczba, rozmiar)
2. Statystyki datasetu
3. Szczegóły strategii negatywów
4. Wyniki treningu (loss, accuracy)
5. Wykresy (training curves)
6. Metryki walidacyjne
7. Analiza błędów
8. Wnioski

### 9.2 Przykłady Predykcji

Dołącz:
- 3-5 poprawnych predykcji
- 3-5 błędnych predykcji z analizą

### 9.3 Analiza Ablacyjna (opcjonalnie)

Przetestuj różne konfiguracje:
- ResNet-34 vs ResNet-50
- LSTM 1 layer vs 2 layers
- Różne strategie fusion
- Wpływ hard negatives

---

## 🎓 Krok 10: Ulepszenia (dla wyższych ocen)

### Dla 4.0 (75%+)

1. **Więcej danych:**
   - Użyj COCO zamiast Flickr8k
   - Data augmentation

2. **Lepsze negatywy:**
   - Więcej strategii hard negatives
   - Attribute swapping
   - Semantic negatives

3. **Hyperparameter tuning:**
   - Grid search dla LR, dropout, etc.
   - Dłuższy trening

### Dla 4.5-5.0 (80-85%+)

1. **Zaawansowana fuzja:**
   - Attention mechanism
   - Cross-modal attention
   - Multiple fusion strategies

2. **Większy model:**
   - ResNet-101 zamiast ResNet-50
   - Głębszy LSTM (3-4 warstwy)

3. **Ensemble:**
   - Trenuj 3-5 modeli
   - Averaging/voting

4. **Augmentacja:**
   - Image: rotation, crop, color jitter
   - Text: synonym replacement, back-translation

---

## ✅ Checklist Finalna

### Przed Submission

- [ ] Model trenuje się bez błędów
- [ ] Validation accuracy >70%
- [ ] Wszystkie unit testy przechodzą
- [ ] `test_submission.py` OK
- [ ] `submission.zip` utworzony (<100MB)
- [ ] Raport wypełniony
- [ ] Wykresy dołączone

### Po Submission

- [ ] Zestaw weryfikacyjny passed
- [ ] Test set 1 results reviewed
- [ ] Ewentualne ulepszenia dla następnego submission
- [ ] Dokumentacja zaktualizowana

---

## 📚 Dodatkowe Zasoby

### Datasety
- Flickr8k: https://www.kaggle.com/datasets/adityajn105/flickr8k
- COCO: https://cocodataset.org/

### Dokumentacja
- PyTorch: https://pytorch.org/docs/
- ResNet: https://arxiv.org/abs/1512.03385
- LSTM: https://colah.github.io/posts/2015-08-Understanding-LSTMs/

### Google Colab
- Użyj `colab_training.py` dla treningu w Colab
- Darmowy GPU (Tesla T4)
- Limit ~12h sesji

---

## 🎉 Podsumowanie

Masz teraz:
✅ Kompletną implementację CNN+LSTM
✅ System generowania hard negatives
✅ Pipeline treningu i walidacji
✅ Testy i dokumentację
✅ Gotowy submission system

**Następny krok:** Pobierz prawdziwy dataset (Flickr8k/COCO) i trenuj model!

**Powodzenia w osiągnięciu 75%+ accuracy! 🚀**
