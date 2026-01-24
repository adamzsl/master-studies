# Raport - System Weryfikacji Krzyżowo-Modalnej

**Autor:** [Twoje imię i nazwisko]  
**Data:** [Data]  
**Ścieżka:** A - Klasyczna (CNN + LSTM)

---

## 1. Architektura modelu

### 1.1. Schemat architektury

```
┌─────────────────────┐         ┌─────────────────────┐
│   Image (224x224)   │         │    Text (słowa)     │
└──────────┬──────────┘         └──────────┬──────────┘
           │                               │
           ↓                               ↓
    ┌──────────────┐              ┌────────────────┐
    │  ResNet-50   │              │ Word Embedding │
    │ (pretrained) │              │   (256-dim)    │
    └──────┬───────┘              └────────┬───────┘
           │                               │
           ↓                               ↓
    ┌──────────────┐              ┌────────────────┐
    │ CNN Features │              │ Bi-LSTM (2x)   │
    │   (2048-d)   │              │   Hidden: 256  │
    └──────┬───────┘              └────────┬───────┘
           │                               │
           ↓                               ↓
    ┌──────────────┐              ┌────────────────┐
    │  Projection  │              │   Projection   │
    │   (256-d)    │              │    (256-d)     │
    └──────┬───────┘              └────────┬───────┘
           │                               │
           └───────────┬───────────────────┘
                       │
                       ↓
            ┌──────────────────┐
            │  Fusion Module   │
            │ concat + multiply │
            └─────────┬─────────┘
                      │
                      ↓
            ┌──────────────────┐
            │   MLP Layers     │
            │  512→256→128→1   │
            └─────────┬─────────┘
                      │
                      ↓
            ┌──────────────────┐
            │   Sigmoid(1)     │
            │  Output: [0, 1]  │
            └──────────────────┘
```

### 1.2. Komponenty szczegółowo

#### Image Encoder (CNN)
- **Backbone:** ResNet-50 pretrenowany na ImageNet
- **Wymiary wejściowe:** 224×224×3 RGB
- **Wymiary wyjściowe:** 2048-wymiarowe cechy
- **Projekcja:** Linear(2048 → 256) + ReLU + Dropout(0.3)
- **Transfer learning:** Wykorzystanie pretrenowanych wag

#### Text Encoder (LSTM)
- **Architektura:** Bidirectional LSTM, 2 warstwy
- **Embeddings:** 256 wymiarów, vocabulary ~10,000 słów
- **Hidden state:** 256 wymiarów na kierunek (512 razem)
- **Projekcja:** Linear(512 → 256) + ReLU + Dropout(0.3)
- **Sekwencja:** max 50 tokenów

#### Fusion Module
- **Strategie fuzji:**
  1. Konkatenacja: [img_emb, text_emb] → 512-d
  2. Element-wise multiplication: img_emb ⊙ text_emb → 256-d
  3. Połączenie: [concat, multiply] → 768-d
- **MLP:** 768 → 256 → 128 → 1
- **Funkcja aktywacji:** ReLU + Dropout(0.3) + Sigmoid

### 1.3. Parametry modelu

- **Całkowita liczba parametrów:** [do wypełnienia po treningu]
- **Trenowalne parametry:** [do wypełnienia]
- **Rozmiar modelu:** [do wypełnienia] MB

---

## 2. Dane treningowe

### 2.1. Źródło danych

- **Dataset:** [Flickr8k / COCO Captions]
- **Liczba obrazów:** [do wypełnienia]
- **Liczba opisów na obraz:** [do wypełnienia]
- **Podział:**
  - Training: [X]% ([Y] przykładów)
  - Validation: [X]% ([Y] przykładów)

### 2.2. Strategia generowania negatywów

#### Hard Negatives (70% negatywów)

**Strategia 1: Podmiana kolorów**
- Wykrycie słów kolorów w opisie
- Zastąpienie innym kolorem
- Przykład: "red car" → "blue car"

**Strategia 2: Podmiana liczb**
- Wykrycie liczebników
- Zmiana na inną liczbę
- Przykład: "three dogs" → "two dogs"

**Strategia 3: Podobne obrazy**
- Użycie opisu z semantycznie podobnego obrazu
- Np. różne rasy psów, różne pojazdy

#### Easy Negatives (30% negatywów)

- Losowe parowanie obrazu z opisem z innego, niepowiązanego obrazu
- Cel: nauka podstawowego odrzucania niepasujących par

### 2.3. Proporcje

- **Pozytywne przykłady:** [X]%
- **Negatywne przykłady:** [X]%
  - Hard negatives: [X] przykładów
  - Easy negatives: [X] przykładów

---

## 3. Proces treningu

### 3.1. Hiperparametry

| Parametr | Wartość |
|----------|---------|
| Batch size | 32 |
| Learning rate | 0.0001 |
| Optimizer | Adam |
| Weight decay | 1e-5 |
| Scheduler | ReduceLROnPlateau |
| Liczba epoch | 20 |
| Loss function | Binary Cross-Entropy |

### 3.2. Augmentacja danych

[Opisz użyte augmentacje obrazów, jeśli zastosowano]

### 3.3. Wykresy procesu treningu

#### Training & Validation Loss

[Wklej/opisz wykres training_curves.png - Loss]

**Obserwacje:**
- [Opis trendów w loss]
- [Czy występuje overfitting?]
- [Kiedy model zaczął się zbiegać?]

#### Training & Validation Accuracy

[Wklej/opisz wykres training_curves.png - Accuracy]

**Obserwacje:**
- Najlepsza validation accuracy: [X]%
- Osiągnięta w epoce: [X]
- [Inne obserwacje]

### 3.4. Wyniki walidacyjne

| Metryka | Wartość |
|---------|---------|
| Accuracy | [X]% |
| Precision | [X]% |
| Recall | [X]% |
| F1-Score | [X]% |
| Loss | [X] |

---

## 4. Analiza wyników

### 4.1. Mocne strony modelu

- [Punkt 1]
- [Punkt 2]
- [Punkt 3]

### 4.2. Słabe strony / wyzwania

- [Punkt 1]
- [Punkt 2]
- [Punkt 3]

### 4.3. Przykłady predykcji

#### Poprawne predykcje

**Przykład 1:**
- Obraz: [opis]
- Tekst: [tekst]
- Predykcja: [X] (poprawnie [match/no match])

**Przykład 2:**
[...]

#### Błędne predykcje

**Przykład 1:**
- Obraz: [opis]
- Tekst: [tekst]
- Predykcja: [X] (niepoprawnie, powinno być [Y])
- Analiza: [dlaczego model się pomylił]

---

## 5. Analiza ablacyjna (opcjonalnie)

### Eksperymenty z różnymi konfiguracjami:

| Konfiguracja | Val Accuracy | Uwagi |
|--------------|--------------|-------|
| ResNet-50 + LSTM (baseline) | [X]% | [uwagi] |
| ResNet-34 + LSTM | [X]% | [uwagi] |
| Tylko konkatenacja | [X]% | [uwagi] |
| Bez hard negatives | [X]% | [uwagi] |

**Wnioski:**
- [Co zadziałało najlepiej?]
- [Co miało największy wpływ na wyniki?]

---

## 6. Wnioski i dalszy rozwój

### 6.1. Wnioski

1. [Wniosek 1]
2. [Wniosek 2]
3. [Wniosek 3]

### 6.2. Możliwe ulepszenia

- [ ] Użycie większego datasetu (COCO)
- [ ] Bardziej zaawansowane strategie hard negatives
- [ ] Attention mechanism w fusion module
- [ ] Zwiększenie rozmiaru modelu
- [ ] Dłuższy trening / lepszy scheduler
- [ ] Ensemble methods

### 6.3. Obserwacje dot. zadania

[Twoje przemyślenia na temat zadania, wyzwań, etc.]

---

## 7. Instrukcje reprodukcji

### Środowisko:
- **Platform:** [Google Colab / Local / ...]
- **GPU:** [Tesla T4 / RTX 3090 / ...]
- **CUDA version:** [X.X]
- **PyTorch version:** [X.X]

### Kroki reprodukcji:

```bash
# 1. Przygotowanie środowiska
cd sem2/dlf
pip install -r requirements.txt

# 2. Przygotowanie danych
python prepare_data.py ./data

# 3. Trening
python train.py

# 4. Walidacja submission
python test_submission.py

# 5. Utworzenie ZIP
zip submission.zip model.py weights.pth vocab.json
```

### Czas treningu:
- Czas na epokę: ~[X] minut
- Całkowity czas treningu: ~[X] godzin

---

**Podsumowanie:** Model osiągnął [X]% accuracy na zbiorze walidacyjnym, spełniając wymagania dla oceny [X]. Głównym wyzwaniem było [X], które rozwiązano poprzez [Y].
