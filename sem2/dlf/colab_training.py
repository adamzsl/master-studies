# Cross-Modal Verification - Google Colab Training
# Kopiuj i wklej do Google Colab

# =============================================
# 1. Setup i instalacja
# =============================================

# Clone repozytorium (jeśli jeszcze nie sklonowane)
!git clone https://github.com/adamzsl/master-studies.git
%cd master-studies/sem2/dlf

# Instalacja zależności
!pip install -q torch torchvision tqdm matplotlib scikit-learn Pillow

# =============================================
# 2. Przygotowanie danych
# =============================================

# Opcja A: Pobierz Flickr8k z Kaggle (wymaga Kaggle API)
# !pip install -q kaggle
# !kaggle datasets download -d adityajn105/flickr8k
# !unzip -q flickr8k.zip -d data/

# Opcja B: Utwórz minimalny dataset do testów
!python prepare_data.py data/ << EOF
y
EOF

# =============================================
# 3. Sprawdź strukturę danych
# =============================================

!ls -la data/
!ls -la data/images/ | head -10

# =============================================
# 4. Konfiguracja (opcjonalnie edytuj config.py)
# =============================================

# Możesz edytować parametry treningu bezpośrednio w Colabie
# np. zmienić liczbę epoch, batch size, etc.

# =============================================
# 5. Trening modelu
# =============================================

# Start treningu
!python train.py

# Po zakończeniu treningu sprawdź wyniki
from IPython.display import Image, display
display(Image('training_curves.png'))

# =============================================
# 6. Testowanie i walidacja submission
# =============================================

!python test_submission.py

# =============================================
# 7. Przygotowanie submission
# =============================================

# Utwórz ZIP
!zip submission.zip model.py weights.pth vocab.json

# Pobierz ZIP (w Colab)
from google.colab import files
files.download('submission.zip')

# =============================================
# 8. Metryki i analiza
# =============================================

# Załaduj i wyświetl metryki
import json
import matplotlib.pyplot as plt

# Jeśli zapisałeś metryki, możesz je załadować i zwizualizować
# (to wymaga modyfikacji train.py aby zapisać metryki do JSON)

print("✓ Training completed!")
print("✓ Download submission.zip and upload to submission system")

# =============================================
# 9. (Opcjonalnie) Test pojedynczej predykcji
# =============================================

import torch
from model import SubmissionModel, get_transform
from PIL import Image

# Załaduj model
model = SubmissionModel()
model.load_state_dict(torch.load('weights.pth', map_location='cpu'))
model.eval()

# Załaduj przykładowy obraz
transform = get_transform()
img_path = 'data/images/image0000.jpg'  # zmień na rzeczywisty obraz
image = Image.open(img_path)
image_tensor = transform(image)

# Test
test_captions = [
    "a dog playing in the park",
    "a red car on the street",
    "three birds flying"
]

print("\nTest predictions:")
for caption in test_captions:
    score = model.predict(image_tensor, caption)
    print(f"  '{caption}' -> {score:.4f} ({'MATCH' if score > 0.5 else 'NO MATCH'})")
