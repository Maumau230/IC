"""
Gera a matriz de confusao da CNN_B (contagens e normalizada) sobre o conjunto
de teste. Usa o modelo ja treinado (Resultados/Modelos/CNN_B.keras) e a MESMA divisao
treino/validacao/teste do RoadSafe_Meta1.py (random_state=42).

Correr na pasta onde esta a pasta "Road Issues" e a pasta "Resultados":
    python Gerar_Matriz_CNN_B.py
"""
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import keras

DATA_DIR = "Road Issues"
MODEL = "Resultados/Modelos/CNN_B.keras"
OUT = Path("Resultados/Matrizes")
SEED, IMG_SIZE, BATCH = 42, (128, 128), 32

def carregar(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    return tf.image.resize(img, IMG_SIZE), label


def main():
    # Garante que a pasta de saida existe
    OUT.mkdir(parents=True, exist_ok=True)

    modelo_path = MODEL
    if not Path(modelo_path).exists() and Path("Resultados/CNN_B.keras").exists():
        modelo_path = "Resultados/CNN_B.keras"

    d = Path(DATA_DIR)
    classes = sorted(p.name for p in d.iterdir() if p.is_dir())
    paths, labels = [], []
    for i, c in enumerate(classes):
        for p in (d / c).rglob("*"):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                paths.append(str(p))
                labels.append(i)
    paths, labels = np.array(paths), np.array(labels)

    _, p_tmp, _, y_tmp = train_test_split(paths, labels, test_size=0.30, stratify=labels, random_state=SEED)
    _, p_te, _, y_te = train_test_split(p_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
    print("Teste:", len(p_te))

    ds = tf.data.Dataset.from_tensor_slices((p_te, y_te)).map(carregar, deterministic=True).batch(BATCH)
    model = keras.models.load_model(modelo_path)
    pred = model.predict(ds, verbose=0).argmax(axis=1)

    curto = [c.replace(" Issues", "").replace(" issues", "") for c in classes]
    for norm, suf, fmt in [(None, "", "d"), ("true", "_Normalizada", ".2f")]:
        cm = confusion_matrix(y_te, pred, normalize=norm)
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        ConfusionMatrixDisplay(cm, display_labels=curto).plot(
            ax=ax, xticks_rotation=30, colorbar=False, values_format=fmt, cmap="Blues")
        ax.set_title("Matriz de confusao: CNN_B (conjunto de teste)")
        ax.set_xlabel("Classe prevista")
        ax.set_ylabel("Classe real")
        plt.tight_layout()
        plt.savefig(OUT / f"Matriz_Confusao_CNN_B{suf}.png", dpi=200)
        plt.close()
    print("Guardado em", OUT)


if __name__ == "__main__":
    main()