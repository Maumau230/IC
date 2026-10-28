from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow import layers

# ---------- Configuracoes ----------
PASTA_DADOS = "Road Issues"
PASTA_SAIDA = Path("Resultados")

# Subpastas organizadas
PASTA_CURVAS = PASTA_SAIDA / "Curvas"
PASTA_MATRIZES = PASTA_SAIDA / "Matrizes"
PASTA_RELATORIOS = PASTA_SAIDA / "Relatorios"
PASTA_MODELOS = PASTA_SAIDA / "Modelos"

SEED = 42
TAMANHO_IMG = (128, 128)
BATCH = 32
EPOCAS = 40
PACIENCIA = 8
POTENCIA_PESOS = 1.0

# Modelos a treinar nesta execucao
A_CORRER = ["MLP_B_cw", "CNN_A_cw", "CNN_B_cw"]


# ---------- Preparacao dos Dados ----------
def listar_imagens(pasta):
    pasta = Path(pasta)
    classes = sorted(d.name for d in pasta.iterdir() if d.is_dir())
    caminhos, rotulos = [], []
    for i, nome_classe in enumerate(classes):
        for f in (pasta / nome_classe).rglob("*"):
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                caminhos.append(str(f))
                rotulos.append(i)
    return classes, np.array(caminhos), np.array(rotulos)


def carregar(caminho, rotulo):
    img = tf.io.read_file(caminho)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, TAMANHO_IMG)
    return img, rotulo


def criar_dataset(caminhos, rotulos, treino=False):
    ds = tf.data.Dataset.from_tensor_slices((caminhos, rotulos))
    if treino:
        # Shuffle a cada epoca durante o treino
        ds = ds.shuffle(
            buffer_size=len(caminhos), seed=SEED, reshuffle_each_iteration=True
        )
    # deterministic=True garante que a ordem no teste coincide sempre com os rotulos reais
    ds = ds.map(carregar, num_parallel_calls=tf.data.AUTOTUNE, deterministic=not treino)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)


def preparar_splits(caminhos, rotulos):
    """
    Divisao: 70% treino, 15% validacao e 15% teste, sempre estratificado.
    """
    x_tr, x_resto, y_tr, y_resto = train_test_split(
        caminhos, rotulos, test_size=0.30, stratify=rotulos, random_state=SEED
    )
    x_val, x_te, y_val, y_te = train_test_split(
        x_resto, y_resto, test_size=0.50, stratify=y_resto, random_state=SEED
    )
    return (x_tr, y_tr), (x_val, y_val), (x_te, y_te)


# ---------- Modelos ----------
def compilar(modelo, lr):
    modelo.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modelo


def criar_mlp(n_classes, camadas_ocultas, dropout, lr, nome):
    m = keras.Sequential(name=nome)
    m.add(layers.Input(TAMANHO_IMG + (3,)))
    m.add(layers.Rescaling(1.0 / 255))
    m.add(layers.Flatten())
    for neuronios in camadas_ocultas:
        m.add(layers.Dense(neuronios, activation="relu"))
        m.add(layers.Dropout(dropout))
    m.add(layers.Dense(n_classes, activation="softmax"))
    return compilar(m, lr)


def criar_cnn(n_classes, filtros, dropout, lr, nome, augment=True, batchnorm=False):
    m = keras.Sequential(name=nome)
    m.add(layers.Input(TAMANHO_IMG + (3,)))
    if augment:
        m.add(layers.RandomFlip("horizontal"))
        m.add(layers.RandomRotation(0.05))
        m.add(layers.RandomZoom(0.1))
    m.add(layers.Rescaling(1.0 / 255))
    for f in filtros:
        m.add(layers.Conv2D(f, 3, padding="same", activation="relu"))
        if batchnorm:
            m.add(layers.BatchNormalization())
        m.add(layers.MaxPooling2D())
    m.add(layers.GlobalAveragePooling2D())
    m.add(layers.Dense(128, activation="relu"))
    m.add(layers.Dropout(dropout))
    m.add(layers.Dense(n_classes, activation="softmax"))
    return compilar(m, lr)


def obter_modelos(n_classes):
    return {
        # Ronda 1 (sem pesos)
        "MLP_A": (lambda: criar_mlp(n_classes, [256, 128], 0.3, 1e-3, "MLP_A"), False),
        "MLP_B": (lambda: criar_mlp(n_classes, [512, 256, 128], 0.4, 5e-4, "MLP_B"), False),
        "CNN_A": (lambda: criar_cnn(n_classes, [32, 64, 128], 0.4, 1e-3, "CNN_A"), False),
        "CNN_B": (
            lambda: criar_cnn(n_classes, [32, 64, 128, 256], 0.4, 5e-4, "CNN_B", batchnorm=True),
            False,
        ),
        # Ronda 2 (com pesos de classe)
        "MLP_B_cw": (lambda: criar_mlp(n_classes, [512, 256, 128], 0.4, 5e-4, "MLP_B_cw"), True),
        "CNN_A_cw": (lambda: criar_cnn(n_classes, [32, 64, 128], 0.4, 1e-3, "CNN_A_cw"), True),
        "CNN_B_cw": (
            lambda: criar_cnn(n_classes, [32, 64, 128, 256], 0.4, 5e-4, "CNN_B_cw", batchnorm=True),
            True,
        ),
    }


# ---------- Avaliacao e Visualizacoes ----------
def avaliar(modelo, ds_teste, y_true, classes):
    probs = modelo.predict(ds_teste, verbose=0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    previsto = probs.argmax(axis=1)
    cm = confusion_matrix(y_true, previsto)

    # Especificidade One-vs-Rest para cada classe
    n_classes = len(classes)
    especificidades = []
    for i in range(n_classes):
        tp = cm[i, i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        especificidades.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)

    metricas = {
        "accuracy": float(accuracy_score(y_true, previsto)),
        "recall_macro": float(recall_score(y_true, previsto, average="macro", zero_division=0)),
        "especificidade_macro": float(np.mean(especificidades)),
        "f1_macro": float(f1_score(y_true, previsto, average="macro", zero_division=0)),
        "auc_macro_ovr": float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro")),
    }
    recall_por_classe = {
        f"recall_{c}": float(cm[i, i] / cm[i].sum()) if cm[i].sum() > 0 else 0.0
        for i, c in enumerate(classes)
    }
    return metricas, recall_por_classe, cm, previsto


def guardar_curvas(hist, nome, pasta_curvas):
    pasta_curvas.mkdir(parents=True, exist_ok=True)
    epocas = range(1, len(hist.history["loss"]) + 1)
    fig, eixos = plt.subplots(1, 2, figsize=(10, 4))

    for eixo, chave, titulo in [
        (eixos[0], "loss", "Loss"),
        (eixos[1], "accuracy", "Accuracy"),
    ]:
        eixo.plot(epocas, hist.history[chave], label="Treino")
        eixo.plot(epocas, hist.history["val_" + chave], label="Validacao")
        eixo.set_title(f"{nome}: {titulo}")
        eixo.set_xlabel("Epoca")
        eixo.set_ylabel(titulo)
        eixo.grid(True, linestyle="--", alpha=0.5)
        eixo.legend()

    plt.tight_layout()
    plt.savefig(pasta_curvas / f"Curvas_{nome}.png", dpi=150)
    plt.close()


def guardar_matrizes(nome, y_true, previsto, classes, pasta_matrizes):
    pasta_matrizes.mkdir(parents=True, exist_ok=True)
    rotulos_curtos = [c.replace(" Issues", "").replace(" issues", "") for c in classes]

    for normalizar, sufixo, formato in [(None, "", "d"), ("true", "_Normalizada", ".2f")]:
        cm = confusion_matrix(y_true, previsto, normalize=normalizar)
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        ConfusionMatrixDisplay(cm, display_labels=rotulos_curtos).plot(
            ax=ax, xticks_rotation=30, colorbar=False, values_format=formato, cmap="Blues"
        )
        ax.set_title(f"Matriz de confusao - {nome} (teste)")
        ax.set_xlabel("Classe prevista")
        ax.set_ylabel("Classe real")
        plt.tight_layout()
        plt.savefig(pasta_matrizes / f"Matriz_Confusao_{nome}{sufixo}.png", dpi=150)
        plt.close()


# ---------- Usar o modelo numa nova fotografia ----------
def prever_imagem(modelo_ou_caminho, caminho_imagem, classes):
    """Devolve a classe prevista e a confianca para uma fotografia."""
    if isinstance(modelo_ou_caminho, (str, Path)):
        modelo = keras.models.load_model(modelo_ou_caminho)
    else:
        modelo = modelo_ou_caminho

    img, _ = carregar(caminho_imagem, 0)
    probs = modelo.predict(tf.expand_dims(img, 0), verbose=0)[0]
    i = int(np.argmax(probs))
    return classes[i], float(probs[i])


# ---------- Execucao Principal ----------
def main():
    # Cria todas as pastas de saida se nao existirem
    for pasta in [PASTA_SAIDA, PASTA_CURVAS, PASTA_MATRIZES, PASTA_RELATORIOS, PASTA_MODELOS]:
        pasta.mkdir(parents=True, exist_ok=True)

    keras.utils.set_random_seed(SEED)

    classes, caminhos, rotulos = listar_imagens(PASTA_DADOS)
    n_classes = len(classes)
    print(f"{len(caminhos)} imagens encontradas, {n_classes} classes: {classes}")

    contagem = pd.Series(rotulos).map(dict(enumerate(classes))).value_counts()
    print("\nImagens por classe:")
    print(contagem)
    contagem.to_csv(PASTA_RELATORIOS / "Distribuicao_Classes.csv")

    (x_tr, y_tr), (x_val, y_val), (x_te, y_te) = preparar_splits(caminhos, rotulos)
    print(f"\nTreino={len(x_tr)}  Validacao={len(x_val)}  Teste={len(x_te)}")

    ds_treino = criar_dataset(x_tr, y_tr, treino=True)
    ds_val = criar_dataset(x_val, y_val, treino=False)
    ds_teste = criar_dataset(x_te, y_te, treino=False)

    pesos = compute_class_weight(
        class_weight="balanced", classes=np.arange(n_classes), y=y_tr
    )
    pesos_classe = {i: float(pesos[i] ** POTENCIA_PESOS) for i in range(n_classes)}
    print("\nPesos por classe:")
    for i, nome_classe in enumerate(classes):
        print(f"  {nome_classe}: {pesos_classe[i]:.3f}")
    pd.Series({classes[i]: pesos_classe[i] for i in range(n_classes)}).to_csv(
        PASTA_RELATORIOS / "Pesos_Classe.csv"
    )

    modelos_disponiveis = obter_modelos(n_classes)
    resultados = []

    for nome, (construir, com_pesos) in modelos_disponiveis.items():
        if nome not in A_CORRER:
            continue

        print(f"\n===== {nome} (pesos de classe: {com_pesos}) =====")
        modelo = construir()
        modelo.summary()

        hist = modelo.fit(
            ds_treino,
            validation_data=ds_val,
            epochs=EPOCAS,
            class_weight=pesos_classe if com_pesos else None,
            shuffle=False,  # O shuffle ja e gerido pelo tf.data.Dataset
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=PACIENCIA, restore_best_weights=True
                )
            ],
            verbose=2,
        )

        metricas, recall_por_classe, cm, previsto = avaliar(modelo, ds_teste, y_te, classes)
        print("\nMetricas no teste:", metricas)
        relatorio = classification_report(
            y_te, previsto, target_names=classes, digits=3, zero_division=0
        )
        print("\nRelatorio de Classificacao:\n", relatorio)
        (PASTA_RELATORIOS / f"Relatorio_{nome}.txt").write_text(relatorio, encoding="utf-8")

        guardar_curvas(hist, nome, PASTA_CURVAS)
        guardar_matrizes(nome, y_te, previsto, classes, PASTA_MATRIZES)
        modelo.save(PASTA_MODELOS / f"{nome}.keras")

        resultados.append(
            {
                "modelo": nome,
                "class_weight": com_pesos,
                "parametros": modelo.count_params(),
                "epocas": len(hist.history["loss"]),
                **metricas,
                **recall_por_classe,
            }
        )

        # Atualiza a tabela a cada modelo para nao perder dados se for interrompido
        pd.DataFrame(resultados).round(4).to_csv(
            PASTA_RELATORIOS / "Resultados_Ronda2.csv", index=False
        )

    tabela = pd.DataFrame(resultados).round(4)
    print("\nResumo Final:")
    print(tabela.to_string(index=False))


if __name__ == "__main__":
    main()