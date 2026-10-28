import time
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
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers

from SwarmPackagePy import gwo, pso

# ---------- Configuracoes Globais ----------
PASTA_DADOS = "Road Issues"
PASTA_REF_META1 = Path("Resultados/Modelos/CNN_B_cw.keras")
PASTA_SAIDA = Path("Resultados_Meta2")

PASTA_CURVAS = PASTA_SAIDA / "Curvas"
PASTA_MATRIZES = PASTA_SAIDA / "Matrizes"
PASTA_RELATORIOS = PASTA_SAIDA / "Relatorios"
PASTA_MODELOS = PASTA_SAIDA / "Modelos"
PASTA_CONVERGENCIA = PASTA_SAIDA / "Convergencia"

SEED = 42
TAMANHO_IMG = (128, 128)
BATCH = 32

# Configuracao otimizada para execucao rapida no CPU (conforme recomendacao do enunciado)
N_PARTICULAS = 4          # Populacao do enxame
N_ITERACOES = 3           # Iteracoes por algoritmo 
EPOCAS_AVALIACAO = 2      # Epocas por avaliacao da funcao objetivo
FRACAO_DADOS_OTIM = 0.20  # Sub de 850 imagens para busca rapida

#Treino
EPOCAS_TREINO_FINAL = 25
PACIENCIA = 6

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


def criar_dataset(caminhos, rotulos, treino=False, fracao=1.0):
    if fracao < 1.0 and treino:
        # Extrai sub
        _, caminhos, _, rotulos = train_test_split(
            caminhos, rotulos, test_size=fracao, stratify=rotulos, random_state=SEED
        )

    ds = tf.data.Dataset.from_tensor_slices((caminhos, rotulos))
    if treino:
        ds = ds.shuffle(
            buffer_size=len(caminhos), seed=SEED, reshuffle_each_iteration=True
        )
    ds = ds.map(carregar, num_parallel_calls=tf.data.AUTOTUNE, deterministic=not treino)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)


def preparar_splits(caminhos, rotulos):
    """Divisao identica a Meta I: 70% treino, 15% validacao, 15% teste estratificado."""
    x_tr, x_resto, y_tr, y_resto = train_test_split(
        caminhos, rotulos, test_size=0.30, stratify=rotulos, random_state=SEED
    )
    x_val, x_te, y_val, y_te = train_test_split(
        x_resto, y_resto, test_size=0.50, stratify=y_resto, random_state=SEED
    )
    return (x_tr, y_tr), (x_val, y_val), (x_te, y_te)

def construir_cnn(n_classes, lr, dropout, neuronios_dense, nome="CNN_Otimizada"):
    m = keras.Sequential(name=nome)
    m.add(layers.Input(TAMANHO_IMG + (3,)))
    m.add(layers.RandomFlip("horizontal"))
    m.add(layers.RandomRotation(0.05))
    m.add(layers.RandomZoom(0.1))
    m.add(layers.Rescaling(1.0 / 255))

    for f in [32, 64, 128, 256]:
        m.add(layers.Conv2D(f, 3, padding="same", activation="relu"))
        m.add(layers.BatchNormalization())
        m.add(layers.MaxPooling2D())

    m.add(layers.GlobalAveragePooling2D())
    m.add(layers.Dense(int(neuronios_dense), activation="relu"))
    m.add(layers.Dropout(float(dropout)))
    m.add(layers.Dense(n_classes, activation="softmax"))

    m.compile(
        optimizer=keras.optimizers.Adam(learning_rate=float(lr)),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return m

LIMITES_INFERIORES = [-4.0, 0.10, 64.0]
LIMITES_SUPERIORES = [-2.3, 0.50, 256.0]


def decodificar_hiperparametros(x):
    lr = 10 ** float(x[0])
    dropout = float(x[1])
    neuronios = int(round(x[2]))
    return lr, dropout, neuronios

class FuncaoObjetivoEnxame:
    def __init__(self, n_classes, ds_tr_rapido, ds_val, pesos_classe, nome_algoritmo="Enxame"):
        self.n_classes = n_classes
        self.ds_tr = ds_tr_rapido
        self.ds_val = ds_val
        self.pesos_classe = pesos_classe
        self.nome_algoritmo = nome_algoritmo
        self.historico_avaliacoes = []
        self.n_avaliacoes = 0

    def __call__(self, x):
        lr, dropout, neuronios = decodificar_hiperparametros(x)
        self.n_avaliacoes += 1

        t0 = time.time()
        tf.keras.backend.clear_session()
        modelo = construir_cnn(self.n_classes, lr, dropout, neuronios)

        hist = modelo.fit(
            self.ds_tr,
            validation_data=self.ds_val,
            epochs=EPOCAS_AVALIACAO,
            class_weight=self.pesos_classe,
            verbose=0,
        )

        val_loss = float(min(hist.history["val_loss"]))
        tempo = time.time() - t0

        self.historico_avaliacoes.append(
            {
                "avaliacao": self.n_avaliacoes,
                "algoritmo": self.nome_algoritmo,
                "lr": lr,
                "dropout": dropout,
                "neuronios": neuronios,
                "val_loss": val_loss,
                "tempo_seg": round(tempo, 1),
            }
        )

        print(
            f"  [{self.nome_algoritmo} #{self.n_avaliacoes:02d}] lr={lr:.5f}, "
            f"dropout={dropout:.2f}, dense={neuronios:3d} -> val_loss={val_loss:.4f} ({tempo:.1f}s)"
        )
        return val_loss

def avaliar_modelo(modelo, ds_teste, y_true, classes):
    probs = modelo.predict(ds_teste, verbose=0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    previsto = probs.argmax(axis=1)
    cm = confusion_matrix(y_true, previsto)

    especificidades = []
    n_classes = len(classes)
    for i in range(n_classes):
        tp = cm[i, i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        especificidades.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)

    metricas = {
        "accuracy": float(accuracy_score(y_true, previsto)),
        "precision_macro": float(precision_score(y_true, previsto, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, previsto, average="macro", zero_division=0)),
        "especificidade_macro": float(np.mean(especificidades)),
        "f1_macro": float(f1_score(y_true, previsto, average="macro", zero_division=0)),
        "auc_macro_ovr": float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro")),
    }
    return metricas, cm, previsto


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


def guardar_grafico_convergencia(historico_pso, historico_gwo, pasta_convergencia):
    pasta_convergencia.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    melhor_pso = np.minimum.accumulate([h["val_loss"] for h in historico_pso])
    melhor_gwo = np.minimum.accumulate([h["val_loss"] for h in historico_gwo])

    ax.plot(range(1, len(melhor_pso) + 1), melhor_pso, label="PSO (Particle Swarm Optimization)", marker="o")
    ax.plot(range(1, len(melhor_gwo) + 1), melhor_gwo, label="GWO (Grey Wolf Optimizer)", marker="s")
    ax.set_title("Convergencia da Otimizacao por Enxame (Meta II)")
    ax.set_xlabel("Avaliacoes da Funcao Objetivo")
    ax.set_ylabel("Melhor val_loss acumulada")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(pasta_convergencia / "Convergencia_PSO_vs_GWO.png", dpi=150)
    plt.close()

def main():
    for pasta in [PASTA_SAIDA, PASTA_CURVAS, PASTA_MATRIZES, PASTA_RELATORIOS, PASTA_MODELOS, PASTA_CONVERGENCIA]:
        pasta.mkdir(parents=True, exist_ok=True)

    keras.utils.set_random_seed(SEED)
    classes, caminhos, rotulos = listar_imagens(PASTA_DADOS)
    n_classes = len(classes)
    print(f"Dataset: {len(caminhos)} imagens, {n_classes} classes: {classes}")

    (x_tr, y_tr), (x_val, y_val), (x_te, y_te) = preparar_splits(caminhos, rotulos)
    print(f"Divisao: Treino={len(x_tr)}, Validacao={len(x_val)}, Teste={len(x_te)}")

    pesos = compute_class_weight("balanced", classes=np.arange(n_classes), y=y_tr)
    pesos_classe = {i: float(pesos[i]) for i in range(n_classes)}

    # Datasets
    ds_treino_completo = criar_dataset(x_tr, y_tr, treino=True)
    ds_treino_rapido = criar_dataset(x_tr, y_tr, treino=True, fracao=FRACAO_DADOS_OTIM)
    ds_val = criar_dataset(x_val, y_val, treino=False)
    ds_teste = criar_dataset(x_te, y_te, treino=False)

    resultados_comparativos = []

    print("\n" + "=" * 60)
    print(" 1. CARREGAMENTO E AVALIACAO DO MODELO DE REFERENCIA (META I)")
    print("=" * 60)

    if PASTA_REF_META1.exists():
        print(f"A carregar pesos do modelo de referencia da Meta I: {PASTA_REF_META1}")
        modelo_base = keras.models.load_model(PASTA_REF_META1)
        tempo_base = 0.0
    else:
        print("Modelo gravado nao encontrado, a treinar referencia...")
        t0 = time.time()
        modelo_base = construir_cnn(n_classes, lr=5e-4, dropout=0.4, neuronios_dense=128, nome="CNN_Referencia")
        hist_base = modelo_base.fit(
            ds_treino_completo,
            validation_data=ds_val,
            epochs=15,
            class_weight=pesos_classe,
            callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
            verbose=1,
        )
        tempo_base = time.time() - t0
        guardar_curvas(hist_base, "CNN_Referencia", PASTA_CURVAS)

    metricas_base, cm_base, pred_base = avaliar_modelo(modelo_base, ds_teste, y_te, classes)
    print("Metricas da Referencia no Teste:", metricas_base)
    guardar_matrizes("CNN_Referencia", y_te, pred_base, classes, PASTA_MATRIZES)

    resultados_comparativos.append(
        {
            "modelo": "CNN_Referencia (Meta I)",
            "algoritmo": "Nenhum (Baseline)",
            "lr": 5e-4,
            "dropout": 0.40,
            "neuronios_dense": 128,
            "tempo_treino_seg": round(tempo_base, 1),
            "tempo_otimizacao_seg": 0.0,
            **metricas_base,
        }
    )

    # Particle Swarm
    print("\n" + "=" * 60)
    print(f" 2. OTIMIZACAO COM PSO ({N_PARTICULAS} particulas, {N_ITERACOES} iteracoes)")
    print("=" * 60)
    func_pso = FuncaoObjetivoEnxame(n_classes, ds_treino_rapido, ds_val, pesos_classe, nome_algoritmo="PSO")

    t_otim_pso_inicio = time.time()
    otimizador_pso = pso(
        n=N_PARTICULAS,
        function=func_pso,
        lb=LIMITES_INFERIORES,
        ub=LIMITES_SUPERIORES,
        dimension=3,
        iteration=N_ITERACOES,
    )
    tempo_otim_pso = time.time() - t_otim_pso_inicio

    melhor_x_pso = otimizador_pso.get_Gbest()
    melhor_lr_pso, melhor_dp_pso, melhor_neu_pso = decodificar_hiperparametros(melhor_x_pso)
    print(f"\n[PSO Concluido] Melhor solucao: lr={melhor_lr_pso:.6f}, dropout={melhor_dp_pso:.2f}, dense={melhor_neu_pso} (demorou {tempo_otim_pso/60:.1f} min)")

    # Treino PSO
    print("\n[PSO] A treinar modelo final com hiper-parametros otimizados...")
    t_treino_pso = time.time()
    tf.keras.backend.clear_session()
    modelo_pso = construir_cnn(n_classes, melhor_lr_pso, melhor_dp_pso, melhor_neu_pso, nome="CNN_Otimizada_PSO")
    hist_pso = modelo_pso.fit(
        ds_treino_completo,
        validation_data=ds_val,
        epochs=EPOCAS_TREINO_FINAL,
        class_weight=pesos_classe,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=PACIENCIA, restore_best_weights=True)],
        verbose=1,
    )
    tempo_treino_pso_total = time.time() - t_treino_pso
    metricas_pso, cm_pso, pred_pso = avaliar_modelo(modelo_pso, ds_teste, y_te, classes)

    guardar_curvas(hist_pso, "CNN_Otimizada_PSO", PASTA_CURVAS)
    guardar_matrizes("CNN_Otimizada_PSO", y_te, pred_pso, classes, PASTA_MATRIZES)
    modelo_pso.save(PASTA_MODELOS / "CNN_Otimizada_PSO.keras")

    resultados_comparativos.append(
        {
            "modelo": "CNN_Otimizada_PSO",
            "algoritmo": "PSO",
            "lr": round(melhor_lr_pso, 6),
            "dropout": round(melhor_dp_pso, 3),
            "neuronios_dense": melhor_neu_pso,
            "tempo_treino_seg": round(tempo_treino_pso_total, 1),
            "tempo_otimizacao_seg": round(tempo_otim_pso, 1),
            **metricas_pso,
        }
    )

    # Grey Wolf
    print("\n" + "=" * 60)
    print(f" 3. OTIMIZACAO COM GWO ({N_PARTICULAS} lobos, {N_ITERACOES} iteracoes)")
    print("=" * 60)
    func_gwo = FuncaoObjetivoEnxame(n_classes, ds_treino_rapido, ds_val, pesos_classe, nome_algoritmo="GWO")

    t_otim_gwo_inicio = time.time()
    otimizador_gwo = gwo(
        n=N_PARTICULAS,
        function=func_gwo,
        lb=LIMITES_INFERIORES,
        ub=LIMITES_SUPERIORES,
        dimension=3,
        iteration=N_ITERACOES,
    )
    tempo_otim_gwo = time.time() - t_otim_gwo_inicio

    melhor_x_gwo = otimizador_gwo.get_Gbest()
    melhor_lr_gwo, melhor_dp_gwo, melhor_neu_gwo = decodificar_hiperparametros(melhor_x_gwo)
    print(f"\n[GWO Concluido] Melhor solucao: lr={melhor_lr_gwo:.6f}, dropout={melhor_dp_gwo:.2f}, dense={melhor_neu_gwo} (demorou {tempo_otim_gwo/60:.1f} min)")

    # Treino Grey
    print("\n[GWO] A treinar modelo final com hiper-parametros otimizados...")
    t_treino_gwo = time.time()
    tf.keras.backend.clear_session()
    modelo_gwo = construir_cnn(n_classes, melhor_lr_gwo, melhor_dp_gwo, melhor_neu_gwo, nome="CNN_Otimizada_GWO")
    hist_gwo = modelo_gwo.fit(
        ds_treino_completo,
        validation_data=ds_val,
        epochs=EPOCAS_TREINO_FINAL,
        class_weight=pesos_classe,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=PACIENCIA, restore_best_weights=True)],
        verbose=1,
    )
    tempo_treino_gwo_total = time.time() - t_treino_gwo
    metricas_gwo, cm_gwo, pred_gwo = avaliar_modelo(modelo_gwo, ds_teste, y_te, classes)

    guardar_curvas(hist_gwo, "CNN_Otimizada_GWO", PASTA_CURVAS)
    guardar_matrizes("CNN_Otimizada_GWO", y_te, pred_gwo, classes, PASTA_MATRIZES)
    modelo_gwo.save(PASTA_MODELOS / "CNN_Otimizada_GWO.keras")

    resultados_comparativos.append(
        {
            "modelo": "CNN_Otimizada_GWO",
            "algoritmo": "GWO",
            "lr": round(melhor_lr_gwo, 6),
            "dropout": round(melhor_dp_gwo, 3),
            "neuronios_dense": melhor_neu_gwo,
            "tempo_treino_seg": round(tempo_treino_gwo_total, 1),
            "tempo_otimizacao_seg": round(tempo_otim_gwo, 1),
            **metricas_gwo,
        }
    )

    # Results
    guardar_grafico_convergencia(func_pso.historico_avaliacoes, func_gwo.historico_avaliacoes, PASTA_CONVERGENCIA)

    df_res = pd.DataFrame(resultados_comparativos).round(4)
    df_res.to_csv(PASTA_RELATORIOS / "Resultados_Comparativos_Meta2.csv", index=False)

    pd.DataFrame(func_pso.historico_avaliacoes).round(4).to_csv(
        PASTA_RELATORIOS / "Historico_Avaliacoes_PSO.csv", index=False
    )
    pd.DataFrame(func_gwo.historico_avaliacoes).round(4).to_csv(
        PASTA_RELATORIOS / "Historico_Avaliacoes_GWO.csv", index=False
    )

    print("\n" + "=" * 60)
    print(" TABELA COMPARATIVA FINAL (META II)")
    print("=" * 60)
    print(df_res.to_string(index=False))

    # Relatorios 
    for nome_m, pred_m in [("Referencia", pred_base), ("PSO", pred_pso), ("GWO", pred_gwo)]:
        rep = classification_report(y_te, pred_m, target_names=classes, digits=3, zero_division=0)
        (PASTA_RELATORIOS / f"Relatorio_Classificacao_{nome_m}.txt").write_text(rep, encoding="utf-8")


if __name__ == "__main__":
    main()