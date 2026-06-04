import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    classification_report
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# =========================
# 1. RUTAS V2
# =========================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

TSV_PATH = os.path.join(BASE_DIR, "outputs_v2", "tsv", "training_data_v2.tsv")

METRICS_DIR = os.path.join(BASE_DIR, "outputs_v2", "metrics")
MODELS_DIR = os.path.join(BASE_DIR, "outputs_v2", "models")

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# =========================
# 2. CARGAR DATOS
# =========================

print("Cargando TSV v2...")
df = pd.read_csv(TSV_PATH, sep="\t")

print("Columnas del dataset:")
print(df.columns.tolist())

print("\nDistribución original:")
print(df["class_name"].value_counts())


# =========================
# 3. BALANCEAR DATASET
# =========================

min_count = df["class_name"].value_counts().min()

print(f"\nBalanceando dataset con {min_count} muestras por clase...")

balanced_parts = []

for class_name, group in df.groupby("class_name"):
    sampled_group = group.sample(n=min_count, random_state=42)
    balanced_parts.append(sampled_group)

df_balanced = pd.concat(balanced_parts, ignore_index=True)

print("\nDistribución balanceada:")
print(df_balanced["class_name"].value_counts())


# =========================
# 4. DEFINIR X, y
# =========================

feature_columns = [
    "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B11", "B12"
]

X = df_balanced[feature_columns].astype(float)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df_balanced["class_name"])

class_labels = list(label_encoder.classes_)
numeric_labels = list(range(len(class_labels)))

print("\nClases:")
for i, class_name in enumerate(class_labels):
    print(f"{i} -> {class_name}")


# =========================
# 5. TRAIN / TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nTamaño entrenamiento:", X_train.shape)
print("Tamaño prueba:", X_test.shape)


# =========================
# 6. MODELOS E HIPERPARÁMETROS
# =========================

models = {
    "Decision_Tree": {
        "pipeline": Pipeline([
            ("model", DecisionTreeClassifier(random_state=42))
        ]),
        "params": {
            "model__criterion": ["gini", "entropy"],
            "model__max_depth": [5, 10, 20, None],
            "model__min_samples_split": [2, 5, 10]
        }
    },

    "SVM": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(random_state=42))
        ]),
        "params": {
            "model__C": [0.1, 1, 10],
            "model__kernel": ["rbf", "linear"],
            "model__gamma": ["scale", "auto"]
        }
    },

    "ANN": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(
                random_state=42,
                max_iter=500,
                early_stopping=True
            ))
        ]),
        "params": {
            "model__hidden_layer_sizes": [(50,), (100,), (50, 50)],
            "model__activation": ["relu", "tanh"],
            "model__learning_rate_init": [0.001, 0.01]
        }
    },

    "KNN": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier())
        ]),
        "params": {
            "model__n_neighbors": [3, 5, 7, 9],
            "model__weights": ["uniform", "distance"],
            "model__metric": ["euclidean", "manhattan"]
        }
    },

    "Naive_Bayes": {
        "pipeline": Pipeline([
            ("model", GaussianNB())
        ]),
        "params": {
            "model__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]
        }
    }
}


# =========================
# 7. ENTRENAMIENTO Y EVALUACIÓN
# =========================

summary_results = []
best_models = {}

for model_name, config in models.items():
    print("\n" + "=" * 60)
    print(f"Entrenando modelo: {model_name}")
    print("=" * 60)

    grid = GridSearchCV(
        estimator=config["pipeline"],
        param_grid=config["params"],
        scoring="f1_macro",
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    best_models[model_name] = best_model

    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"\nMejores parámetros para {model_name}:")
    print(grid.best_params_)

    print(f"\nMétricas {model_name}:")
    print("Accuracy:", accuracy)
    print("Kappa:", kappa)
    print("Precision macro:", precision_macro)
    print("Recall macro:", recall_macro)
    print("F1 macro:", f1_macro)

    print("\nClassification report:")
    report = classification_report(
        y_test,
        y_pred,
        labels=numeric_labels,
        target_names=class_labels,
        zero_division=0
    )
    print(report)

    cm = confusion_matrix(y_test, y_pred, labels=numeric_labels)

    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual_{c}" for c in class_labels],
        columns=[f"Predicted_{c}" for c in class_labels]
    )

    cm_path = os.path.join(METRICS_DIR, f"confusion_matrix_{model_name}.csv")
    cm_df.to_csv(cm_path, index=True)

    report_path = os.path.join(METRICS_DIR, f"classification_report_{model_name}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    joblib.dump(best_model, model_path)

    summary_results.append({
        "model": model_name,
        "accuracy": accuracy,
        "kappa": kappa,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "best_params": grid.best_params_,
        "model_path": model_path
    })


# =========================
# 8. GUARDAR TABLA COMPARATIVA
# =========================

summary_df = pd.DataFrame(summary_results)

summary_df = summary_df.sort_values(
    by=["f1_macro", "kappa"],
    ascending=False
)

summary_path = os.path.join(METRICS_DIR, "model_comparison.csv")
summary_df.to_csv(summary_path, index=False)

print("\n" + "=" * 60)
print("TABLA COMPARATIVA FINAL V2")
print("=" * 60)
print(summary_df)

best_model_name = summary_df.iloc[0]["model"]
best_model_path = summary_df.iloc[0]["model_path"]

print("\nMejor modelo:")
print(best_model_name)
print("Ruta:")
print(best_model_path)

best_model_output_path = os.path.join(MODELS_DIR, "best_model.pkl")
joblib.dump(best_models[best_model_name], best_model_output_path)

label_encoder_path = os.path.join(MODELS_DIR, "label_encoder.pkl")
joblib.dump(label_encoder, label_encoder_path)

print("\nModelo ganador guardado en:")
print(best_model_output_path)

print("\nLabel encoder guardado en:")
print(label_encoder_path)

print("\nProceso terminado correctamente.")