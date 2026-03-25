import copy
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

st.set_page_config(page_title="Parameter Sweep (PyTorch)", layout="wide")
st.title("Neural Network Hyperparameter Tuning & Evaluation")

# ── Default (baseline) hyperparameters ────────────────────────────────────────
DEFAULTS = {
    "num_layers":    2,
    "num_neurons":   64,
    "learning_rate": 0.001,
    "dropout_rate":  0.2,
    "l2_rate":       0.0,       # NEW: L2 regularisation (weight_decay in Adam)
}

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Sweep Configuration")

    num_layers_input = st.multiselect(
        "num_layers values",
        options=[1, 2, 3, 4, 5],
        default=[1, 2, 3],
    )
    num_neurons_input = st.multiselect(
        "num_neurons values",
        options=[16, 32, 64, 128, 256],
        default=[32, 64, 128],
    )

    st.markdown("**learning_rate values** (comma-separated)")
    lr_text = st.text_input("learning_rate", value="0.01, 0.001, 0.0001")
    try:
        learning_rate_input = [float(x.strip()) for x in lr_text.split(",") if x.strip()]
    except ValueError:
        st.error("Invalid learning rate values.")
        learning_rate_input = [0.01, 0.001, 0.0001]

    dropout_input = st.multiselect(
        "dropout_rate values",
        options=[0.0, 0.1, 0.2, 0.3, 0.5],
        default=[0.0, 0.2, 0.5],
    )

    # NEW: L2 regularisation sweep
    st.markdown("**l2_rate values** (comma-separated)")
    l2_text = st.text_input("l2_rate", value="0.0, 0.0001, 0.001")
    try:
        l2_input = [float(x.strip()) for x in l2_text.split(",") if x.strip()]
    except ValueError:
        st.error("Invalid L2 rate values.")
        l2_input = [0.0, 0.0001, 0.001]

    st.header("Dataset & Training")
    n_samples    = st.slider("n_samples",   min_value=200,  max_value=5000, value=1000, step=100)
    epochs       = st.slider("max epochs",  min_value=5,    max_value=100,  value=30)
    patience     = st.slider("early-stop patience", min_value=1, max_value=20, value=5)   # NEW
    random_seed  = st.number_input("random_seed", min_value=0, max_value=9999, value=42)

    run_button = st.button("Run Experiments", type="primary")

sweep_params = {
    "num_layers":    num_layers_input,
    "num_neurons":   num_neurons_input,
    "learning_rate": learning_rate_input,
    "dropout_rate":  dropout_input,
    "l2_rate":       l2_input,          # NEW
}


# ══════════════════════════════════════════════════════════════════════════════
# Data helpers
# ══════════════════════════════════════════════════════════════════════════════

def build_dataset(n_samples: int, random_seed: int):
    """Generate a synthetic binary-classification dataset and split it.

    Split: 80 % train / 20 % test (then 80 % of train = train, 20 % = val).
    All splits are normalised with a StandardScaler fitted on the training set.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=random_seed,
    )
    # 80 / 20 primary split
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_seed
    )
    # carve out a validation split (20 % of training = 16 % of total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=random_seed
    )
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)
    return X_train, X_val, X_test, y_train, y_val, y_test


def to_loader(X, y, batch_size: int = 32, shuffle: bool = False) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ══════════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════════

class BinaryClassifier(nn.Module):
    """Fully-connected binary classifier with variable depth and dropout."""

    def __init__(self, num_layers: int, num_neurons: int, dropout_rate: float):
        super().__init__()
        layers = [nn.Linear(20, num_neurons), nn.ReLU(), nn.Dropout(dropout_rate)]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(num_neurons, num_neurons), nn.ReLU(), nn.Dropout(dropout_rate)]
        layers += [nn.Linear(num_neurons, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def create_model(
    learning_rate: float = 0.001,
    dropout_rate:  float = 0.2,
    l2_rate:       float = 0.0,
    num_layers:    int   = 2,
    num_neurons:   int   = 64,
) -> tuple[BinaryClassifier, torch.optim.Optimizer]:
    """Factory that mirrors the Keras create_model pattern.

    Parameters
    ----------
    learning_rate : float
        Adam learning rate.
    dropout_rate : float
        Fraction of units randomly zeroed during training.
    l2_rate : float
        L2 weight-decay coefficient applied via Adam's ``weight_decay``.
    num_layers : int
        Number of hidden layers (≥ 1).
    num_neurons : int
        Width of every hidden layer.

    Returns
    -------
    model : BinaryClassifier
    optimizer : torch.optim.Adam   (with BCE loss used during training)
    """
    model = BinaryClassifier(
        num_layers=num_layers,
        num_neurons=num_neurons,
        dropout_rate=dropout_rate,
    )
    # weight_decay implements L2 regularisation in Adam
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=l2_rate,
    )
    return model, optimizer


# ══════════════════════════════════════════════════════════════════════════════
# Training utilities
# ══════════════════════════════════════════════════════════════════════════════

def run_epoch(
    model:     BinaryClassifier,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    training:  bool,
) -> tuple[float, float]:
    """Run one epoch; returns (accuracy, avg_loss)."""
    model.train(training)
    correct, total, running_loss = 0, 0, 0.0
    with torch.set_grad_enabled(training):
        for X_batch, y_batch in loader:
            preds = model(X_batch).squeeze(1)
            loss  = criterion(preds, y_batch)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * len(y_batch)
            correct      += ((preds >= 0.5) == y_batch.bool()).sum().item()
            total        += len(y_batch)
    return correct / total, running_loss / total


def compute_full_metrics(
    model:  BinaryClassifier,
    loader: DataLoader,
) -> dict:
    """Return accuracy, precision, recall, and F1 on *loader*."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            preds = (model(X_batch).squeeze(1) >= 0.5).long()
            all_preds.append(preds.numpy())
            all_labels.append(y_batch.long().numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    return {
        "accuracy":  (y_pred == y_true).mean(),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
    }


def train_experiment(
    param_name:  str,
    param_value,
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    epochs:      int,
    patience:    int,
    random_seed: int,
) -> tuple[dict, dict]:
    """Train one experiment with early stopping; returns (history, test_metrics)."""
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Build kwargs by overriding the chosen parameter
    kwargs = dict(DEFAULTS)
    kwargs[param_name] = param_value

    model, optimizer = create_model(
        learning_rate=kwargs["learning_rate"],
        dropout_rate=kwargs["dropout_rate"],
        l2_rate=kwargs["l2_rate"],
        num_layers=kwargs["num_layers"],
        num_neurons=kwargs["num_neurons"],
    )
    criterion    = nn.BCELoss()
    train_loader = to_loader(X_train, y_train, shuffle=True)
    val_loader   = to_loader(X_val,   y_val)
    test_loader  = to_loader(X_test,  y_test)

    history = {
        "accuracy":     [],
        "val_accuracy": [],
        "loss":         [],
        "val_loss":     [],
    }

    # ── Early stopping state ──────────────────────────────────────────────────
    best_val_loss   = float("inf")
    epochs_no_improv = 0
    best_weights    = copy.deepcopy(model.state_dict())   # save best checkpoint

    for _ in range(epochs):
        train_acc,  train_loss = run_epoch(model, train_loader, criterion, optimizer, training=True)
        val_acc,    val_loss   = run_epoch(model, val_loader,   criterion, optimizer, training=False)

        history["accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)
        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Early stopping check (patience = 5 consecutive non-improving epochs)
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            epochs_no_improv = 0
            best_weights     = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improv += 1
            if epochs_no_improv >= patience:
                break   # restore best weights and stop

    # Restore the best checkpoint before evaluating on the test set
    model.load_state_dict(best_weights)
    test_metrics = compute_full_metrics(model, test_loader)
    return history, test_metrics


# ══════════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ══════════════════════════════════════════════════════════════════════════════

def make_curve_figure(history: dict, param_name: str, param_value) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    # Accuracy curve
    axes[0].plot(history["accuracy"],     label="Train")
    axes[0].plot(history["val_accuracy"], label="Val")
    axes[0].set_title(f"Accuracy  [{param_name} = {param_value}]")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Loss curve
    axes[1].plot(history["loss"],     label="Train")
    axes[1].plot(history["val_loss"], label="Val")
    axes[1].set_title(f"Loss  [{param_name} = {param_value}]")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("BCE Loss")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def make_summary_figure(param_name: str, values: list, test_accs: list) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(range(len(values)), test_accs, marker="o")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels([str(v) for v in values], rotation=15)
    ax.set_title(f"Test Accuracy vs {param_name}")
    ax.set_xlabel(param_name); ax.set_ylabel("Test Accuracy")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Run experiments
# ══════════════════════════════════════════════════════════════════════════════

if run_button:
    valid = all(len(v) > 0 for v in sweep_params.values())
    if not valid:
        st.error("Each parameter must have at least one value selected.")
    else:
        X_train, X_val, X_test, y_train, y_val, y_test = build_dataset(n_samples, random_seed)

        total_experiments = sum(len(v) for v in sweep_params.values())
        progress_bar = st.progress(0)
        status_text  = st.empty()

        results   = {}
        completed = 0

        for param_name, values in sweep_params.items():
            results[param_name] = []
            for val in values:
                status_text.text(f"Testing {param_name} = {val}  …")
                history, test_metrics = train_experiment(
                    param_name, val,
                    X_train, X_val, X_test,
                    y_train, y_val, y_test,
                    epochs, patience, random_seed,
                )
                results[param_name].append({
                    "value":        val,
                    "history":      history,
                    "test_metrics": test_metrics,
                    "epochs_run":   len(history["accuracy"]),
                })
                completed += 1
                progress_bar.progress(completed / total_experiments)

        status_text.text("✅ All experiments complete.")
        st.session_state["results"] = results


# ══════════════════════════════════════════════════════════════════════════════
# Display results
# ══════════════════════════════════════════════════════════════════════════════

if "results" in st.session_state:
    results = st.session_state["results"]

    # Tab labels: one per param + a Performance Report tab
    tab_labels = list(results.keys()) + ["📊 Performance Report"]
    tabs = st.tabs(tab_labels)

    # ── Per-parameter tabs ─────────────────────────────────────────────────────
    for tab, param_name in zip(tabs[:-1], results.keys()):
        with tab:
            st.subheader(f"Results for `{param_name}`")
            param_results = results[param_name]

            for entry in param_results:
                m = entry["test_metrics"]
                header = (
                    f"{param_name} = {entry['value']}  —  "
                    f"Acc {m['accuracy']:.3f} | "
                    f"P {m['precision']:.3f} | "
                    f"R {m['recall']:.3f} | "
                    f"F1 {m['f1']:.3f} | "
                    f"Epochs run: {entry['epochs_run']}"
                )
                with st.expander(header):
                    fig = make_curve_figure(entry["history"], param_name, entry["value"])
                    st.pyplot(fig)
                    plt.close(fig)

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Accuracy",  f"{m['accuracy']:.4f}")
                    col2.metric("Precision", f"{m['precision']:.4f}")
                    col3.metric("Recall",    f"{m['recall']:.4f}")
                    col4.metric("F1 Score",  f"{m['f1']:.4f}")

            values    = [e["value"]                     for e in param_results]
            test_accs = [e["test_metrics"]["accuracy"]  for e in param_results]
            summary_fig = make_summary_figure(param_name, values, test_accs)
            st.pyplot(summary_fig)
            plt.close(summary_fig)

    # ── Performance Report tab ─────────────────────────────────────────────────
    with tabs[-1]:
        st.subheader("📊 Performance Report — All Experiments")
        st.markdown(
            "Each row represents **one experiment**: a single hyperparameter varied "
            "while all others remain at their default values. "
            "The model uses **early stopping** (patience configured in sidebar) "
            "and **L2 regularisation** (via Adam `weight_decay`)."
        )

        rows = []
        for param_name, param_results in results.items():
            for entry in param_results:
                m = entry["test_metrics"]
                rows.append({
                    "Parameter":   param_name,
                    "Value":       entry["value"],
                    "Epochs Run":  entry["epochs_run"],
                    "Accuracy":    round(m["accuracy"],  4),
                    "Precision":   round(m["precision"], 4),
                    "Recall":      round(m["recall"],    4),
                    "F1 Score":    round(m["f1"],        4),
                    "Baseline?":   entry["value"] == DEFAULTS.get(param_name),
                })

        df = pd.DataFrame(rows)

        # Style the table
        def highlight_baseline(row):
            colour = "background-color: #d4edda" if row["Baseline?"] else ""
            return [colour] * len(row)

        styled = (
            df.style
            .apply(highlight_baseline, axis=1)
            .format({
                "Accuracy":  "{:.4f}",
                "Precision": "{:.4f}",
                "Recall":    "{:.4f}",
                "F1 Score":  "{:.4f}",
            })
            .hide(axis="index")
        )
        st.dataframe(styled, use_container_width=True)
        st.caption("🟢 Green rows = baseline (default) hyperparameter value")

        # ── Best config callout ────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🏆 Best Configuration by F1 Score")
        best_row = df.loc[df["F1 Score"].idxmax()]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("F1 Score",  f"{best_row['F1 Score']:.4f}")
        c2.metric("Accuracy",  f"{best_row['Accuracy']:.4f}")
        c3.metric("Precision", f"{best_row['Precision']:.4f}")
        c4.metric("Recall",    f"{best_row['Recall']:.4f}")
        st.info(
            f"**Best parameter:** `{best_row['Parameter']}` "
            f"= **{best_row['Value']}**  |  "
            f"Epochs until early stop: **{int(best_row['Epochs Run'])}**"
        )

        # ── Per-parameter summary bars ─────────────────────────────────────────
        st.markdown("---")
        st.subheader("Per-Parameter F1 Summary")
        fig_cols = st.columns(2)
        col_idx = 0
        for param_name, param_results in results.items():
            vals = [str(e["value"]) for e in param_results]
            f1s  = [e["test_metrics"]["f1"] for e in param_results]
            defaults_val = str(DEFAULTS.get(param_name))
            colours = ["#2196F3" if v != defaults_val else "#4CAF50" for v in vals]

            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar(vals, f1s, color=colours)
            ax.set_title(f"{param_name}")
            ax.set_xlabel("Value"); ax.set_ylabel("F1")
            ax.set_ylim(0, 1); ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            with fig_cols[col_idx % 2]:
                st.pyplot(fig)
            plt.close(fig)
            col_idx += 1
