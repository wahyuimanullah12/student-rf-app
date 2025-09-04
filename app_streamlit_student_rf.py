"""App Streamlit Student RF (robust)

This file contains helper functions to create a demo student dataset and train a Random Forest
classifier/regressor, plus two execution modes:

1. Streamlit UI mode: when `streamlit` is installed, the app will start a Streamlit web UI.
2. CLI fallback mode: when `streamlit` is NOT installed (sandboxed environment), a CLI runner
   will run a demo training, print metrics, save a model file, and run lightweight tests.

Why this change: the original file imported `streamlit` at module import time which caused
`ModuleNotFoundError` in environments that don't have `streamlit`. To make the script usable in
both environments we *lazy-import* streamlit inside the UI function and provide a CLI fallback
so the script can still be executed and debugged in sandboxed environments.

How to use:
- If you have streamlit installed locally: `streamlit run app_streamlit_student_rf.py`
- Otherwise run CLI demo: `python app_streamlit_student_rf.py --cli`
- Run tests: `python app_streamlit_student_rf.py --test`

"""

import io
import os
import pickle
import argparse
from typing import Tuple, Union

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)

# -----------------------------
# Helper functions (same API as original)
# -----------------------------

def make_demo_dataset(n: int = 300, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    # Fitur: nilai_akademik (0-100), absen_persen (0-100), skor_psikologi (0-100), eskul_jam (0-15)
    nilai = np.clip(rng.normal(75, 10, n), 0, 100)
    absen = np.clip(rng.normal(90, 8, n), 0, 100)
    psik = np.clip(rng.normal(70, 12, n), 0, 100)
    eskul = np.clip(rng.normal(4, 2, n), 0, 15)

    # Label: status ("Risiko", "Aman") berdasarkan skor gabungan + noise
    score = 0.5 * nilai + 0.3 * absen + 0.15 * psik + 2.0 * eskul + rng.normal(0, 10, n)
    label = np.where(score > 85, "Aman", "Risiko")

    return pd.DataFrame(
        {
            "nilai_akademik": np.round(nilai, 1),
            "absen_persen": np.round(absen, 1),
            "skor_psikologi": np.round(psik, 1),
            "eskul_jam": np.round(eskul, 1),
            "status": label,
        }
    )


def train_model(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    test_size: float = 0.2,
    n_estimators: int = 200,
    max_depth: Union[int, None] = None,
    random_state: int = 42,
):
    X = df[feature_cols]
    y = df[target_col]

    # Apakah target numerik (regresi) atau kategorikal (klasifikasi)?
    is_regression = pd.api.types.is_numeric_dtype(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=None if is_regression else y
    )

    if is_regression:
        model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )
    else:
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )

    model.fit(X_train, y_train)

    results = {}
    if is_regression:
        pred = model.predict(X_test)
        results["r2"] = r2_score(y_test, pred)
        results["mae"] = mean_absolute_error(y_test, pred)
        results["rmse"] = float(np.sqrt(mean_squared_error(y_test, pred)))
    else:
        pred = model.predict(X_test)
        results["accuracy"] = accuracy_score(y_test, pred)
        results["report"] = classification_report(y_test, pred, zero_division=0)
        results["cm"] = confusion_matrix(y_test, pred)
        results["y_test"] = y_test
        results["pred"] = pred

    return model, (X_train, X_test, y_train, y_test), results, is_regression


def pack_model_bytes(model, feature_cols, target_col, is_regression):
    payload = {
        "model": model,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "is_regression": is_regression,
    }
    bio = io.BytesIO()
    pickle.dump(payload, bio)
    bio.seek(0)
    return bio


def try_load_model(file) -> Tuple[object, list, str, bool]:
    payload = pickle.load(file)
    return (
        payload["model"],
        payload["feature_cols"],
        payload["target_col"],
        payload.get("is_regression", False),
    )


# -----------------------------
# Streamlit UI (lazy import streamlit inside function)
# -----------------------------

def streamlit_app():
    """Start the Streamlit web UI. This function imports streamlit lazily so the module
    can still be executed in environments without streamlit installed.
    """
    import streamlit as st
    import matplotlib.pyplot as plt

    st.set_page_config(page_title="Prediksi Siswa - Random Forest", layout="wide")
    st.title("📊 Prediksi Kinerja Siswa (Random Forest)")
    st.caption(
        "Web app ini membantu Anda melatih model Random Forest untuk memprediksi status siswa berdasarkan nilai akademik, absensi, skor psikologi, dan aktivitas ekstrakurikuler."
    )

    with st.sidebar:
        st.header("⚙️ Pengaturan Model")
        seed = st.number_input("Random State", min_value=0, value=42, step=1)
        n_estimators = st.slider("n_estimators", 50, 500, 200, 25)
        use_max_depth = st.checkbox("Batasi max_depth?", value=False)
        max_depth = st.slider("max_depth", 2, 30, 10) if use_max_depth else None
        test_size = st.slider("Test Size (proporsi data uji)", 0.1, 0.5, 0.2, 0.05)

    st.subheader("1) Data Sumber")

    demo = st.checkbox("Gunakan data contoh (tanpa upload)", value=True)

    if demo:
        df = make_demo_dataset()
        st.info(
            "Sedang menggunakan **data contoh**. Untuk data nyata, matikan opsi ini dan upload file CSV Anda."
        )
    else:
        up = st.file_uploader(
            "Upload CSV (header baris pertama)", type=["csv"], accept_multiple_files=False
        )
        if up is not None:
            try:
                df = pd.read_csv(up)
            except Exception:
                up.seek(0)
                df = pd.read_csv(up, sep=";")
        else:
            df = pd.DataFrame()

    if df.empty:
        st.warning("Belum ada data. Gunakan data contoh atau upload CSV.")
        st.stop()

    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("2) Pilih Fitur & Target")

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    all_cols = list(df.columns)

    # Rekomendasi default fitur umum
    default_features = [
        c
        for c in ["nilai_akademik", "absen_persen", "skor_psikologi", "eskul_jam"]
        if c in df.columns
    ]

    feature_cols = st.multiselect(
        "Fitur (X)", all_cols, default=default_features if default_features else num_cols
    )

    if not feature_cols:
        st.error("Pilih minimal satu kolom fitur.")
        st.stop()

    # Target: default ke kolom terakhir yang bukan fitur, atau 'status' bila ada
    default_target = "status" if "status" in df.columns else (
        [c for c in all_cols if c not in feature_cols][-1] if len(all_cols) > len(feature_cols) else all_cols[-1]
    )

    target_col = st.selectbox("Target (y)", [c for c in all_cols if c not in feature_cols] or all_cols, index=(
        ([c for c in all_cols if c not in feature_cols] or all_cols).index(default_target)
        if default_target in ([c for c in all_cols if c not in feature_cols] or all_cols)
        else 0
    ))

    st.divider()
    train_btn = st.button("🚀 Latih Model Random Forest", use_container_width=True)

    if train_btn:
        with st.spinner("Melatih model..."):
            model, splits, results, is_reg = train_model(
                df.copy(), feature_cols, target_col, test_size, n_estimators, max_depth, seed
            )

        X_train, X_test, y_train, y_test = splits

        st.success("Model selesai dilatih.")

        colL, colR = st.columns([1, 1])

        if is_reg:
            with colL:
                st.metric("R² (Test)", f"{results['r2']:.3f}")
                st.metric("MAE (Test)", f"{results['mae']:.3f}")
                st.metric("RMSE (Test)", f"{results['rmse']:.3f}")
            with colR:
                st.write("**Pentingnya Fitur**")
                importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
                st.bar_chart(importances)
        else:
            with colL:
                st.metric("Akurasi (Test)", f"{results['accuracy']:.3f}")
                st.text("\nLaporan Klasifikasi:\n" + results["report"])  # klasifikasi ringkas
            with colR:
                st.write("**Confusion Matrix**")
                try:
                    fig, ax = plt.subplots()
                    disp = ConfusionMatrixDisplay(results["cm"])
                    disp.plot(ax=ax, colorbar=False)
                    st.pyplot(fig, clear_figure=True)
                except Exception:
                    st.write(pd.DataFrame(results["cm"]))

        st.divider()
        st.subheader("3) Unduh / Muat Model")
        bio = pack_model_bytes(model, feature_cols, target_col, is_reg)
        st.download_button(
            label="💾 Unduh Model (.pkl)",
            data=bio,
            file_name="model_random_forest.pkl",
            mime="application/octet-stream",
        )

        load_col1, load_col2 = st.columns([1, 2])
        with load_col1:
            st.caption("Muat model yang sudah disimpan:")
            load_file = st.file_uploader("Upload .pkl model", type=["pkl"], key="load_model")
        with load_col2:
            if load_file is not None:
                mdl, fcols, tcol, is_reg_loaded = try_load_model(load_file)
                st.success(f"Model termuat. Fitur: {fcols} | Target: {tcol} | Regresi: {is_reg_loaded}")

        st.divider()
        st.subheader("4) Prediksi Siswa (Form Input)")

        with st.form("predict_form"):
            in_cols = feature_cols
            inputs = {}
            grid = st.columns(min(4, len(in_cols)) or 1)
            for idx, col in enumerate(in_cols):
                with grid[idx % len(grid)]:
                    # Ambil rentang wajar dari data untuk bantu pengguna
                    col_min = float(df[col].min())
                    col_max = float(df[col].max())
                    default_val = float(df[col].median())
                    step = max((col_max - col_min) / 100.0, 0.1)
                    inputs[col] = st.number_input(
                        f"{col}", value=default_val, step=step, format="%.3f"
                    )
            submitted = st.form_submit_button("Prediksi Sekarang")

        if submitted:
            sample = pd.DataFrame([inputs])[feature_cols]
            mdl = model  # gunakan model yang baru dilatih
            if is_reg:
                yhat = mdl.predict(sample)[0]
                st.info(f"Prediksi nilai target: **{yhat:.3f}**")
            else:
                proba = getattr(mdl, "predict_proba", None)
                pred = mdl.predict(sample)[0]
                st.success(f"Prediksi kelas: **{pred}**")
                if proba is not None:
                    prob_df = pd.DataFrame(
                        proba(sample), columns=[f"P({c})" for c in mdl.classes_]
                    )
                    st.write("Probabilitas:")
                    st.dataframe(prob_df, use_container_width=True)

        st.divider()
        with st.expander("ℹ️ Petunjuk Format CSV"):
            st.markdown(
                """
                **Contoh header & baris data (klasifikasi)**:

                ```csv
                nilai_akademik,absen_persen,skor_psikologi,eskul_jam,status
                80,92,74,3,Aman
                65,85,60,2,Risiko
                ```

                **Contoh untuk regresi** (target numerik, mis. skor akhir):

                ```csv
                nilai_akademik,absen_persen,skor_psikologi,eskul_jam,skor_akhir
                80,92,74,3,86
                65,85,60,2,72
                ```

                Catatan:
                - Pastikan kolom fitur bertipe numerik.
                - Target bertipe teks ➜ model klasifikasi. Target numerik ➜ model regresi.
                - Anda bisa memilih kolom mana saja sebagai fitur/target di langkah (2).
                ```
                """
            )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Tips: Mulai dengan data contoh untuk mencoba alur, lalu upload data sekolah Anda untuk melatih model nyata."
    )


# -----------------------------
# CLI fallback / lightweight demo and tests
# -----------------------------


def run_cli_demo(save_model_path: str = "model_random_forest_cli.pkl"):
    print("Streamlit not detected — running CLI demo mode")
    df = make_demo_dataset(n=500)

    # default features & target
    feature_cols = [c for c in ["nilai_akademik", "absen_persen", "skor_psikologi", "eskul_jam"] if c in df.columns]
    target_col = "status" if "status" in df.columns else df.columns[-1]

    model, splits, results, is_reg = train_model(df, feature_cols, target_col, test_size=0.2, n_estimators=150, max_depth=10, random_state=42)

    print("--- Results ---")
    if is_reg:
        print(f"R2: {results['r2']:.3f}")
        print(f"MAE: {results['mae']:.3f}")
        print(f"RMSE: {results['rmse']:.3f}")
    else:
        print(f"Accuracy: {results['accuracy']:.3f}")
        print("Classification report:\n", results["report"])

    # save model
    bio = pack_model_bytes(model, feature_cols, target_col, is_reg)
    with open(save_model_path, "wb") as f:
        f.write(bio.read())
    print(f"Saved model to: {os.path.abspath(save_model_path)}")
    print("You can later load it with 'try_load_model' or via the Streamlit UI when available.")


# -----------------------------
# Lightweight tests (no external test framework required)
# -----------------------------

def run_tests():
    print("Running lightweight tests...")
    # Test 1: training runs for classification
    df = make_demo_dataset(n=300, random_state=1)
    feature_cols = [c for c in ["nilai_akademik", "absen_persen", "skor_psikologi", "eskul_jam"] if c in df.columns]
    target_col = "status"

    model, splits, results, is_reg = train_model(df, feature_cols, target_col, test_size=0.2, n_estimators=50, max_depth=8, random_state=1)
    assert not is_reg, "Expected classification problem for 'status'"
    assert "accuracy" in results, "Results should contain accuracy for classification"
    acc = results["accuracy"]
    print(f"Test 1 - classification accuracy = {acc:.3f}")
    assert 0.0 <= acc <= 1.0, "Accuracy must be a valid probability"

    # Test 2: model packing & loading
    bio = pack_model_bytes(model, feature_cols, target_col, is_reg)
    bio.seek(0)
    loaded_model, fcols, tcol, loaded_is_reg = try_load_model(bio)
    assert isinstance(loaded_model, object)
    assert fcols == feature_cols
    assert tcol == target_col
    assert loaded_is_reg == is_reg
    print("Test 2 - pack & load OK")

    # Test 3: regression path
    # Create a regression-like dataset by making a numeric target
    df2 = make_demo_dataset(n=200, random_state=2)
    df2["skor_akhir"] = (0.5 * df2["nilai_akademik"] + 0.3 * df2["absen_persen"] + 0.2 * df2["skor_psikologi"]).round(1)
    feature_cols_reg = ["nilai_akademik", "absen_persen", "skor_psikologi", "eskul_jam"]
    model_r, splits_r, results_r, is_reg_r = train_model(df2, feature_cols_reg, "skor_akhir", test_size=0.2, n_estimators=50, max_depth=6, random_state=2)
    assert is_reg_r, "Expected regression for numeric target"
    assert "r2" in results_r
    # additional checks for regression metrics
    assert "mae" in results_r and "rmse" in results_r, "Regression results should include MAE and RMSE"
    assert results_r["rmse"] >= 0.0
    print(f"Test 3 - regression R2 = {results_r['r2']:.3f}")
    assert is_reg_r, "Expected regression for numeric target"
    assert "r2" in results_r
    print(f"Test 3 - regression R2 = {results_r['r2']:.3f}")

    print("All tests passed.")


# -----------------------------
# Entrypoint
# -----------------------------

if __name__ == "__main__":
    # We attempt to import streamlit only to decide which mode to run — but we never import it at
    # module import time. This avoids ModuleNotFoundError in sandboxed environments.
    try:
        import streamlit  # noqa: F401
        has_streamlit = True
    except Exception:
        has_streamlit = False

    parser = argparse.ArgumentParser(description="App Streamlit Student RF - fallback CLI available")
    parser.add_argument("--cli", action="store_true", help="Run CLI demo (no streamlit required)")
    parser.add_argument("--test", action="store_true", help="Run lightweight tests")
    parser.add_argument("--save", type=str, default="model_random_forest_cli.pkl", help="Path to save the model in CLI mode")
    args = parser.parse_args()

    if args.test:
        run_tests()
    elif args.cli or not has_streamlit:
        # CLI or no streamlit installed → run CLI demo
        run_cli_demo(save_model_path=args.save)
    else:
        # Streamlit is installed — run the UI
        # Note: when you run `streamlit run app_streamlit_student_rf.py` Streamlit will execute this
        # module as a script and we call streamlit_app() to start the UI.
        streamlit_app()
