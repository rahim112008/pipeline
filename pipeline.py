# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split

# Importation des classes depuis bio_analyse.py
from bio_analyse import DataValidator, BioDataModeler, Reporter

# Configuration de la page
st.set_page_config(
    page_title="Bio-Analyse Intégrée",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧬 Bio-Analyse Intégrée - Pipeline d'analyse de données biologiques")
st.markdown("""
Cette application vous permet de :
- Charger un fichier CSV contenant des données biologiques (features + cible)
- Valider, imputer, normaliser et encoder automatiquement les données
- Entraîner un modèle de classification (Random Forest)
- Visualiser les performances et l'importance des variables
- Générer un rapport pour un échantillon spécifique
""")

# Sidebar pour les paramètres
with st.sidebar:
    st.header("⚙️ Paramètres")
    uploaded_file = st.file_uploader("📂 Charger un fichier CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"Fichier chargé : {uploaded_file.name} ({df.shape[0]} lignes, {df.shape[1]} colonnes)")
    else:
        # Utiliser un jeu de données synthétique par défaut
        st.info("Aucun fichier chargé. Utilisation d'un jeu de données synthétique.")
        # Génération des données synthétiques
        np.random.seed(42)
        n_samples = 500
        data_num = np.random.randn(n_samples, 10)
        data_cat = np.random.choice(['A', 'B', 'C'], size=(n_samples, 5))
        target = (data_num[:, 0] + data_num[:, 1] - data_num[:, 2] + np.random.randn(n_samples) * 0.5 > 0).astype(int)
        columns_num = [f'gene_{i}' for i in range(10)]
        columns_cat = [f'variant_{i}' for i in range(5)]
        df_num = pd.DataFrame(data_num, columns=columns_num)
        df_cat = pd.DataFrame(data_cat, columns=columns_cat)
        df = pd.concat([df_num, df_cat], axis=1)
        df['target'] = target
        # Ajout de quelques NaN
        df.iloc[10:20, 0] = np.nan
        df.iloc[30:35, 5] = np.nan
        df.iloc[50:55, 10] = np.nan
        st.info("Jeu de données synthétique généré (500 échantillons, 15 features, cible binaire)")

    # Sélection de la colonne cible
    if df is not None:
        target_col = st.selectbox("🎯 Colonne cible", options=df.columns, index=len(df.columns)-1)
        # Séparation features / target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        st.write(f"Features : {X.shape[1]} colonnes, Target : {y.name}")

        # Paramètres du modèle
        st.subheader("Modèle")
        n_estimators = st.slider("Nombre d'arbres", 50, 300, 100, step=10)
        max_depth = st.slider("Profondeur maximale", 3, 20, 10)
        test_size = st.slider("Taille du test (%)", 10, 40, 30, step=5) / 100

        # Bouton pour lancer l'analyse
        run = st.button("🚀 Lancer l'analyse", type="primary")

# Zone principale
if run and (uploaded_file is not None or df is not None):
    with st.spinner("Traitement en cours..."):

        # ---- Prétraitement ----
        # Déterminer le schéma (optionnel) : toutes les colonnes numériques => float, catégorielles => category
        schema = {}
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                schema[col] = 'float'
            else:
                schema[col] = 'category'

        validator = DataValidator(schema=schema)
        X_validated = validator.run_validation_pipeline(X, fit=True)

        # ---- Division train/test ----
        X_train, X_test, y_train, y_test = train_test_split(
            X_validated, y, test_size=test_size, random_state=42, stratify=y
        )

        # ---- Entraînement du modèle ----
        modeler = BioDataModeler(model=RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        ))
        modeler.train(X_train, y_train)
        acc = modeler.evaluate(X_test, y_test)

        # ---- Importance des features ----
        importance_df = modeler.get_feature_importance()

        # ---- Sauvegarde (optionnelle) ----
        # On peut stocker dans session_state pour une utilisation ultérieure
        st.session_state['modeler'] = modeler
        st.session_state['validator'] = validator
        st.session_state['X_test'] = X_test
        st.session_state['y_test'] = y_test
        st.session_state['importance_df'] = importance_df

    # ---- Affichage des résultats ----
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Précision sur le test", f"{acc:.2%}")
        # Rapport de classification sous forme de DataFrame
        report_df = pd.DataFrame(modeler.class_report).transpose()
        st.dataframe(report_df.style.background_gradient(cmap='Blues', subset=['precision', 'recall', 'f1-score']))

    with col2:
        # Graphique de l'importance des features
        fig, ax = plt.subplots(figsize=(8, 6))
        top_n = min(15, len(importance_df))
        top_features = importance_df.head(top_n)
        ax.barh(top_features['feature'], top_features['importance'], color='steelblue')
        ax.set_xlabel("Importance")
        ax.set_title(f"Top {top_n} features les plus importantes")
        ax.invert_yaxis()
        st.pyplot(fig)

    # ---- Prédiction sur un échantillon ----
    st.subheader("🔍 Prédiction sur un échantillon du test")
    sample_idx = st.selectbox("Choisir un index d'échantillon (parmi le test)", options=range(len(X_test)))
    sample = X_test.iloc[sample_idx]
    true_label = y_test.iloc[sample_idx]

    reporter = Reporter(modeler, validator)
    report = reporter.generate_report(sample, true_label)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**Diagnostic** : {report['diagnostic']}")
    with col2:
        st.success(f"**Confiance** : {report['confiance']}")
    with col3:
        if 'correction' in report:
            if report['correction'] == "Correct":
                st.success("✅ Prédiction correcte")
            else:
                st.error("❌ Prédiction incorrecte")

    st.write("**Features influentes (top 5 global)** :")
    if report['features_influentes_globales']:
        for feat in report['features_influentes_globales']:
            st.write(f"- {feat['feature']} : {feat['importance']:.4f}")

    # ---- Téléchargement du modèle ----
    st.download_button(
        label="💾 Télécharger le modèle entraîné (joblib)",
        data=joblib.dumps(modeler.model),
        file_name="bio_model.joblib",
        mime="application/octet-stream"
    )

else:
    if not run:
        st.info("Chargez un fichier CSV ou utilisez les données synthétiques, puis cliquez sur 'Lancer l'analyse'.")

# Pied de page
st.markdown("---")
st.caption("Application développée avec Streamlit • Pipeline Bio-Analyse Intégrée v2.0")