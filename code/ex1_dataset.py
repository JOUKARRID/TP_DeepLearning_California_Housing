# ex1_dataset.py - VERSION COMPLÈTE ET CORRIGÉE
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import pickle
import os

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Créer le dossier pour les images
if not os.path.exists('rapport_images'):
    os.makedirs('rapport_images')

print("=" * 80)
print("🏠 EXERCICE 1 - Exploration du dataset California Housing")
print("=" * 80)

# ============== Q1: CHARGEMENT ==============
print("\n📌 Q1: Chargement...")
housing = fetch_california_housing()

df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target

print(f"✅ Dataset chargé: {df.shape[0]} quartiers, {df.shape[1]-1} features")
print(f"   Features: {', '.join(housing.feature_names)}")
print(f"\n📋 5 premiers quartiers:")
print(df.head())

# ============== Q2: DISTRIBUTION ==============
print("\n📌 Q2: Distribution des prix...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['MedHouseVal'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].set_xlabel('Prix médian (centaines de k$ = 100 000$)', fontsize=11)
axes[0].set_ylabel('Nombre de quartiers', fontsize=11)
axes[0].set_title('Distribution des prix des maisons', fontsize=12, fontweight='bold')
axes[0].axvline(df['MedHouseVal'].mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Moyenne: {df["MedHouseVal"].mean():.2f}')
axes[0].axvline(df['MedHouseVal'].median(), color='green', linestyle='--', linewidth=2, 
                label=f'Médiane: {df["MedHouseVal"].median():.2f}')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].boxplot(df['MedHouseVal'], vert=True)
axes[1].set_ylabel('Prix médian (centaines de k$)', fontsize=11)
axes[1].set_title('Boxplot des prix (détection des outliers)', fontsize=12, fontweight='bold')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('rapport_images/target_distribution.png', dpi=120)
print("✅ Graphique 1 sauvegardé: rapport_images/target_distribution.png")

# ============== Q3: CORRÉLATIONS ==============
print("\n📌 Q3: Matrice de corrélation...")

plt.figure(figsize=(10, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', 
            square=True, linewidths=0.5)
plt.title('Matrice des corrélations', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('rapport_images/correlation_matrix.png', dpi=120)
print("✅ Graphique 2 sauvegardé: rapport_images/correlation_matrix.png")

# Analyse des corrélations
target_corr = corr_matrix['MedHouseVal'].drop('MedHouseVal').abs()
best_feature = target_corr.idxmax()
print(f"\n🔗 Feature la plus corrélée au prix: '{best_feature}' (corrélation: {target_corr.max():.3f})")

# ============== Q4: SPLIT ==============
print("\n📌 Q4: Division des données...")

# Stratification
df['target_bins'] = pd.cut(df['MedHouseVal'], bins=10, labels=False)

X = df[housing.feature_names].values
y = df['MedHouseVal'].values

X_train, X_temp, y_train, y_temp, bins_train, bins_temp = train_test_split(
    X, y, df['target_bins'].values, test_size=0.3, random_state=SEED, stratify=df['target_bins']
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=bins_temp
)

print(f"✅ Train: {X_train.shape[0]} ({X_train.shape[0]/len(df)*100:.0f}%)")
print(f"✅ Validation: {X_val.shape[0]} ({X_val.shape[0]/len(df)*100:.0f}%)")
print(f"✅ Test: {X_test.shape[0]} ({X_test.shape[0]/len(df)*100:.0f}%)")

# ============== Q5: NORMALISATION ==============
print("\n📌 Q5: Normalisation...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Normalisation effectuée")
print(f"   Moyennes (train): {X_train_scaled.mean(axis=0).round(3)}")
print(f"   Écarts-types (train): {X_train_scaled.std(axis=0).round(3)}")

# ============== Q6: DATALOADERS ==============
print("\n📌 Q6: Création des DataLoaders...")

# Conversion en tenseurs PyTorch
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

# Datasets
train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# Vérification
sample_x, sample_y = next(iter(train_loader))
print(f"✅ Batch d'entraînement: {sample_x.shape} (features)")
print(f"✅ Batch de targets: {sample_y.shape} (prix)")

# ============== SAUVEGARDE ==============
print("\n📌 Sauvegarde des DataLoaders pour l'Exercice 2...")

data_dict = {
    'train_loader': train_loader,
    'val_loader': val_loader,
    'test_loader': test_loader,
    'scaler': scaler
}

with open('data_loaders.pkl', 'wb') as f:
    pickle.dump(data_dict, f)

# Vérification
if os.path.exists('data_loaders.pkl'):
    size = os.path.getsize('data_loaders.pkl')
    print(f"✅ Fichier 'data_loaders.pkl' créé (taille: {size} octets)")
else:
    print("❌ ERREUR: Le fichier n'a pas été créé!")

print("\n" + "=" * 80)
print("🎉 EXERCICE 1 TERMINÉ AVEC SUCCÈS!")
print("=" * 80)
print("\n📁 Fichiers générés dans le dossier:")
print("   📊 rapport_images/target_distribution.png")
print("   📊 rapport_images/correlation_matrix.png")
print("   💾 data_loaders.pkl ← PRÊT POUR L'EXERCICE 2")

# Afficher les graphiques
plt.show()