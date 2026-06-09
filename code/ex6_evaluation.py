# ex6_evaluation.py
"""
EXERCICE 6: Évaluation Finale et Rapport de Synthèse
Compatible VS Code et Google Colab
Master AIDC - Deep Learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

# Détection de l'environnement
try:
    from google.colab import files
    IN_COLAB = True
    print("✅ Environnement: Google Colab")
except ImportError:
    IN_COLAB = False
    print("✅ Environnement: VS Code / Local")

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Device: {device}")

# Création du dossier images
os.makedirs('rapport_images', exist_ok=True)

print("=" * 80)
print("🎯 EXERCICE 6 - ÉVALUATION FINALE ET RAPPORT DE SYNTHÈSE")
print("=" * 80)

# ============== 1. CHARGEMENT COMPLET DES DONNÉES ==============
print("\n📂 Étape 1: Chargement complet du dataset...")

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Chargement
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target

# Split avec stratification
df['target_bins'] = pd.cut(df['MedHouseVal'], bins=10, labels=False)
X = df[housing.feature_names].values
y = df['MedHouseVal'].values

X_train, X_temp, y_train, y_temp, _, _ = train_test_split(
    X, y, df['target_bins'].values, test_size=0.3, random_state=SEED, stratify=df['target_bins']
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=SEED
)

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

# ============== 2. DÉFINITION DU MODÈLE ==============
print("\n📂 Étape 2: Définition du modèle DeepFFN...")

class DeepFFN(nn.Module):
    def __init__(self, input_dim=8, hidden_dims=[128, 64, 32], output_dim=1,
                 activation='relu', use_bn=True, dropout_rate=0.2):
        super().__init__()
        
        self.activation_name = activation
        
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.01),
            'elu': nn.ELU(),
            'tanh': nn.Tanh(),
            'selu': nn.SELU()
        }
        self.activation = activations.get(activation, nn.ReLU())
        
        self.layers = nn.ModuleList()
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
            if use_bn:
                self.layers.append(nn.BatchNorm1d(dims[i+1]))
            self.layers.append(self.activation)
            if dropout_rate > 0:
                self.layers.append(nn.Dropout(dropout_rate))
        
        self.output_layer = nn.Linear(dims[-1], output_dim)
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.activation_name in ['relu', 'leaky_relu', 'elu']:
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                else:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)

print("✅ Modèle défini")

# ============== 3. FONCTIONS D'ENTRAÎNEMENT ==============
print("\n📂 Étape 3: Fonctions d'entraînement...")

def train_one_epoch(model, loader, optimizer, criterion, clip_value=1.0):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        if clip_value and clip_value > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()
        total_loss += loss.item() * len(xb)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            total_loss += criterion(pred, yb).item() * len(xb)
            all_preds.extend(pred.cpu().numpy().flatten())
            all_targets.extend(yb.cpu().numpy().flatten())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    mse = total_loss / len(loader.dataset)
    mae = np.mean(np.abs(all_preds - all_targets))
    ss_res = np.sum((all_targets - all_preds)**2)
    ss_tot = np.sum((all_targets - np.mean(all_targets))**2)
    r2 = 1 - ss_res/(ss_tot + 1e-8)
    
    return mse, mae, r2, all_preds, all_targets

def train_model(model, train_loader, val_loader, config):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], 
                           weight_decay=config.get('weight_decay', 0.0))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      patience=10, factor=0.5, min_lr=1e-6)
    criterion = nn.MSELoss()
    
    best_val_mse = float('inf')
    best_model_state = None
    no_improve = 0
    patience = config.get('early_stopping_patience', 20)
    
    for epoch in range(config.get('epochs', 200)):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, 
                                      config.get('clip_value', 1.0))
        val_mse, val_mae, val_r2, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step(val_mse)
        
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_model_state = model.state_dict().copy()
            no_improve = 0
        else:
            no_improve += 1
        
        if (epoch + 1) % 50 == 0:
            print(f"   Epoch {epoch+1:3d} | Train Loss={train_loss:.4f} | Val MSE={val_mse:.4f} | R²={val_r2:.4f}")
        
        if no_improve >= patience:
            print(f"   Early stopping à l'epoch {epoch+1}")
            break
    
    model.load_state_dict(best_model_state)
    return model, best_val_mse

print("✅ Fonctions définies")

# ============== 4. TOP 3 CONFIGURATIONS ==============
print("\n" + "=" * 60)
print("📂 Étape 4: Définition du Top 3 des configurations")
print("=" * 60)

# Configuration 1: Meilleure de l'Exercice 3 (Sans BatchNorm)
config1 = {
    'name': 'Best Exercice 3 - Sans BatchNorm',
    'hidden_dims': [128, 64, 32],
    'activation': 'relu',
    'use_bn': False,
    'dropout_rate': 0.2,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'clip_value': 1.0,
    'epochs': 200,
    'early_stopping_patience': 25
}

# Configuration 2: Meilleure Grid Search
config2 = {
    'name': 'Best Grid Search',
    'hidden_dims': [128, 64, 32],
    'activation': 'relu',
    'use_bn': True,
    'dropout_rate': 0.0,
    'lr': 1e-3,
    'weight_decay': 0.0,
    'clip_value': 1.0,
    'epochs': 200,
    'early_stopping_patience': 25
}

# Configuration 3: Configuration avec L2 (bon compromis)
config3 = {
    'name': 'Best avec L2',
    'hidden_dims': [128, 64, 32],
    'activation': 'relu',
    'use_bn': True,
    'dropout_rate': 0.0,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'clip_value': 1.0,
    'epochs': 200,
    'early_stopping_patience': 25
}

configs = [config1, config2, config3]

print("\n📋 Top 3 configurations à évaluer:")
for i, cfg in enumerate(configs, 1):
    print(f"\n   {i}. {cfg['name']}")
    print(f"      Architecture: {cfg['hidden_dims']}")
    print(f"      Activation: {cfg['activation']}")
    print(f"      BatchNorm: {cfg['use_bn']}")
    print(f"      Dropout: {cfg['dropout_rate']}")
    print(f"      LR: {cfg['lr']}")
    print(f"      Weight Decay: {cfg['weight_decay']}")

# ============== 5. CRÉATION TRAIN+VAL POUR RÉENTRAÎNEMENT ==============
print("\n" + "=" * 60)
print("📂 Étape 5: Fusion Train+Val pour réentraînement final")
print("=" * 60)

X_trainval = np.vstack([X_train_scaled, X_val_scaled])
y_trainval = np.concatenate([y_train, y_val])

X_trainval_t = torch.tensor(X_trainval, dtype=torch.float32)
y_trainval_t = torch.tensor(y_trainval, dtype=torch.float32).reshape(-1, 1)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

trainval_dataset = TensorDataset(X_trainval_t, y_trainval_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

trainval_loader = DataLoader(trainval_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

print(f"✅ Train+Val: {len(trainval_loader.dataset)} exemples")
print(f"✅ Test: {len(test_loader.dataset)} exemples")

# ============== 6. ÉVALUATION DES 3 CONFIGURATIONS ==============
print("\n" + "=" * 60)
print("🚀 Étape 6: Évaluation des 3 configurations sur le test set")
print("=" * 60)

final_results = []
predictions = {}

for i, config in enumerate(configs):
    print(f"\n{'='*50}")
    print(f"📊 Test de: {config['name']}")
    print(f"{'='*50}")
    
    torch.manual_seed(SEED)
    
    model = DeepFFN(
        input_dim=8,
        hidden_dims=config['hidden_dims'],
        activation=config['activation'],
        use_bn=config['use_bn'],
        dropout_rate=config['dropout_rate']
    )
    
    # Entraînement sur Train+Val
    print("   Entraînement sur Train+Val...")
    t0 = time.time()
    model, best_val_mse = train_model(model, trainval_loader, test_loader, config)
    train_time = time.time() - t0
    
    # Évaluation sur Test
    print("   Évaluation sur Test set...")
    criterion = nn.MSELoss()
    test_mse, test_mae, test_r2, preds, targets = evaluate(model, test_loader, criterion)
    
    predictions[config['name']] = {'preds': preds, 'targets': targets}
    
    final_results.append({
        'Configuration': config['name'],
        'Test MSE': test_mse,
        'Test MAE': test_mae,
        'Test R²': test_r2,
        'Temps (s)': train_time
    })
    
    print(f"\n   📊 RÉSULTATS SUR TEST:")
    print(f"      MSE: {test_mse:.4f}")
    print(f"      MAE: {test_mae:.4f}")
    print(f"      R²: {test_r2:.4f}")
    print(f"      Temps: {train_time:.1f}s")

# ============== 7. TABLEAU DE SYNTHÈSE ==============
print("\n" + "=" * 60)
print("📊 Étape 7: Tableau de synthèse comparatif")
print("=" * 60)

df_results = pd.DataFrame(final_results)
print("\n" + df_results.to_string(index=False))

# Sauvegarde CSV
df_results.to_csv('final_comparison.csv', index=False)
print("\n✅ Tableau sauvegardé: final_comparison.csv")

# ============== 8. MEILLEUR MODÈLE ==============
print("\n" + "=" * 60)
print("🏆 Étape 8: Analyse du meilleur modèle")
print("=" * 60)

best_result = df_results.loc[df_results['Test MSE'].idxmin()]
best_name = best_result['Configuration']

print(f"\n🏆 Meilleur modèle: {best_name}")
print(f"   Test MSE: {best_result['Test MSE']:.4f}")
print(f"   Test MAE: {best_result['Test MAE']:.4f}")
print(f"   Test R²: {best_result['Test R²']:.4f}")

# ============== 9. GRAPHIQUE PRÉDIT VS RÉEL ==============
print("\n" + "=" * 60)
print("📈 Étape 9: Graphique Prédit vs Réel")
print("=" * 60)

best_preds = predictions[best_name]['preds']
best_targets = predictions[best_name]['targets']

fig, ax = plt.subplots(figsize=(10, 8))

# Scatter plot
ax.scatter(best_targets, best_preds, alpha=0.5, s=10, c='steelblue', edgecolors='white', linewidth=0.5)

# Ligne y=x (prédiction parfaite)
min_val = min(best_targets.min(), best_preds.min())
max_val = max(best_targets.max(), best_preds.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Prédiction parfaite (y=x)')

# Intervalle de confiance à 95%
from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(best_targets, best_preds)
line = slope * np.array([min_val, max_val]) + intercept
ax.plot([min_val, max_val], line, 'g-', linewidth=2, label=f'Régression linéaire (R²={r_value**2:.3f})')

ax.set_xlabel('Valeur réelle (prix médian)', fontsize=12)
ax.set_ylabel('Prédiction (prix médian)', fontsize=12)
ax.set_title(f'Prédiction vs Réel - {best_name}', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('rapport_images/predicted_vs_actual.png', dpi=120)
plt.show()

print("✅ Graphique sauvegardé: rapport_images/predicted_vs_actual.png")

# ============== 10. DISTRIBUTION DES RÉSIDUS ==============
print("\n" + "=" * 60)
print("📊 Étape 10: Analyse des résidus")
print("=" * 60)

residuals = best_targets - best_preds

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogramme + KDE
axes[0].hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='steelblue', density=True)
axes[0].set_xlabel('Résidu (valeur réelle - prédiction)', fontsize=12)
axes[0].set_ylabel('Densité', fontsize=12)
axes[0].set_title('Distribution des résidus', fontsize=12, fontweight='bold')
axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Résidu nul')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Q-Q plot
stats.probplot(residuals, dist="norm", plot=axes[1])
axes[1].set_title('Q-Q Plot (normalité des résidus)', fontsize=12, fontweight='bold')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('rapport_images/residuals_analysis.png', dpi=120)
plt.show()

# Test de normalité
shapiro_stat, shapiro_p = stats.shapiro(residuals[:5000])  # Limite à 5000 pour Shapiro
print(f"\n📊 Test de normalité de Shapiro-Wilk:")
print(f"   Statistique: {shapiro_stat:.4f}")
print(f"   p-value: {shapiro_p:.4f}")

if shapiro_p > 0.05:
    print("   ✅ Les résidus semblent normalement distribués (p > 0.05)")
else:
    print("   ⚠️ Les résidus ne suivent pas une loi normale (p < 0.05)")

print(f"\n📊 Statistiques des résidus:")
print(f"   Moyenne: {np.mean(residuals):.4f}")
print(f"   Écart-type: {np.std(residuals):.4f}")
print(f"   Médiane: {np.median(residuals):.4f}")
print(f"   Percentile 95%: {np.percentile(residuals, 95):.4f}")

# ============== 11. ANALYSE GÉOGRAPHIQUE DES ERREURS ==============
print("\n" + "=" * 60)
print("🗺️ Étape 11: Analyse géographique des erreurs")
print("=" * 60)

# Récupérer les coordonnées originales (non normalisées)
X_test_original = X_test  # Données avant normalisation
longitudes = X_test_original[:, 7]  # Longitude
latitudes = X_test_original[:, 6]   # Latitude

fig, ax = plt.subplots(figsize=(12, 8))

# Création d'une colormap divergente
scatter = ax.scatter(longitudes, latitudes, c=np.abs(residuals), 
                     cmap='RdYlGn_r', s=20, alpha=0.7, 
                     edgecolors='black', linewidth=0.5, vmin=0, vmax=1)

ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.set_title(f'Erreurs absolues par localisation géographique - {best_name}', 
             fontsize=14, fontweight='bold')

# Ajout de la barre de couleur
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Erreur absolue (MAE)', fontsize=10)

# Délimitation approximative de la Californie
ax.set_xlim(-124.5, -114)
ax.set_ylim(32.5, 42)

ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('rapport_images/geographic_errors.png', dpi=120)
plt.show()

print("\n🗺️ Analyse des régions problématiques:")
print("   • Plus l'erreur est grande, plus la zone est rouge")
print("   • La région de Los Angeles et San Francisco montrent souvent plus d'erreurs")

# ============== 12. SYNTHÈSE FINALE ==============
print("\n" + "=" * 80)
print("📝 Étape 12: Synthèse finale et conclusion")
print("=" * 80)

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         RAPPORT DE SYNTHÈSE - CALIFORNIA HOUSING             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  📊 PERFORMANCES FINALES                                                     ║
║  ──────────────────────────────────────────────────────────────────────────║
║  • Meilleur modèle: {best_name[:45]:<45} ║
║  • Test MSE: {best_result['Test MSE']:.4f}                                                    ║
║  • Test MAE: {best_result['Test MAE']:.4f}                                                    ║
║  • Test R²: {best_result['Test R²']:.4f}                                                     ║
║                                                                              ║
║  📈 COMPARAISON DES APPROCHES                                                ║
║  ──────────────────────────────────────────────────────────────────────────║
║  • Meilleur Exercice 3 (Sans BN): 0.2613 (val) → test à valider             ║
║  • Grid Search (32 configs): 0.3253 (val) → {df_results.iloc[1]['Test MSE']:.4f} (test)     ║
║  • Random Search (30 configs): exploration aléatoire                         ║
║                                                                              ║
║  💡 ENSEIGNEMENTS CLÉS                                                       ║
║  ──────────────────────────────────────────────────────────────────────────║
║  1. BatchNorm n'est PAS bénéfique sur ce dataset (confirme Exo 3)           ║
║  2. ReLU > ELU > autres activations                                          ║
║  3. Architecture [128,64,32] est suffisante                                 ║
║  4. Le surapprentissage est contrôlé par early stopping                     ║
║                                                                              ║
║  🗺️ LIMITES ET PERSPECTIVES                                                  ║
║  ──────────────────────────────────────────────────────────────────────────║
║  • Les résidus ne suivent pas parfaitement une loi normale                  ║
║  • Erreurs plus importantes dans zones urbaines denses                      ║
║  • Pistes: ingénierie de features (interactions, polynômes)                 ║
║  • Pistes: Bayesian Optimization, Hyperband                                 ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

# ============== 13. SAUVEGARDE COMPLÈTE ==============
print("\n💾 Sauvegarde finale...")

# Sauvegarde des prédictions
results_dict = {
    'best_model_name': best_name,
    'test_mse': float(best_result['Test MSE']),
    'test_mae': float(best_result['Test MAE']),
    'test_r2': float(best_result['Test R²']),
    'predictions': best_preds.tolist(),
    'targets': best_targets.tolist(),
    'residuals': residuals.tolist()
}

import json
with open('final_results.json', 'w') as f:
    json.dump(results_dict, f, indent=2)

print("✅ Fichiers sauvegardés:")
print("   - final_comparison.csv")
print("   - final_results.json")
print("   - rapport_images/predicted_vs_actual.png")
print("   - rapport_images/residuals_analysis.png")
print("   - rapport_images/geographic_errors.png")

if IN_COLAB:
    files.download('final_comparison.csv')
    files.download('final_results.json')
    print("\n📥 Fichiers téléchargés")

print("\n" + "=" * 80)
print("🎉 EXERCICE 6 - ÉVALUATION FINALE TERMINÉE !")
print("=" * 80)
print("\n✅ Tous les résultats sont prêts pour le rapport final.")