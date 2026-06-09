# ex5_random_search.py
"""
EXERCICE 5: Random Search - Recherche aléatoire d'hyperparamètres
Compatible VS Code et Google Colab
Master AIDC - Deep Learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
import os
import warnings
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
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# Création du dossier images
os.makedirs('rapport_images', exist_ok=True)

print("=" * 80)
print("🎲 EXERCICE 5 - RANDOM SEARCH")
print("=" * 80)

# ============== 1. CHARGEMENT DES DONNÉES ==============
print("\n📂 Étape 1: Chargement du dataset...")

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Chargement
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target

# Split
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

# DataLoaders
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)

train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

print(f"✅ Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

# ============== 2. DÉFINITION DU MODÈLE ==============
print("\n📂 Étape 2: Définition du modèle...")

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
    ss_res = np.sum((all_targets - all_preds)**2)
    ss_tot = np.sum((all_targets - np.mean(all_targets))**2)
    r2 = 1 - ss_res/(ss_tot+1e-8)
    return mse, r2

def train_model(model, train_loader, val_loader, config):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'],
                           weight_decay=config.get('weight_decay', 0.0))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                      patience=10, factor=0.5, min_lr=1e-6)
    criterion = nn.MSELoss()

    best_val_mse = float('inf')
    no_improve = 0
    patience = config.get('early_stopping_patience', 15)

    for epoch in range(config.get('epochs', 60)):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                      config.get('clip_value', 1.0))
        val_mse, val_r2 = evaluate(model, val_loader, criterion)
        scheduler.step(val_mse)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    return best_val_mse

print("✅ Fonctions d'entraînement définies")

# ============== 4. ESPACE DE RECHERCHE ==============
print("\n" + "=" * 60)
print("📂 Étape 4: Définition de l'espace de recherche")
print("=" * 60)

# Espace de recherche avec distributions
search_space = {
    'hidden_dims': ['choice', [[64, 32], [128, 64], [128, 64, 32], [256, 128, 64], [256, 128, 64, 32]]],
    'activation': ['choice', ['relu', 'leaky_relu', 'elu', 'tanh']],
    'dropout_rate': ['uniform', 0.0, 0.5],
    'lr': ['log_uniform', 1e-4, 1e-2],
    'weight_decay': ['log_uniform', 1e-5, 1e-2],
    'clip_value': ['uniform', 0.5, 5.0],
}

print("\n📊 Espace de recherche:")
print("   • hidden_dims: choix entre 5 architectures")
print("   • activation: choix entre relu, leaky_relu, elu, tanh")
print("   • dropout_rate: uniforme [0.0, 0.5]")
print("   • lr: log-uniforme [1e-4, 1e-2]")
print("   • weight_decay: log-uniforme [1e-5, 1e-2]")
print("   • clip_value: uniforme [0.5, 5.0]")

# ============== 5. FONCTION D'ÉCHANTILLONNAGE ==============
print("\n📂 Étape 5: Implémentation de l'échantillonnage...")

def sample_config(space):
    """Tire une configuration aléatoire"""
    config = {}

    for key, spec in space.items():
        dist_type = spec[0]

        if dist_type == 'uniform':
            # Distribution uniforme continue
            value = random.uniform(spec[1], spec[2])
        elif dist_type == 'log_uniform':
            # Distribution log-uniforme (pour LR et WD)
            log_min = np.log(spec[1])
            log_max = np.log(spec[2])
            value = np.exp(random.uniform(log_min, log_max))
        elif dist_type == 'choice':
            # Choix discret
            value = random.choice(spec[1])

        config[key] = value

    return config

# Test
test_config = sample_config(search_space)
print("\n🔍 Test d'échantillonnage:")
for k, v in test_config.items():
    print(f"   {k}: {v}")

# ============== 6. RANDOM SEARCH ==============
print("\n" + "=" * 60)
print("🚀 Étape 6: Lancement du Random Search")
print("=" * 60)

N_TRIALS = 30  # 30 configurations aléatoires
EPOCHS = 50

print(f"\n📊 {N_TRIALS} configurations aléatoires × {EPOCHS} epochs max")
print(f"⏱️ Durée estimée: ~{N_TRIALS * 1.2:.0f} minutes")
print("-" * 70)

results = []
start_time = time.time()

for trial in range(N_TRIALS):
    # Échantillonnage aléatoire
    config = sample_config(search_space)
    config['epochs'] = EPOCHS
    config['early_stopping_patience'] = 12
    config['use_bn'] = True  # BatchNorm activé pour tous

    print(f"\n🎲 [Trial {trial+1}/{N_TRIALS}]")
    print(f"   hidden_dims: {config['hidden_dims']}")
    print(f"   activation: {config['activation']}")
    print(f"   dropout_rate: {config['dropout_rate']:.3f}")
    print(f"   lr: {config['lr']:.2e}")
    print(f"   weight_decay: {config['weight_decay']:.2e}")
    print(f"   clip_value: {config['clip_value']:.2f}")

    # Réinitialisation du seed
    torch.manual_seed(trial)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(trial)

    # Création du modèle
    model = DeepFFN(
        input_dim=8,
        hidden_dims=config['hidden_dims'],
        activation=config['activation'],
        use_bn=config['use_bn'],
        dropout_rate=config['dropout_rate']
    )

    # Entraînement
    t0 = time.time()
    best_mse = train_model(model, train_loader, val_loader, config)
    duration = time.time() - t0

    # Stockage
    results.append({
        'trial': trial + 1,
        'hidden_dims': str(config['hidden_dims']),
        'activation': config['activation'],
        'dropout_rate': config['dropout_rate'],
        'lr': config['lr'],
        'weight_decay': config['weight_decay'],
        'clip_value': config['clip_value'],
        'best_val_mse': best_mse,
        'duration': duration
    })

    print(f"   ✅ MSE: {best_mse:.4f} | Temps: {duration:.1f}s")

total_time = time.time() - start_time

# ============== 7. RÉSULTATS ==============
print("\n" + "=" * 60)
print("📊 Étape 7: Résultats du Random Search")
print("=" * 60)

df_results = pd.DataFrame(results).sort_values('best_val_mse')

print(f"\n⏱️ Temps total: {total_time/60:.1f} minutes")
print(f"\n🏆 TOP 10 DES MEILLEURES CONFIGURATIONS:")
print("-" * 100)
print(f"{'Rank':<4} {'Hidden Dims':<22} {'Activation':<12} {'Dropout':<8} {'LR':<10} {'MSE':<10}")
print("-" * 100)

for i, (idx, row) in enumerate(df_results.head(10).iterrows()):
    print(f"{i+1:<4} {row['hidden_dims']:<22} {row['activation']:<12} "
          f"{row['dropout_rate']:<8.3f} {row['lr']:<10.2e} {row['best_val_mse']:<10.4f}")

# Meilleure configuration
best = df_results.iloc[0]
print("\n" + "=" * 60)
print("🏆 MEILLEURE CONFIGURATION RANDOM SEARCH")
print("=" * 60)
print(f"   📐 Hidden dims: {best['hidden_dims']}")
print(f"   ⚡ Activation: {best['activation']}")
print(f"   🎲 Dropout rate: {best['dropout_rate']:.3f}")
print(f"   📉 Learning rate: {best['lr']:.2e}")
print(f"   ⚖️ Weight decay: {best['weight_decay']:.2e}")
print(f"   ✂️ Clip value: {best['clip_value']:.2f}")
print(f"   🎯 Best val MSE: {best['best_val_mse']:.4f}")

# ============== 8. COMPARAISON GRID VS RANDOM ==============
print("\n" + "=" * 60)
print("📈 Étape 8: Comparaison Grid Search vs Random Search")
print("=" * 60)

# Meilleurs résultats
best_grid = 0.3253  # Meilleur MSE du Grid Search
best_random = best['best_val_mse']

print(f"\n📊 Comparaison:")
print(f"   • Meilleur Grid Search: {best_grid:.4f}")
print(f"   • Meilleur Random Search: {best_random:.4f}")
print(f"   • Différence: {(best_random - best_grid)*100:.2f}%")

if best_random < best_grid:
    print(f"   ✅ Random Search a trouvé une meilleure configuration!")
else:
    print(f"   ⚠️ Grid Search reste meilleur sur ce petit échantillon")

# ============== 9. VISUALISATIONS ==============
print("\n" + "=" * 60)
print("📈 Étape 9: Visualisations")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. MSE vs Learning Rate
ax = axes[0, 0]
scatter = ax.scatter(df_results['lr'], df_results['best_val_mse'],
                     c=df_results['dropout_rate'], cmap='viridis',
                     s=100, alpha=0.7, edgecolors='black')
ax.set_xlabel('Learning Rate (log scale)')
ax.set_ylabel('Best Val MSE')
ax.set_title('MSE vs Learning Rate (couleur = dropout)')
ax.set_xscale('log')
ax.grid(alpha=0.3)
plt.colorbar(scatter, ax=ax, label='Dropout Rate')

# 2. MSE vs Weight Decay
ax = axes[0, 1]
scatter = ax.scatter(df_results['weight_decay'], df_results['best_val_mse'],
                     c=df_results['clip_value'], cmap='plasma',
                     s=100, alpha=0.7, edgecolors='black')
ax.set_xlabel('Weight Decay (log scale)')
ax.set_ylabel('Best Val MSE')
ax.set_title('MSE vs Weight Decay (couleur = clip_value)')
ax.set_xscale('log')
ax.grid(alpha=0.3)
plt.colorbar(scatter, ax=ax, label='Clip Value')

# 3. Boxplot par activation
ax = axes[1, 0]
activations = df_results['activation'].unique()
data_by_act = [df_results[df_results['activation'] == act]['best_val_mse'].values
               for act in activations]
bp = ax.boxplot(data_by_act, labels=activations, patch_artist=True)
ax.set_xlabel('Activation Function')
ax.set_ylabel('Best Val MSE')
ax.set_title('Distribution MSE par activation')
ax.grid(alpha=0.3)

# 4. Convergence (best-so-far)
ax = axes[1, 1]
df_sorted = df_results.sort_values('trial')
best_so_far = np.minimum.accumulate(df_sorted['best_val_mse'].values)
ax.plot(range(1, len(best_so_far)+1), best_so_far, 'b-', linewidth=2, marker='o', markersize=4)
ax.set_xlabel('Nombre de configurations évaluées')
ax.set_ylabel('Meilleur MSE trouvé')
ax.set_title('Convergence du Random Search')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('rapport_images/random_search_results.png', dpi=120)
plt.show()

# ============== 10. ANALYSE DES CORRÉLATIONS ==============
print("\n" + "=" * 60)
print("📊 Étape 10: Analyse d'importance des hyperparamètres")
print("=" * 60)

# Corrélation de Spearman avec la MSE
from scipy.stats import spearmanr

print("\n📈 Corrélation de Spearman avec la MSE:")
print("-" * 50)

for param in ['lr', 'weight_decay', 'dropout_rate', 'clip_value']:
    corr, p_value = spearmanr(df_results[param], df_results['best_val_mse'])
    print(f"   {param:15s}: corrélation = {corr:.3f} (p-value = {p_value:.4f})")

print("\n💡 Interprétation:")
print("   • Corrélation positive → plus la valeur est élevée, plus la MSE est élevée (mauvais)")
print("   • Corrélation négative → plus la valeur est élevée, plus la MSE est faible (bon)")

# ============== 11. SAUVEGARDE ==============
df_results.to_csv('random_search_results.csv', index=False)
print("\n✅ Résultats sauvegardés:")
print("   - random_search_results.csv")
print("   - rapport_images/random_search_results.png")

if IN_COLAB:
    files.download('random_search_results.csv')
    print("📥 Fichier CSV téléchargé")

# ============== 12. SYNTHÈSE FINALE ==============
print("\n" + "=" * 80)
print("🎉 EXERCICE 5 - RANDOM SEARCH TERMINÉ !")
print("=" * 80)

print(f"""
📊 RÉSUMÉ FINAL:
   • Configurations testées: {N_TRIALS}
   • Meilleure MSE: {best['best_val_mse']:.4f}
   • Architecture optimale: {best['hidden_dims']}
   • Activation optimale: {best['activation']}
   • Dropout optimal: {best['dropout_rate']:.3f}
   • Learning rate optimal: {best['lr']:.2e}
   • Weight decay optimal: {best['weight_decay']:.2e}
   • Temps total: {total_time/60:.1f} minutes

🏆 COMPARAISON:
   • Grid Search (32 configs): {best_grid:.4f}
   • Random Search (30 configs): {best_random:.4f}
   • Meilleur global (Exo 3 - Sans BN): 0.2613

💡 CONCLUSION:
   Le Random Search a exploré {N_TRIALS} configurations en {total_time/60:.1f} minutes.
   Pour un même budget, il permet d'explorer des zones non visitées par Grid Search.
""")