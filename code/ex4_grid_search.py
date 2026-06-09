# ex4_grid_search.py
"""
EXERCICE 4: Grid Search - Recherche exhaustive d'hyperparamètres
Compatible VS Code et Google Colab
Master AIDC - Deep Learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import time
import os
import warnings
warnings.filterwarnings('ignore')

# Détection de l'environnement (Colab ou local)
try:
    from google.colab import files
    IN_COLAB = True
    print("✅ Environnement: Google Colab")
except ImportError:
    IN_COLAB = False
    print("✅ Environnement: VS Code / Local")

# Vérification de PyTorch et GPU
print(f"✅ PyTorch version: {torch.__version__}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# Création du dossier pour les images
os.makedirs('rapport_images', exist_ok=True)

print("=" * 80)
print("🔍 EXERCICE 4 - GRID SEARCH")
print("=" * 80)

# ============== 1. CHARGEMENT ET PRÉPARATION DES DONNÉES ==============
print("\n📂 Étape 1: Chargement du dataset California Housing...")

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Fixer les seeds pour reproductibilité
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Chargement du dataset
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target

print(f"✅ Dataset chargé: {df.shape[0]} quartiers, {df.shape[1]-1} features")

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

print(f"✅ Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# DataLoaders
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)

train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

print(f"✅ DataLoaders créés")

# ============== 2. DÉFINITION DU MODÈLE ==============
print("\n📂 Étape 2: Définition du modèle DeepFFN...")

class DeepFFN(nn.Module):
    """Réseau Feedforward Profond avec régularisation"""
    
    def __init__(self, input_dim=8, hidden_dims=[128, 64, 32], output_dim=1,
                 activation='relu', use_bn=True, dropout_rate=0.2):
        super().__init__()
        
        self.activation_name = activation
        self.use_bn = use_bn
        self.dropout_rate = dropout_rate
        
        # Sélection de l'activation
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.01),
            'elu': nn.ELU(),
            'tanh': nn.Tanh(),
            'selu': nn.SELU()
        }
        self.activation = activations.get(activation, nn.ReLU())
        
        # Construction des couches
        self.layers = nn.ModuleList()
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            # Couche linéaire
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
            
            # BatchNorm (optionnelle)
            if use_bn:
                self.layers.append(nn.BatchNorm1d(dims[i+1]))
            
            # Activation
            self.layers.append(self.activation)
            
            # Dropout (optionnel)
            if dropout_rate > 0:
                self.layers.append(nn.Dropout(dropout_rate))
        
        # Couche de sortie
        self.output_layer = nn.Linear(dims[-1], output_dim)
        
        # Initialisation des poids
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

print("✅ Modèle DeepFFN défini")

# ============== 3. FONCTIONS D'ENTRAÎNEMENT ==============
print("\n📂 Étape 3: Définition des fonctions d'entraînement...")

def train_one_epoch(model, loader, optimizer, criterion, clip_value=1.0):
    """Entraîne sur une époque avec gradient clipping"""
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
    """Évalue le modèle et retourne MSE, MAE, R²"""
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
    
    # R²
    ss_res = np.sum((all_targets - all_preds) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    
    return mse, mae, r2


def train_model(model, train_loader, val_loader, config):
    """Boucle d'entraînement complète avec early stopping"""
    
    model = model.to(device)
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config.get('weight_decay', 0.0)
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5, min_lr=1e-6
    )
    
    criterion = nn.MSELoss()
    clip_value = config.get('clip_value', 1.0)
    
    best_val_mse = float('inf')
    no_improve = 0
    patience = config.get('early_stopping_patience', 15)
    
    for epoch in range(config.get('epochs', 60)):
        # Entraînement
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, clip_value)
        
        # Évaluation
        val_mse, val_mae, val_r2 = evaluate(model, val_loader, criterion)
        
        # Scheduler
        scheduler.step(val_mse)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Early stopping
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            no_improve = 0
        else:
            no_improve += 1
        
        # Affichage périodique
        if (epoch + 1) % 15 == 0:
            print(f"   Epoch {epoch+1:3d} | Train MSE={train_loss:.4f} | "
                  f"Val MSE={val_mse:.4f} | R²={val_r2:.4f} | LR={current_lr:.6f}")
        
        if no_improve >= patience:
            break
    
    return best_val_mse


print("✅ Fonctions d'entraînement définies")

# ============== 4. DÉFINITION DE LA GRILLE ==============
print("\n" + "=" * 60)
print("📂 Étape 4: Définition de la grille d'hyperparamètres")
print("=" * 60)

# Version réduite pour un temps d'exécution raisonnable (16 configurations)
param_grid = {
    'hidden_dims': [[128, 64, 32], [256, 128, 64]],
    'activation': ['relu', 'elu'],
    'dropout_rate': [0.0, 0.2],
    'lr': [1e-3, 5e-4],
    'weight_decay': [0.0, 1e-4],
}

# Version complète (décommentez si vous avez plus de temps)
# param_grid = {
#     'hidden_dims': [[64, 32], [128, 64], [128, 64, 32], [256, 128, 64]],
#     'activation': ['relu', 'leaky_relu', 'elu'],
#     'dropout_rate': [0.0, 0.2, 0.3],
#     'lr': [1e-3, 5e-4],
#     'weight_decay': [0.0, 1e-4],
# }

# Calcul du nombre de configurations
n_configs = 1
for v in param_grid.values():
    n_configs *= len(v)

print(f"\n📊 Grille d'hyperparamètres:")
for key, values in param_grid.items():
    print(f"   {key}: {values}")
print(f"\n📈 Nombre total de configurations: {n_configs}")
print(f"⏱️ Durée estimée: ~{n_configs * 1.5:.0f} minutes")

# ============== 5. EXÉCUTION DU GRID SEARCH ==============
print("\n" + "=" * 60)
print("🚀 Étape 5: Lancement du Grid Search")
print("=" * 60)

keys = list(param_grid.keys())
values = list(param_grid.values())
combos = list(itertools.product(*values))

results = []
start_time = time.time()

for i, combo in enumerate(combos):
    # Création de la configuration
    config = dict(zip(keys, combo))
    config['epochs'] = 60
    config['early_stopping_patience'] = 15
    config['use_bn'] = True
    config['clip_value'] = 1.0
    
    print(f"\n🔍 [{i+1}/{n_configs}] Configuration:")
    print(f"   Hidden dims: {config['hidden_dims']}")
    print(f"   Activation: {config['activation']}")
    print(f"   Dropout: {config['dropout_rate']}")
    print(f"   LR: {config['lr']}")
    print(f"   Weight decay: {config['weight_decay']}")
    
    # Réinitialisation du seed
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    
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
    
    # Stockage des résultats
    results.append({
        'hidden_dims': str(config['hidden_dims']),
        'activation': config['activation'],
        'dropout_rate': config['dropout_rate'],
        'lr': config['lr'],
        'weight_decay': config['weight_decay'],
        'best_val_mse': best_mse,
        'duration': duration
    })
    
    print(f"   ✅ Best val MSE: {best_mse:.4f} | Temps: {duration:.1f}s")

total_time = time.time() - start_time

# ============== 6. RÉSULTATS ==============
print("\n" + "=" * 60)
print("📊 Étape 6: Résultats du Grid Search")
print("=" * 60)

df_results = pd.DataFrame(results).sort_values('best_val_mse')

print(f"\n⏱️ Temps total: {total_time/60:.1f} minutes")
print(f"\n🏆 CLASSEMENT (Top 10):")
print("-" * 100)
print(f"{'Rank':<4} {'Hidden Dims':<22} {'Activation':<10} {'Dropout':<8} {'LR':<10} {'WD':<12} {'MSE':<10}")
print("-" * 100)

for i, (idx, row) in enumerate(df_results.head(min(10, len(df_results))).iterrows()):
    print(f"{i+1:<4} {row['hidden_dims']:<22} {row['activation']:<10} "
          f"{row['dropout_rate']:<8.2f} {row['lr']:<10.4f} {row['weight_decay']:<12.4f} "
          f"{row['best_val_mse']:<10.4f}")

# Meilleure configuration
best = df_results.iloc[0]
print("\n" + "=" * 60)
print("🏆 MEILLEURE CONFIGURATION TROUVÉE")
print("=" * 60)
print(f"   📐 Hidden dims: {best['hidden_dims']}")
print(f"   ⚡ Activation: {best['activation']}")
print(f"   🎲 Dropout rate: {best['dropout_rate']}")
print(f"   📉 Learning rate: {best['lr']}")
print(f"   ⚖️ Weight decay: {best['weight_decay']}")
print(f"   🎯 Best val MSE: {best['best_val_mse']:.4f}")

# ============== 7. VISUALISATIONS ==============
print("\n" + "=" * 60)
print("📈 Étape 7: Visualisation des résultats")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. MSE par activation
ax = axes[0, 0]
for activation in df_results['activation'].unique():
    subset = df_results[df_results['activation'] == activation]
    ax.scatter(subset['dropout_rate'], subset['best_val_mse'], 
               label=activation, s=100, alpha=0.7)
ax.set_xlabel('Dropout Rate')
ax.set_ylabel('Best Val MSE')
ax.set_title('Impact de l\'activation et du dropout')
ax.legend()
ax.grid(alpha=0.3)

# 2. MSE par learning rate
ax = axes[0, 1]
for lr in df_results['lr'].unique():
    subset = df_results[df_results['lr'] == lr]
    ax.scatter(subset['weight_decay'], subset['best_val_mse'], 
               label=f'LR={lr}', s=100, alpha=0.7)
ax.set_xlabel('Weight Decay (L2)')
ax.set_ylabel('Best Val MSE')
ax.set_title('Impact du learning rate et weight decay')
ax.legend()
ax.grid(alpha=0.3)
if df_results['weight_decay'].max() > 0:
    ax.set_xscale('log')

# 3. Boxplot par architecture
ax = axes[1, 0]
hidden_list = df_results['hidden_dims'].unique()
data_to_plot = [df_results[df_results['hidden_dims'] == hd]['best_val_mse'].values 
                for hd in hidden_list]
bp = ax.boxplot(data_to_plot, labels=[str(hd).replace('[', '').replace(']', '') 
                                       for hd in hidden_list], patch_artist=True)
ax.set_xlabel('Architecture (hidden dims)')
ax.set_ylabel('Best Val MSE')
ax.set_title('Distribution MSE par architecture')
ax.grid(alpha=0.3)

# 4. Top configurations
ax = axes[1, 1]
top_n = min(8, len(df_results))
top_data = df_results.head(top_n)
colors = plt.cm.Greens(np.linspace(0.3, 0.8, top_n))
bars = ax.barh(range(top_n), top_data['best_val_mse'].values, color=colors[::-1])
ax.set_yticks(range(top_n))
ax.set_yticklabels([f"{row['activation']}, dr={row['dropout_rate']}" 
                    for _, row in top_data.iterrows()], fontsize=9)
ax.set_xlabel('Best Val MSE')
ax.set_title(f'Top {top_n} des configurations')
ax.invert_yaxis()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('rapport_images/grid_search_results.png', dpi=120)
plt.show()

# Sauvegarde CSV
df_results.to_csv('grid_search_results.csv', index=False)
print("\n✅ Résultats sauvegardés:")
print("   - grid_search_results.csv")
print("   - rapport_images/grid_search_results.png")

# Export pour Colab
if IN_COLAB:
    from google.colab import files
    files.download('grid_search_results.csv')
    print("\n📥 Fichier CSV téléchargé automatiquement")

# ============== 8. SYNTHÈSE FINALE ==============
print("\n" + "=" * 80)
print("🎉 EXERCICE 4 - GRID SEARCH TERMINÉ !")
print("=" * 80)

print(f"""
📊 RÉSUMÉ FINAL:
   • Configurations testées: {len(df_results)}
   • Meilleure MSE: {best['best_val_mse']:.4f}
   • Architecture optimale: {best['hidden_dims']}
   • Activation optimale: {best['activation']}
   • Dropout optimal: {best['dropout_rate']}
   • Learning rate optimal: {best['lr']}
   • Weight decay optimal: {best['weight_decay']}
   • Temps total: {total_time/60:.1f} minutes

💡 Comparaison avec Exercice 3:
   • Meilleur résultat Exercice 3 (Sans BN): 0.2613
   • Meilleur résultat Grid Search: {best['best_val_mse']:.4f}
   • Amélioration: {(0.2613 - best['best_val_mse']) * 100:.1f}%
""")