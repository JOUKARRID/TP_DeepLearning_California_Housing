# ex2_model.py
"""
EXERCICE 2: Implémentation du MLP avec régularisation
Master AIDC - Deep Learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import os

# Créer le dossier pour les images s'il n'existe pas
if not os.path.exists('rapport_images'):
    os.makedirs('rapport_images')

print("=" * 80)
print("🏗️ EXERCICE 2 - Construction du MLP avec régularisation")
print("=" * 80)

# ============== CHARGEMENT DES DONNÉES ==============
print("\n📂 Chargement des DataLoaders depuis l'Exercice 1...")
with open('data_loaders.pkl', 'rb') as f:
    data = pickle.load(f)
    train_loader = data['train_loader']
    val_loader = data['val_loader']
    test_loader = data['test_loader']
    scaler = data['scaler']

print(f"✅ Train: {len(train_loader.dataset)} exemples")
print(f"✅ Validation: {len(val_loader.dataset)} exemples")
print(f"✅ Test: {len(test_loader.dataset)} exemples")

# ============== Q1-Q3: CLASSE DeepFFN ==============
print("\n" + "=" * 60)
print("Q1-Q3: Construction du réseau DeepFFN")
print("=" * 60)


class DeepFFN(nn.Module):
    """Réseau Feedforward Profond avec BatchNorm, Dropout et régularisation"""
    
    def __init__(self,
                 input_dim: int = 8,
                 hidden_dims: list = [128, 64, 32],
                 output_dim: int = 1,
                 activation: str = 'relu',
                 use_bn: bool = True,
                 dropout_rate: float = 0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation_name = activation
        self.use_bn = use_bn
        self.dropout_rate = dropout_rate
        
        # Récupérer l'activation
        self.activation = self._get_activation(activation)
        
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
    
    def _get_activation(self, name):
        """Retourne la fonction d'activation"""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.01),
            'tanh': nn.Tanh(),
            'elu': nn.ELU(),
            'selu': nn.SELU()
        }
        return activations.get(name, nn.ReLU())
    
    def _init_weights(self):
        """Initialisation adaptée (He pour ReLU, Xavier pour les autres)"""
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


# Test du modèle
model_test = DeepFFN()
print(model_test)
total_params = sum(p.numel() for p in model_test.parameters() if p.requires_grad)
print(f"\n📊 Paramètres entraînables: {total_params:,}")

# Test forward pass
sample_x, _ = next(iter(train_loader))
with torch.no_grad():
    sample_out = model_test(sample_x)
print(f"✅ Forward pass: {sample_x.shape} → {sample_out.shape}")

# ============== Q4: TRAIN ONE EPOCH ==============
print("\n" + "=" * 60)
print("Q4: Fonction train_one_epoch avec Gradient Clipping")
print("=" * 60)


def train_one_epoch(model, loader, optimizer, criterion, clip_value=1.0):
    """Entraîne sur une époque avec gradient clipping"""
    model.train()
    total_loss = 0.0
    total_grad_norm = 0.0
    n_batches = 0
    
    for xb, yb in loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        
        # Gradient clipping
        if clip_value is not None and clip_value > 0:
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            total_grad_norm += grad_norm.item()
        
        optimizer.step()
        total_loss += loss.item() * len(xb)
        n_batches += 1
    
    avg_loss = total_loss / len(loader.dataset)
    avg_grad_norm = total_grad_norm / n_batches if clip_value else 0
    return avg_loss, avg_grad_norm


print("✅ train_one_epoch créée")

# ============== Q5: EVALUATE ==============
print("\n" + "=" * 60)
print("Q5: Fonction evaluate (MSE, MAE, R²)")
print("=" * 60)


def evaluate(model, loader, criterion):
    """Évalue le modèle et retourne MSE, MAE, R²"""
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb)
            total_loss += criterion(pred, yb).item() * len(xb)
            all_preds.extend(pred.numpy().flatten())
            all_targets.extend(yb.numpy().flatten())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    mse = total_loss / len(loader.dataset)
    mae = np.mean(np.abs(all_preds - all_targets))
    
    # R²
    ss_res = np.sum((all_targets - all_preds) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    
    return mse, mae, r2


# Test
test_mse, test_mae, test_r2 = evaluate(model_test, val_loader, nn.MSELoss())
print(f"✅ Test (modèle non entraîné): MSE={test_mse:.4f}, MAE={test_mae:.4f}, R²={test_r2:.4f}")

# ============== Q6: TRAIN MODEL ==============
print("\n" + "=" * 60)
print("Q6: Boucle d'entraînement avec Early Stopping")
print("=" * 60)


def train_model(model, train_loader, val_loader, config):
    """Entraînement complet avec early stopping et scheduler"""
    
    optimizer = optim.Adam(model.parameters(),
                          lr=config['lr'],
                          weight_decay=config.get('weight_decay', 0.0))
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5, min_lr=1e-6
    )
    
    criterion = nn.MSELoss()
    clip_value = config.get('clip_value', 1.0)
    
    history = {'train_mse': [], 'val_mse': [], 'val_mae': [], 'val_r2': [], 'lr': [], 'grad_norm': []}
    
    best_val_mse = float('inf')
    patience = config.get('early_stopping_patience', 20)
    no_improve = 0
    t0 = time.time()
    
    print(f"\n🚀 Entraînement - LR={config['lr']}, WD={config.get('weight_decay',0)}")
    print("-" * 50)
    
    for epoch in range(config.get('epochs', 200)):
        train_loss, grad_norm = train_one_epoch(model, train_loader, optimizer, criterion, clip_value)
        val_mse, val_mae, val_r2 = evaluate(model, val_loader, criterion)
        scheduler.step(val_mse)
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_mse'].append(train_loss)
        history['val_mse'].append(val_mse)
        history['val_mae'].append(val_mae)
        history['val_r2'].append(val_r2)
        history['lr'].append(current_lr)
        history['grad_norm'].append(grad_norm)
        
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            no_improve += 1
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d} | Train MSE={train_loss:.4f} | Val MSE={val_mse:.4f} | R²={val_r2:.4f} | LR={current_lr:.6f}")
        
        if no_improve >= patience:
            print(f"\n⏹️ Early stopping à l'epoch {epoch+1}")
            break
    
    duration = time.time() - t0
    print(f"\n✅ Terminé en {duration:.1f}s | Best Val MSE={best_val_mse:.4f}")
    
    return history, best_val_mse, duration


print("✅ train_model créée")

# ============== Q7: MODÈLE BASELINE ==============
print("\n" + "=" * 60)
print("Q7: Entraînement du modèle baseline")
print("=" * 60)

config_baseline = {
    'hidden_dims': [128, 64, 32],
    'activation': 'relu',
    'use_bn': True,
    'dropout_rate': 0.2,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'clip_value': 1.0,
    'epochs': 200,
    'early_stopping_patience': 25,
}

print("Configuration baseline:")
for k, v in config_baseline.items():
    print(f"   {k}: {v}")

torch.manual_seed(42)
model_baseline = DeepFFN(
    input_dim=8,
    hidden_dims=config_baseline['hidden_dims'],
    activation=config_baseline['activation'],
    use_bn=config_baseline['use_bn'],
    dropout_rate=config_baseline['dropout_rate']
)

history, best_val_mse, duration = train_model(model_baseline, train_loader, val_loader, config_baseline)

# ============== TRACÉ DES COURBES ==============
print("\n" + "=" * 60)
print("Tracé des courbes d'apprentissage")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. MSE
axes[0,0].plot(history['train_mse'], label='Train MSE', linewidth=2)
axes[0,0].plot(history['val_mse'], label='Val MSE', linewidth=2)
axes[0,0].set_xlabel('Epoch')
axes[0,0].set_ylabel('MSE')
axes[0,0].set_title('MSE (Train vs Validation)')
axes[0,0].legend()
axes[0,0].grid(alpha=0.3)

# 2. MAE
axes[0,1].plot(history['val_mae'], color='orange', linewidth=2)
axes[0,1].set_xlabel('Epoch')
axes[0,1].set_ylabel('MAE')
axes[0,1].set_title('MAE sur validation')
axes[0,1].grid(alpha=0.3)

# 3. R²
axes[0,2].plot(history['val_r2'], color='green', linewidth=2)
axes[0,2].axhline(y=0, color='red', linestyle='--', label='R²=0')
axes[0,2].set_xlabel('Epoch')
axes[0,2].set_ylabel('R²')
axes[0,2].set_title('R² sur validation')
axes[0,2].legend()
axes[0,2].grid(alpha=0.3)

# 4. Learning Rate
axes[1,0].plot(history['lr'], color='purple', linewidth=2)
axes[1,0].set_xlabel('Epoch')
axes[1,0].set_ylabel('Learning Rate')
axes[1,0].set_title('Learning Rate (échelle log)')
axes[1,0].set_yscale('log')
axes[1,0].grid(alpha=0.3)

# 5. Gradient Norm
axes[1,1].plot(history['grad_norm'], color='red', linewidth=2)
axes[1,1].set_xlabel('Epoch')
axes[1,1].set_ylabel('Norme du gradient')
axes[1,1].set_title('Norme du gradient (avec clipping)')
axes[1,1].grid(alpha=0.3)

# 6. Résumé
axes[1,2].axis('off')
axes[1,2].text(0.1, 0.9, f"📊 RÉSUMÉ FINAL", fontsize=12, fontweight='bold')
axes[1,2].text(0.1, 0.7, f"Best Val MSE: {best_val_mse:.4f}", fontsize=11)
axes[1,2].text(0.1, 0.55, f"Final R²: {history['val_r2'][-1]:.4f}", fontsize=11)
axes[1,2].text(0.1, 0.4, f"Epochs: {len(history['train_mse'])}", fontsize=11)
axes[1,2].text(0.1, 0.25, f"Temps: {duration:.1f} sec", fontsize=11)

plt.tight_layout()
plt.savefig('rapport_images/baseline_curves.png', dpi=120)
plt.show()

# ============== RÉSULTATS ==============
print("\n" + "=" * 80)
print("🎉 EXERCICE 2 TERMINÉ!")
print("=" * 80)
print(f"\n📊 RÉSULTATS BASELINE:")
print(f"   🏆 Meilleure MSE (validation): {best_val_mse:.4f}")
print(f"   📈 R² final: {history['val_r2'][-1]:.4f}")
print(f"   ⏱️ Temps: {duration:.1f} secondes")
print(f"   💾 Modèle sauvegardé: best_model.pth")
print(f"   📁 Graphique: rapport_images/baseline_curves.png")

