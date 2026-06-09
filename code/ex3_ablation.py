# ex3_simple.py - Version autonome
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import os

# Créer le dossier images
os.makedirs('rapport_images', exist_ok=True)

print("=" * 80)
print("🔬 EXERCICE 3 - ABLATION STUDY")
print("=" * 80)

# ============== DÉFINITION DU MODÈLE ==============
class DeepFFN(nn.Module):
    def __init__(self, input_dim=8, hidden_dims=[128,64,32], output_dim=1,
                 activation='relu', use_bn=True, dropout_rate=0.2):
        super().__init__()
        self.activation_name = activation
        self.activation = nn.ReLU() if activation=='relu' else nn.ReLU()
        
        self.layers = nn.ModuleList()
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims)-1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
            if use_bn:
                self.layers.append(nn.BatchNorm1d(dims[i+1]))
            self.layers.append(nn.ReLU())
            if dropout_rate > 0:
                self.layers.append(nn.Dropout(dropout_rate))
        
        self.output_layer = nn.Linear(dims[-1], output_dim)
        
        # Initialisation
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)

# ============== FONCTIONS D'ENTRAÎNEMENT ==============
def train_one_epoch(model, loader, optimizer, criterion, clip_value=1.0):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        if clip_value:
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
            pred = model(xb)
            total_loss += criterion(pred, yb).item() * len(xb)
            all_preds.extend(pred.numpy().flatten())
            all_targets.extend(yb.numpy().flatten())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    mse = total_loss / len(loader.dataset)
    mae = np.mean(np.abs(all_preds - all_targets))
    ss_res = np.sum((all_targets - all_preds)**2)
    ss_tot = np.sum((all_targets - np.mean(all_targets))**2)
    r2 = 1 - ss_res/(ss_tot+1e-8)
    return mse, mae, r2

def train_model(model, train_loader, val_loader, config):
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config.get('weight_decay', 0))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5, min_lr=1e-6)
    criterion = nn.MSELoss()
    
    history = {'val_mse': []}
    best_val_mse = float('inf')
    no_improve = 0
    patience = config.get('early_stopping_patience', 15)
    
    for epoch in range(config.get('epochs', 100)):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, config.get('clip_value', 1.0))
        val_mse, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step(val_mse)
        history['val_mse'].append(val_mse)
        
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            no_improve = 0
        else:
            no_improve += 1
        
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1:3d} | Train MSE={train_loss:.4f} | Val MSE={val_mse:.4f}")
        
        if no_improve >= patience:
            print(f"Early stopping à l'epoch {epoch+1}")
            break
    
    return history, best_val_mse

# ============== CHARGEMENT DES DONNÉES ==============
print("\n📂 Chargement des données...")
with open('data_loaders.pkl', 'rb') as f:
    data = pickle.load(f)
    train_loader = data['train_loader']
    val_loader = data['val_loader']

# ============== CONFIGURATIONS ==============
configs = {
    'A: Baseline complet': {'use_bn': True, 'dropout_rate': 0.2, 'weight_decay': 1e-4, 'color': 'blue'},
    'B: Sans BatchNorm': {'use_bn': False, 'dropout_rate': 0.2, 'weight_decay': 1e-4, 'color': 'red'},
    'C: Sans Dropout': {'use_bn': True, 'dropout_rate': 0.0, 'weight_decay': 1e-4, 'color': 'green'},
    'D: Sans L2': {'use_bn': True, 'dropout_rate': 0.2, 'weight_decay': 0.0, 'color': 'orange'},
    'E: Aucune régul': {'use_bn': False, 'dropout_rate': 0.0, 'weight_decay': 0.0, 'color': 'purple'}
}

base_config = {
    'hidden_dims': [128, 64, 32], 'activation': 'relu',
    'lr': 1e-3, 'clip_value': 1.0, 'epochs': 100, 'early_stopping_patience': 15
}

# ============== ENTRAÎNEMENT ==============
results = {}
histories = {}

for name, cfg in configs.items():
    print(f"\n🚀 {name}")
    full_config = {**base_config, **cfg}
    torch.manual_seed(42)
    model = DeepFFN(use_bn=cfg['use_bn'], dropout_rate=cfg['dropout_rate'])
    history, best_mse = train_model(model, train_loader, val_loader, full_config)
    results[name] = best_mse
    histories[name] = history['val_mse']

# ============== GRAPHIQUE ==============
plt.figure(figsize=(12, 6))
for name, hist in histories.items():
    plt.plot(hist, label=name, color=configs[name]['color'], linewidth=2)

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Validation MSE', fontsize=12)
plt.title('Ablation Study - Impact des techniques de régularisation', fontsize=14)
plt.legend(loc='upper right')
plt.grid(alpha=0.3)
plt.yscale('log')
plt.tight_layout()
plt.savefig('rapport_images/ablation_results.png', dpi=120)
plt.show()

# ============== RÉSULTATS ==============
print("\n" + "=" * 60)
print("📊 RÉSULTATS FINAUX")
print("=" * 60)
for name, mse in sorted(results.items(), key=lambda x: x[1]):
    print(f"{name:25} | Best Val MSE: {mse:.4f}")

print("\n🎉 EXERCICE 3 TERMINÉ!")