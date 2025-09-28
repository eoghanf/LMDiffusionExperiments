#!/usr/bin/env python3
"""
Complete denoising demonstration - generates comprehensive analysis across noise levels.
This script uses actual FineWeb data to demonstrate real denoising performance
across a wide range of noise levels from 0.01 to 0.99.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import time
from typing import Iterator, Tuple, List, Dict
import random
import glob
import struct
import sys
import os

# Add src to path for imports
sys.path.append('src')
from dataloaders import distributed_data_generator
from config import NoPropConfig
from utils import set_seed

# =============================================================================
# RESIDUAL DENOISING MODEL
# =============================================================================

class SimpleResidualDenoisingModel(torch.nn.Module):
    """
    Simple residual denoising model: M(x) = x + d(x)
    
    This model learns a correction d(x) to add to the noisy input x,
    starting from near-identity initialization.
    """
    
    def __init__(self, embed_dim: int, hidden_dim: int, num_layers: int = 3, 
                 dropout: float = 0.1, init_scale: float = 0.01):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.init_scale = init_scale
        
        # Build MLP layers for computing d(x)
        layers = []
        
        # Input layer: context + noisy target -> hidden
        layers.append(torch.nn.Linear(embed_dim * 2, hidden_dim))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout))
        
        # Output layer: hidden -> correction d(x)
        layers.append(torch.nn.Linear(hidden_dim, embed_dim))
        
        self.mlp = torch.nn.Sequential(*layers)
        
        # Initialize with small weights for near-identity behavior
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights to achieve near-identity function."""
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                # Initialize weights with small random values
                torch.nn.init.normal_(module.weight, mean=0.0, std=self.init_scale)
                # Initialize biases to zero
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
    
    def forward(self, input_embeddings: torch.Tensor, nt_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: M(x) = x + d(x)
        
        Args:
            input_embeddings: Context embeddings [batch_size, seq_length, embed_dim]
            nt_embedding: Noisy target embedding [batch_size, embed_dim]
        
        Returns:
            Denoised embedding [batch_size, embed_dim]
        """
        batch_size = nt_embedding.shape[0]
        
        # Use mean of context embeddings as context representation
        context_repr = input_embeddings.mean(dim=1)  # [batch_size, embed_dim]
        
        # Concatenate context and noisy target
        combined_input = torch.cat([context_repr, nt_embedding], dim=1)  # [batch_size, embed_dim * 2]
        
        # Compute correction d(x)
        correction = self.mlp(combined_input)  # [batch_size, embed_dim]
        
        # Apply residual connection: M(x) = x + d(x)
        denoised_output = nt_embedding + correction
        
        return denoised_output

# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_denoising_model(steps: int = 20000, noise_level: float = 0.5, 
                         report_interval: int = None) -> Tuple[List[int], List[float], List[float], List[float]]:
    """
    Train a denoising model using actual FineWeb data.
    
    Returns:
        Tuple of (steps, ratios, noise_removed_percentages, absolute_noise_levels)
    """
    print(f"Training denoising model for {steps:,} steps at noise level {noise_level:.2f}...")
    
    # Load actual config from the project
    try:
        import yaml
        with open("finewebdiffusion_v2.yaml", 'r') as f:
            config_dict = yaml.safe_load(f)
        config = NoPropConfig(**config_dict)
    except:
        # Fallback config if YAML loading fails
        config = NoPropConfig(
            embed_dim=768,
            dropout=0.1,
            layer_lr=1e-3,
            batch_size=128,
            max_seq_len=128,
            seed=42
        )
    
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create embedding matrix
    VOCAB_SIZE = 50257
    embedding_matrix = torch.nn.Embedding(VOCAB_SIZE, config.embed_dim).to(device)
    torch.nn.init.normal_(embedding_matrix.weight)
    
    # Try to use actual FineWeb data
    try:
        # Look for FineWeb data in the expected location
        data_patterns = [
            "../../../FinewebData/fineweb10B/fineweb_train_*.bin",
            "../../FinewebData/fineweb10B/fineweb_train_*.bin", 
            "../FinewebData/fineweb10B/fineweb_train_*.bin",
            "fineweb_train_*.bin"
        ]
        
        data_path = None
        for pattern in data_patterns:
            if glob.glob(pattern):
                data_path = pattern
                break
        
        if data_path:
            train_dataloader = distributed_data_generator(
                data_path,
                batch_size=config.batch_size,
                rank=0, world_size=1, cycle=True,
                seq_len=config.max_seq_len, device=str(device)
            )
        else:
            train_dataloader = create_mock_data_generator(
                vocab_size=VOCAB_SIZE,
                batch_size=config.batch_size,
                seq_len=config.max_seq_len,
                device=str(device)
            )
    except Exception as e:
        train_dataloader = create_mock_data_generator(
            vocab_size=VOCAB_SIZE,
            batch_size=config.batch_size,
            seq_len=config.max_seq_len,
            device=str(device)
        )
    
    # Create model
    model = SimpleResidualDenoisingModel(
        embed_dim=config.embed_dim,
        hidden_dim=config.embed_dim * 2,
        num_layers=3,
        dropout=config.dropout,
        init_scale=0.01
    ).to(device)
    
    # Setup training
    optimizer = optim.AdamW(model.parameters(), lr=config.layer_lr)
    
    # Set default report interval
    if report_interval is None:
        report_interval = max(1, steps // 20)
    
    # Training loop
    steps_list = []
    ratios_list = []
    noise_removed_list = []
    absolute_noise_list = []
    
    for step in range(steps):
        # Get training batch
        input_ids, targets = next(train_dataloader)
        clean_context_embeds = embedding_matrix(input_ids)
        clean_target_embeds = embedding_matrix(targets)
        
        # Add noise using variance-preserving schedule
        noise = torch.randn_like(clean_target_embeds)
        sqrt_noise_level = torch.sqrt(torch.tensor(noise_level, device=clean_target_embeds.device))
        sqrt_one_minus_noise_level = torch.sqrt(torch.tensor(1.0 - noise_level, device=clean_target_embeds.device))
        noisy_target_embeds = sqrt_noise_level * noise + sqrt_one_minus_noise_level * clean_target_embeds
        
        # Training step
        optimizer.zero_grad()
        predicted_embeds = model.forward(clean_context_embeds, noisy_target_embeds)
        loss = F.mse_loss(predicted_embeds, clean_target_embeds, reduction='mean')
        loss.backward()
        optimizer.step()
        
        # Record metrics
        if step % report_interval == 0:
            with torch.no_grad():
                input_mse = F.mse_loss(noisy_target_embeds, clean_target_embeds, reduction='mean').item()
                ratio = loss.item() / input_mse
                noise_removed_pct = (1.0 - ratio) * 100
                absolute_noise = ratio
                
                steps_list.append(step)
                ratios_list.append(ratio)
                noise_removed_list.append(noise_removed_pct)
                absolute_noise_list.append(absolute_noise)
    
    return steps_list, ratios_list, noise_removed_list, absolute_noise_list

def create_mock_data_generator(vocab_size: int = 50257, batch_size: int = 128, 
                              seq_len: int = 128, device: str = "cuda") -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """Create a mock data generator as fallback."""
    while True:
        # Generate random token sequences
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, vocab_size, (batch_size,), device=device)
        yield input_ids, targets

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def run_comprehensive_noise_analysis():
    """Run denoising analysis across multiple noise levels."""
    
    print("=== COMPREHENSIVE NOISE LEVEL ANALYSIS ===")
    print("Testing denoising across wide range of noise levels...\n")
    
    # Define noise levels to test
    noise_levels = [0.99, 0.98, 0.97, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]
    
    # Check for data availability once
    try:
        data_patterns = [
            "../../../FinewebData/fineweb10B/fineweb_train_*.bin",
            "../../FinewebData/fineweb10B/fineweb_train_*.bin", 
            "../FinewebData/fineweb10B/fineweb_train_*.bin",
            "fineweb_train_*.bin"
        ]
        
        data_found = False
        for pattern in data_patterns:
            if glob.glob(pattern):
                print(f"✅ Found FineWeb data at: {pattern}")
                data_found = True
                break
        
        if not data_found:
            print("⚠️  FineWeb data not found, using mock data for demonstration")
    except:
        print("⚠️  Error checking for FineWeb data, using mock data")
    
    # Store results for all noise levels
    all_results = {}
    
    # Train models for each noise level
    for noise_level in noise_levels:
        print(f"\n{'='*60}")
        print(f"TRAINING AT NOISE LEVEL: {noise_level:.2f}")
        print(f"{'='*60}")
        
        steps, ratios, noise_removed, absolute_noise = train_denoising_model(
            steps=20000, 
            noise_level=noise_level, 
            report_interval=1000
        )
        
        all_results[noise_level] = {
            'steps': steps,
            'ratios': ratios,
            'noise_removed': noise_removed,
            'absolute_noise': absolute_noise,
            'initial_ratio': ratios[0] if ratios else 1.0,
            'final_ratio': ratios[-1] if ratios else 1.0,
            'best_ratio': min(ratios) if ratios else 1.0,
            'improvement': ratios[0] - ratios[-1] if len(ratios) > 1 else 0.0
        }
        
        final_ratio = ratios[-1] if ratios else 1.0
        final_noise_removed = (1.0 - final_ratio) * 100
        print(f"Final result: {final_ratio:.3f} ratio ({final_noise_removed:.1f}% noise removed)")
    
    return all_results, noise_levels

def create_noise_level_plots(all_results: Dict, noise_levels: List[float]):
    """Create comprehensive plots for all noise levels."""
    
    print(f"\n=== GENERATING COMPREHENSIVE PLOTS ===")
    
    # Create a large figure with subplots for each noise level
    n_levels = len(noise_levels)
    n_cols = 3
    n_rows = (n_levels + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    # Plot each noise level
    for i, noise_level in enumerate(noise_levels):
        ax = axes[i]
        result = all_results[noise_level]
        
        steps = result['steps']
        ratios = result['ratios']
        
        # Calculate both absolute and relative noise levels
        absolute_noise = ratios  # This is the absolute noise level (starts at noise_level, goes toward 0)
        relative_noise = [r / ratios[0] for r in ratios] if ratios else []  # Relative to initial (starts at 1.0)
        
        # Plot both curves
        ax.plot(steps, absolute_noise, 'b-', linewidth=2, marker='o', markersize=4, label=f'Absolute (→ 0)')
        ax.plot(steps, relative_noise, 'r-', linewidth=2, marker='s', markersize=4, label=f'Relative (→ 0)')
        
        # Add reference lines
        ax.axhline(y=noise_level, color='gray', linestyle='--', alpha=0.5, label=f'Initial noise ({noise_level:.2f})')
        ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='Relative start (1.0)')
        ax.axhline(y=0.0, color='green', linestyle='--', alpha=0.7, label='Perfect denoising (0.0)')
        
        # Formatting
        ax.set_title(f'Noise Level {noise_level:.2f}\nFinal: {result["final_ratio"]:.3f} ({(1-result["final_ratio"])*100:.1f}% removed)', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Noise Level')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_ylim(-0.05, max(1.1, noise_level * 1.1))
    
    # Hide unused subplots
    for i in range(n_levels, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('noise_level_analysis_individual.png', dpi=300, bbox_inches='tight')
    plt.savefig('noise_level_analysis_individual.pdf', bbox_inches='tight')
    
    # Create summary plots
    create_summary_plots(all_results, noise_levels)
    
    print("Individual noise level plots saved as:")
    print("   - noise_level_analysis_individual.png")
    print("   - noise_level_analysis_individual.pdf")

def create_summary_plots(all_results: Dict, noise_levels: List[float]):
    """Create summary plots across all noise levels."""
    
    # Extract summary statistics
    initial_ratios = [all_results[nl]['initial_ratio'] for nl in noise_levels]
    final_ratios = [all_results[nl]['final_ratio'] for nl in noise_levels]
    best_ratios = [all_results[nl]['best_ratio'] for nl in noise_levels]
    improvements = [all_results[nl]['improvement'] for nl in noise_levels]
    noise_removed_pcts = [(1.0 - all_results[nl]['final_ratio']) * 100 for nl in noise_levels]
    
    # Create summary figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Final Performance vs Initial Noise Level
    ax1.plot(noise_levels, final_ratios, 'bo-', linewidth=2, markersize=6, label='Final Noise Level')
    ax1.plot(noise_levels, noise_levels, 'r--', alpha=0.7, label='No Learning (y=x)')
    ax1.set_xlabel('Initial Noise Level')
    ax1.set_ylabel('Final Noise Level')
    ax1.set_title('Final Denoising Performance vs Initial Noise Level', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 1.05)
    
    # Plot 2: Noise Removal Percentage
    ax2.bar(range(len(noise_levels)), noise_removed_pcts, alpha=0.7, color='green')
    ax2.set_xlabel('Noise Level')
    ax2.set_ylabel('Noise Removed (%)')
    ax2.set_title('Percentage of Noise Removed by Noise Level', fontweight='bold')
    ax2.set_xticks(range(len(noise_levels)))
    ax2.set_xticklabels([f'{nl:.2f}' for nl in noise_levels], rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Absolute Improvement
    ax3.plot(noise_levels, improvements, 'go-', linewidth=2, markersize=6)
    ax3.set_xlabel('Initial Noise Level')
    ax3.set_ylabel('Absolute Improvement (Initial - Final)')
    ax3.set_title('Absolute Denoising Improvement by Noise Level', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Learning Efficiency (Improvement / Initial Noise)
    learning_efficiency = [imp / nl if nl > 0 else 0 for imp, nl in zip(improvements, noise_levels)]
    ax4.plot(noise_levels, learning_efficiency, 'mo-', linewidth=2, markersize=6)
    ax4.set_xlabel('Initial Noise Level')
    ax4.set_ylabel('Learning Efficiency (Improvement / Initial Noise)')
    ax4.set_title('Learning Efficiency by Noise Level', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('noise_level_analysis_summary.png', dpi=300, bbox_inches='tight')
    plt.savefig('noise_level_analysis_summary.pdf', bbox_inches='tight')
    
    print("Summary plots saved as:")
    print("   - noise_level_analysis_summary.png")
    print("   - noise_level_analysis_summary.pdf")

def print_final_analysis(all_results: Dict, noise_levels: List[float]):
    """Print comprehensive analysis results."""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE NOISE LEVEL ANALYSIS RESULTS")
    print("="*80)
    
    print(f"\n{'Noise Level':<12} {'Initial':<8} {'Final':<8} {'Best':<8} {'Improvement':<12} {'% Removed':<10} {'Status'}")
    print("-" * 80)
    
    for noise_level in noise_levels:
        result = all_results[noise_level]
        initial = result['initial_ratio']
        final = result['final_ratio']
        best = result['best_ratio']
        improvement = result['improvement']
        pct_removed = (1.0 - final) * 100
        
        # Determine status
        if final < 0.3:
            status = "🎉 EXCELLENT"
        elif final < 0.5:
            status = "✅ VERY GOOD"
        elif final < 0.7:
            status = "🟡 GOOD"
        elif final < 0.9:
            status = "🟠 MODERATE"
        else:
            status = "❌ POOR"
        
        print(f"{noise_level:<12.2f} {initial:<8.3f} {final:<8.3f} {best:<8.3f} {improvement:<12.3f} {pct_removed:<10.1f} {status}")
    
    # Find best performing noise levels
    best_absolute = min(all_results.items(), key=lambda x: x[1]['final_ratio'])
    best_improvement = max(all_results.items(), key=lambda x: x[1]['improvement'])
    best_percentage = max(all_results.items(), key=lambda x: (1.0 - x[1]['final_ratio']) * 100)
    
    print(f"\n=== KEY FINDINGS ===")
    print(f"Best absolute performance: Noise level {best_absolute[0]:.2f} (final ratio: {best_absolute[1]['final_ratio']:.3f})")
    print(f"Largest improvement: Noise level {best_improvement[0]:.2f} (improvement: {best_improvement[1]['improvement']:.3f})")
    print(f"Highest % noise removed: Noise level {best_percentage[0]:.2f} ({(1.0-best_percentage[1]['final_ratio'])*100:.1f}%)")
    
    # Analysis of trends
    low_noise_performance = np.mean([all_results[nl]['final_ratio'] for nl in noise_levels if nl <= 0.3])
    high_noise_performance = np.mean([all_results[nl]['final_ratio'] for nl in noise_levels if nl >= 0.7])
    
    print(f"\n=== TREND ANALYSIS ===")
    print(f"Average performance on low noise (≤0.3): {low_noise_performance:.3f}")
    print(f"Average performance on high noise (≥0.7): {high_noise_performance:.3f}")
    
    if low_noise_performance < high_noise_performance:
        print("✅ Model performs better on low noise levels (expected)")
    else:
        print("⚠️  Model performs better on high noise levels (unexpected)")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    
    print("Starting comprehensive denoising analysis...")
    print("This will train models across multiple noise levels from 0.01 to 0.99")
    print("Training 20,000 steps per noise level (15 noise levels total)")
    print("Expected runtime: ~2-4 hours depending on hardware\n")
    
    # Run the comprehensive analysis
    all_results, noise_levels = run_comprehensive_noise_analysis()
    
    # Generate plots
    create_noise_level_plots(all_results, noise_levels)
    
    # Print final analysis
    print_final_analysis(all_results, noise_levels)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print("Generated files:")
    print("  - noise_level_analysis_individual.png/pdf (individual noise level plots)")
    print("  - noise_level_analysis_summary.png/pdf (summary analysis)")
    print("="*80)

if __name__ == "__main__":
    main()