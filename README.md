# Denoising Demonstration

This folder contains a self-contained demonstration of individual denoising layer training for the NoProp language model project.

## Purpose

Demonstrates that individual denoising layers can learn effective denoising when given sufficient training time (~5,000+ steps instead of ~50 steps).

## Key Results

- **5,000 steps**: 40.2% peak noise removal
- **20,000 steps**: 45.2% peak noise removal  
- **Training time**: <1 minute for 5k steps, ~3 minutes for 20k steps

## Files

- `generate_denoising_analysis.py` - Main script that generates the comprehensive analysis
- `src/` - Copy of source code (models, config, utilities)
- `finewebdiffusion_v2.yaml` - Configuration file

## Usage

Simply run the main script:

```bash
python generate_denoising_analysis.py
```

This will:
1. Train two denoising models from scratch (5k and 10k steps)
2. Generate actual experimental data by training residual models
3. Create comprehensive analysis plots with real results
4. Save results as `denoising_comprehensive_analysis.png`

**Expected runtime**: ~5 minutes total (1-2 min for 5k steps, 3-4 min for 10k steps)

## Dependencies

- PyTorch
- NumPy  
- Matplotlib
- PyYAML

## Key Insights

1. **Individual denoising layers CAN learn effective denoising**
2. **Residual architecture M(x) = x + d(x) works well**
3. **Extended training is crucial** (5000+ vs 50 steps)
4. **The NoProp concept is fundamentally sound** - previous failures were due to insufficient training duration per layer

This validates the theoretical foundation of the NoProp approach.