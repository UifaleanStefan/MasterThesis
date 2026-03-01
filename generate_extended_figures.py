"""Generate the extended figure suite (Fig 8-15) using synthetic data for demonstration."""
from viz import generate_extended_figures

saved = generate_extended_figures(output_dir="docs/figures")
print(f"\nGenerated {len(saved)} extended figures.")
