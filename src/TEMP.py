from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import openmc_data

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'figure.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.5,
    'lines.linewidth': 2.0,
    'savefig.bbox': 'tight'
})

jeff_directory = Path(__file__).resolve().parent / "JEFF40"
file_path = jeff_directory / "n_82-Pb-206g.jeff"
output_dir = Path(__file__).resolve().parent / "outputs"

pb_data = openmc_data.IncidentNeutron.from_endf(file_path)

incident_energy_ev = 1.41e7

total_xs = pb_data[1].xs(incident_energy_ev)
elastic_xs = pb_data[2].xs(incident_energy_ev)
inelastic_xs = pb_data[4].xs(incident_energy_ev) if 4 in pb_data else 0.0
capture_xs = pb_data[102].xs(incident_energy_ev)
n2n_xs = pb_data[16].xs(incident_energy_ev) if 16 in pb_data else 0.0

weights = {
    'Elastic': elastic_xs / total_xs,
    'Inelastic': inelastic_xs / total_xs,
    'Capture': capture_xs / total_xs,
    '(n,2n)': n2n_xs / total_xs
}

reaction_names = list(weights.keys())
probabilities = list(weights.values())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.bar(reaction_names, probabilities, color='#1f77b4', edgecolor='black')
ax1.set_ylabel('Probability Weighting')
ax1.set_title(f'Reaction Probabilities at {incident_energy_ev/1e6:.1f} MeV')
ax1.set_ylim(0, 1.0)

for i, v in enumerate(probabilities):
    ax1.text(i, v + 0.02, f"{v:.3f}", ha='center', va='bottom', fontsize=10)

energy_grid = np.logspace(4, 7, 500)
elastic_rx = pb_data[2]
outgoing_energies = []
outgoing_weights = []

if elastic_rx.scatter_format == 'angle':
    for mu in np.linspace(-1, 1, 100):
        target_mass_ratio = pb_data.atomic_weight_ratio
        alpha = ((target_mass_ratio - 1) / (target_mass_ratio + 1))**2
        e_out = incident_energy_ev * ((1 + alpha) + (1 - alpha) * mu) / 2
        
        try:
            prob = elastic_rx.angular_distribution(incident_energy_ev, mu)
            outgoing_energies.append(e_out)
            outgoing_weights.append(prob)
        except ValueError:
            continue

if outgoing_energies:
    outgoing_energies = np.array(outgoing_energies) / 1e6
    outgoing_weights = np.array(outgoing_weights) / np.sum(outgoing_weights) 
    
    ax2.plot(outgoing_energies, outgoing_weights, color='#d62728')
    ax2.set_xlabel('Outgoing Neutron Energy (MeV)')
    ax2.set_ylabel('Relative Probability')
    ax2.set_title('Elastic Scattering Energy Spread')
    ax2.fill_between(outgoing_energies, outgoing_weights, alpha=0.3, color='#d62728')

plt.tight_layout()
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / 'neutron_energy_profile.png')
plt.show()

print(f"Total Macroscopic Attenuation Factor (Microscopic Total XS): {total_xs:.3f} barns")