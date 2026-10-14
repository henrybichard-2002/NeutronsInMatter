import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# --- Configuration ---
project_dir = Path(__file__).resolve().parent
file_path = project_dir / 'H20(liq)_nXsection_data.json'
output_dir = project_dir / 'outputs'
incident_energies_mev = [1.0, 2.45, 10.0, 13.8, 14.1, 14.5, 17.0]
max_thickness_mm = 20
energy_cutoff_mev = 1.0e-9  
xi_parameter = 0.920  #H20: 0.920, C-12: 0.158, dry-air: 0.133, aluminium: 0.072
n0_particles = 10000

def load_and_interpolate_xs(filepath, target_energies):
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    material_name = "Unknown Material"
    total_energy, total_xs = None, None
    elastic_energy, elastic_xs = None, None
    
    for item in data:
        if 'material' in item:
            material_name = item['material']
        elif 'nuclide' in item:
            material_name = item['nuclide']
            
        reaction = item.get('reaction', '')
        mt = item.get('mt')
        
        # Extract and convert values: eV to MeV, and 1/cm to 1/mm
        if reaction == 'MT 1' or mt == 1 or reaction == '(n,total)':
            total_energy = np.array(item['energy_eV']) / 1e6
            total_xs = np.array(item.get('xs_per_cm', item.get('xs'))) / 10.0
        elif reaction == 'MT 2' or mt == 2 or reaction == '(n,elastic)':
            elastic_energy = np.array(item['energy_eV']) / 1e6
            elastic_xs = np.array(item.get('xs_per_cm', item.get('xs'))) / 10.0

    f_total = interp1d(total_energy, total_xs, bounds_error=False, fill_value=0.0)
    f_elastic = interp1d(elastic_energy, elastic_xs, bounds_error=False, fill_value=0.0)
    
    st_targets = f_total(target_energies)
    ss_targets = f_elastic(target_energies)
    
    return material_name, f_total, f_elastic, st_targets, ss_targets

def print_data_table(material_name, energies, sig_t, sig_s):
    print(f"Material: {material_name}")
    print(f"{'Energy (MeV)':<15} | {'Total XS (1/mm)':<18} | {'Elastic XS (1/mm)':<18}")
    print("-" * 56)
    for e, st, ss in zip(energies, sig_t, sig_s):
        print(f"{e:<15.2f} | {st:<18.6f} | {ss:<18.6f}")
    print("\n")

def run_neutron_analysis():
    material, f_sig_t, f_sig_s, st_vals, ss_vals = load_and_interpolate_xs(
        file_path, incident_energies_mev
    )
    print_data_table(material, incident_energies_mev, st_vals, ss_vals)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(incident_energies_mev)))

    x_att = np.linspace(0, max_thickness_mm, 5000)
    
    for i, e_target in enumerate(incident_energies_mev):
        sigma_t = st_vals[i]
        
        attenuation = np.exp(-sigma_t * x_att)
        std_att = np.sqrt(attenuation * (1.0 - attenuation) / n0_particles)
        
        label_str = f"{e_target:.1f} MeV"
        ax1.plot(x_att, attenuation, label=label_str, color=colors[i], linewidth=2.5)
        ax1.fill_between(x_att, 
                         np.maximum(0, attenuation - std_att), 
                         np.minimum(1, attenuation + std_att), 
                         color=colors[i], alpha=0.3)

    ax1.set_xlabel('Thickness (mm)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Uncollided Flux Fraction', fontsize=14, fontweight='bold')
    ax1.set_title(f'Neutron Attenuation in {material}', fontsize=16, fontweight='bold')
    ax1.tick_params(axis='both', labelsize=12)
    ax1.legend(fontsize=11, frameon=True, shadow=True)
    ax1.set_xlim(0, max_thickness_mm)
    #ax1.set_ylim(0, 1.05)
    
    ax1.minorticks_on()
    ax1.grid(which='major', linestyle='-', linewidth=0.8, color='black', alpha=0.2)
    ax1.grid(which='minor', linestyle=':', linewidth=0.5, color='gray', alpha=0.4)

    for i, e0 in enumerate(incident_energies_mev):
        if e0 <= energy_cutoff_mev:
            continue
            
        e_grid = np.logspace(np.log10(e0), np.log10(energy_cutoff_mev), 50000)
        sig_s_interp = f_sig_s(e_grid)
        sig_s_safe = np.where(sig_s_interp == 0, 1e-10, sig_s_interp)
        
        integrand_x = 1.0 / (e_grid * xi_parameter * sig_s_safe)
        integrand_var_x = 1.0 / (e_grid * xi_parameter * sig_s_safe**2)
        
        dE = np.abs(np.diff(e_grid))
        
        avg_integrand_x = 0.5 * (integrand_x[:-1] + integrand_x[1:])
        x_loss = np.cumsum(dE * avg_integrand_x)
        x_loss = np.insert(x_loss, 0, 0.0)
        
        avg_integrand_var_x = 0.5 * (integrand_var_x[:-1] + integrand_var_x[1:])
        var_x = np.cumsum(dE * avg_integrand_var_x)
        var_x = np.insert(var_x, 0, 0.0)
        
        std_x = np.sqrt(var_x)
        dE_dx = e_grid * xi_parameter * sig_s_safe
        std_E = std_x * dE_dx
        
        energy_loss = e0 - e_grid

        if max_thickness_mm <= x_loss[-1]:
            e_loss_end = np.interp(max_thickness_mm, x_loss, energy_loss)
            std_e_end = np.interp(max_thickness_mm, x_loss, std_E)
        else:
            e_loss_end = energy_loss[-1]
            std_e_end = std_E[-1]

        e_end = e0 - e_loss_end
        upper_std = std_e_end
        lower_std = min(e_end, std_e_end) 

        # Dynamically scale precision based on the magnitude of the energy loss
        # This prevents rounding output to 0.00 when over very thin geometries
        decimals = max(2, int(np.ceil(-np.log10(max(e_loss_end, 1e-10)))) + 1)

        label_str = f"Start: {e0:.{decimals}f} | End: {e_end:.{decimals}f} (+{upper_std:.2e}/-{lower_std:.2e}) MeV"
        
        # Clip arrays at 1e-7 to gracefully handle log scale 0s
        y_loss = np.maximum(energy_loss, 1e-7)
        y_lower = np.maximum(energy_loss - std_E, 1e-7)
        y_upper = energy_loss + std_E
        
        ax2.plot(x_loss, y_loss, label=label_str, color=colors[i], linewidth=2.5)
        ax2.fill_between(x_loss, y_lower, y_upper, color=colors[i], alpha=0.3)
                         
        x_coord = min(max_thickness_mm, x_loss[-1])
        y_coord = max(e_loss_end, 1e-7)
        ax2.plot(x_coord, y_coord, 'o', color=colors[i], markersize=6)
        
        annotation_text = f"E_end: {e_end:.{decimals}f} MeV\n(+{upper_std:.2e} / -{lower_std:.2e})"
        ax2.annotate(annotation_text, 
                     xy=(x_coord, y_coord),
                     xytext=(-10, 15), textcoords='offset points',
                     color=colors[i], fontsize=9, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=colors[i], alpha=0.8),
                     horizontalalignment='right')

    ax2.set_xlabel('Penetration Distance (mm)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Energy Loss (MeV)', fontsize=14, fontweight='bold')
    ax2.set_title('Energy Loss via CSDA', fontsize=16, fontweight='bold')
    ax2.tick_params(axis='both', labelsize=12)
    ax2.legend(fontsize=9, frameon=True, shadow=True, loc='upper left')
    ax2.set_xlim(0, max_thickness_mm)
    ax2.set_yscale('log')
    ax2.set_ylim(bottom=1e-7)
    
    ax2.minorticks_on()
    ax2.grid(which='major', linestyle='-', linewidth=0.8, color='black', alpha=0.2)
    ax2.grid(which='minor', linestyle=':', linewidth=0.5, color='gray', alpha=0.4)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'neutron_analysis_results.png', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    run_neutron_analysis()