# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 15:13:08 2026

@author: HenryBichard
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# --- Configuration ---
project_dir = Path(__file__).resolve().parent
file_path = project_dir / 'air(dry)_nXsection_data.json'
output_dir = project_dir / 'outputs'
# Add the energies (in eV) where you want vertical lines and data labels
highlight_energies_ev = [0.0253, 2.45e6, 1.4e7]
minE = 1e5
maxE = 2e7
log_x_scale = False  # Set to False to use a linear x-axis scale

# Standard ENDF MT descriptions for the legend
MT_NAMES = {
    1: "Total", 2: "Elastic", 3: "Non-elastic", 4: "Inelastic",
    16: "(n,2n)", 18: "Fission", 22: "(n,na)", 51: "(n,n1)",
    52: "(n,n2)", 53: "(n,n3)", 54: "(n,n4)", 55: "(n,n5)",
    56: "(n,n6)", 57: "(n,n7)", 58: "(n,n8)", 59: "(n,n9)",
    60: "(n,n10)", 61: "(n,n11)", 62: "(n,n12)", 91: "(n,nc)",
    101: "Neutron Absorption", 102: "Radiative Capture (n,gamma)", 
    103: "(n,p)", 104: "(n,d)", 105: "(n,t)", 106: "(n,3He)", 
    107: "(n,alpha)", 203: "(n,Xp)", 204: "(n,Xd)", 205: "(n,Xt)", 
    206: "(n,X3He)", 207: "(n,Xalpha)", 301: "KERMA", 444: "Damage", 
    901: "DPA"
}

def plot_all_cross_sections(filepath, target_energies=None, min_e=1, max_e=1e9, use_log_x=True):
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    material_name = "Unknown Material"
    if len(data) > 0:
        material_name = data[0].get('material', data[0].get('nuclide', 'Unknown Material'))

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    
    plotted_reactions = {}
    
    for item in data:
        reaction_str = item.get('reaction', '')
        mt = item.get('mt')
        
        # Attempt to parse MT if it's not explicitly given as an integer
        if mt is None:
            if reaction_str.startswith("MT "):
                try:
                    mt = int(reaction_str.split(" ")[1])
                except ValueError:
                    pass
            elif item.get('reactionId', '').isdigit():
                mt = int(item.get('reactionId'))
        
        # Build a descriptive label
        if mt in MT_NAMES:
            label = f"{MT_NAMES[mt]} (MT {mt})"
        elif mt is not None:
            label = f"{reaction_str} (MT {mt})" if reaction_str else f"MT {mt}"
        else:
            label = reaction_str if reaction_str else "Unknown Reaction"
            
        energies = np.array(item['energy_eV'])
        xs_vals = np.array(item.get('xs_per_cm', item.get('xs', [])))
        
        if len(energies) == 0 or len(xs_vals) == 0:
            continue
            
        line, = ax.plot(energies, xs_vals, linewidth=2.0, label=label)
        
        f_interp = interp1d(energies, xs_vals, bounds_error=False, fill_value=0.0)
        plotted_reactions[label] = {
            "interp": f_interp, 
            "color": line.get_color()
        }

    y_axis_label = f"Cross Section ({data[0].get('units', 'Unknown Units')})" if len(data) > 0 else "Cross Section"

    if target_energies:
        for e_target in target_energies:
            # Only annotate if the target energy is within our axis limits
            if not (min_e <= e_target <= max_e):
                continue
                
            ax.axvline(x=e_target, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
            
            for label, info in plotted_reactions.items():
                xs_at_e = float(info["interp"](e_target))
                
                if xs_at_e > 0:
                    ax.plot(e_target, xs_at_e, 'o', color=info["color"], markersize=6)
                    
                    # Offset the text slightly to the right to avoid overlapping the marker
                    ax.annotate(f'{xs_at_e:.2e}', 
                                xy=(e_target, xs_at_e),
                                xytext=(8, 0), textcoords='offset points',
                                color=info["color"], fontsize=10, fontweight='bold',
                                verticalalignment='center',
                                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
                    
            ax.annotate(f'{e_target:.1e} eV', 
                        xy=(e_target, ax.get_ylim()[1]), 
                        xytext=(0, 5), textcoords='offset points',
                        ha='center', va='bottom', fontsize=11, fontweight='bold', color='gray')

    # Apply axis scaling
    if use_log_x:
        ax.set_xscale('log')
    else:
        ax.set_xscale('linear')
    ax.set_yscale('log')

    ax.set_xlabel('Neutron Energy (eV)', fontsize=14, fontweight='bold')
    ax.set_ylabel(y_axis_label, fontsize=14, fontweight='bold')
    ax.set_title(f'Neutron Cross Sections for {material_name}', fontsize=16, fontweight='bold')
    
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth=0.8, color='black', alpha=0.2)
    ax.grid(which='minor', linestyle=':', linewidth=0.5, color='gray', alpha=0.4)
    
    ax.legend(fontsize=11, frameon=True, shadow=True, bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.set_xlim(min_e, max_e)
    ax.set_ylim(bottom=1e-6)
    
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'all_cross_sections_plot.png', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_all_cross_sections(file_path, highlight_energies_ev, minE, maxE, log_x_scale)