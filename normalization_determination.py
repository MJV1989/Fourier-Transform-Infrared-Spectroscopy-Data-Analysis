# File name: normalization_determination.py
# Prompt Engineer and Programmer: Mikko Valkonen, M.Sc. (Tech.)
# Used artificial intelligence tools for code generation: Aalto AI Assistant, GitHub Copilot, and
# Amazon Q.
# Affiliation: Aalto University School of Chemical Engineering, T107 Department of Bioproducts and
# Biosystems, T10400 Wood Material Science
# Date: 27.9.2025
# Software: Visual Studio Code 1.104.2; Python 3.13.7
# Hardware: MacBook Pro (14-inch, Apple M3, Nov 2023); macOS Sequoia 15.7

"""
This script performs normalization of baseline-corrected transmission-mode Fourier-transform
infrared spectral data from a comma-separated value (CSV) file. The normalization is based on the
absorbance value at a specific target wavenumber (ca. 2890 cm⁻¹ of the CH/CH₂ stretching band). The
normalized data is saved to a new CSV file, and the original and normalized spectra are visualised
for quality control.

Functions:
    - Reads spectral data from a CSV file.
    - Identifies the absorbance value at a target wavenumber with a specified tolerance.
    - Normalizes the absorbance values by dividing them by the absorbance value at the target
    wavenumber.
    - Saves the normalized data to a new CSV file.
    - Plots the original and normalized absorbance spectra.

Dependencies:
    - Panel Data or Python Data Analysis (Pandas) Python library as pandas with the pd designation:
        - For data manipulation and analysis.
    - Numerical Python (NumPy) Python library as numpy with the np designation:
        - For numerical operations.
    - Matplotlib [portmanteau of MATLAB (MATrix LABoratory), plot, and library] Python plotting
    library with the Pyplot (Py = Python) module as matplotlib.pyplot with the plt designation:
        - For plotting graphs.

Usage:
    1. Update the `file_path` variable with the path to the input CSV file.
    2. Ensure the input CSV file contains columns named 'cm-1' (wavenumbers) and 
    'Absorbance_Corrected' (absorbance values).
    3. Modify the `output_file_path` variable to specify the desired output file path.
    4. Run the script to generate the normalized data and plots.

Input:
    - A CSV file containing baseline-corrected spectral data with columns:
        - 'cm-1': Wavenumbers (in cm⁻¹).
        - 'Absorbance_Corrected': Absorbance values.

Output:
    - A new CSV file with an additional column:
        - 'Absorbance_Normalized': Normalized absorbance values.

Visualisation:
    - A plot comparing the original and normalized absorbance spectra.
"""

# Importing necessary libraries:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the baseline-corrected spectral data from a CSV file:
FILE_PATH = 'heavy_water_0_final_edited_RubberBand.csv'  # Change this to the actual file path.
data = pd.read_csv(FILE_PATH)

# Assuming the column headers are 'cm-1' for wavenumbers and 'Absorbance_Corrected' for absorbance:
wavenumbers = data['cm-1']
absorbances = data['Absorbance_Corrected']

# Find the absorbance value at approximately 2890 cm⁻¹:
TARGET_WAVENUMBER = 2890
TOLERANCE = 1  # Set a tolerance level to match the target wavenumber approximately.

# Locate the index of the wavenumber closest to the target wavenumber:
closest_idx = np.abs(wavenumbers - TARGET_WAVENUMBER).idxmin()
absor_value_at_target = absorbances[closest_idx]

# Normalize the absorbance values by the absorbance value at the target wavenumber:
data['Absorbance_Normalized'] = absorbances / absor_value_at_target

# Save the normalized data to the same CSV file, adding the normalized data to the next column:
OUTPUT_FILE_PATH = 'heavy_water_0_final_edited_RubberBand_Normalized.csv'  # Change this to the
# desired output file path.
data.to_csv(OUTPUT_FILE_PATH, index = False)

# If need to print the resulting data for verification:
print(data.head())

# Plot the original and normalized absorbance spectra:
plt.figure(figsize = (10, 5))
plt.plot(wavenumbers, absorbances, label = 'Original Rubber-Band-Baselined Absorbance')
plt.plot(wavenumbers, data['Absorbance_Normalized'], label = 'Normalized Absorbance')
plt.xlabel(r'$\bar{\nu}$ (cm$^{-1}$)')
plt.gca().invert_xaxis()
plt.ylabel(r'$A_{\mathrm{corr,norm}}$ (–)')
plt.legend()
plt.show()
