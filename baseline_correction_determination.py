# File name: baseline_correction_determination.py
# Prompt Engineer and Programmer: Mikko Valkonen, M.Sc. (Tech.)
# Used artificial intelligence tools for code generation: Aalto AI Assistant, GitHub Copilot,
# Amazon Q, and Microsoft Copilot.
# Affiliation: Aalto University School of Chemical Engineering, T107 Department of Bioproducts and
# Biosystems, T10400 Wood Material Science
# Date: 27.9.2025
# Software: Visual Studio Code 1.104.2; Python 3.13.7
# Hardware: MacBook Pro (14-inch, Apple M3, Nov 2023); macOS Sequoia 15.7

"""
Baseline correction determination script

This script is used to determine the baseline (BL) correction for the imported transmittance data
of experimentally obtained transmission-mode Fourier-transform infrared spectroscopy data per
wavenumber for thin wood microveneers (MVs) and blank measurements inside a relative humidity cell.
The script ensures that the data arrays are of equal size. The transmittance data is first converted
to absorbance data. The script then compares the used BL correction techniques to the BL correction
method used by the PerkinElmer® Spectrum IR (Application Version: 10.6.2.1159) software. The
optimal parameters for each BL correction method are determined by minimizing the root mean square
error (RMSE) between the converted and corrected absorbance data and the given corrected absorbance
data. Visual inspection of the results is also performed to ensure the accuracy of the BL
correction. For the visualisation, all optimal solutions are plotted as separate figures.

The script uses the following BL correction methods for the comparison:
    - Rubber-band (RB) correction.
    - Polynomial fitting.
    - Asymmetric least squares (AsLS) smoothing.
    - Savitzky-Golay (SG) filtering.
    - Whittaker smoothing.

Modules:
    - Mathematical functions (math) Python library as math:
        - For mathematical operations.
    - Operating system (OS) Python library as os:
        - For interacting with the operating system by checking the current working directory.
    - Dataclasses (dataclasses) Python library as dataclass:
        - For creating data classes.
    - Numerical Python (NumPy) Python library as numpy with the np designation:
        - For numerical operations.
    - Panel Data or Python Data Analysis (Pandas) Python library as pandas with the pd designation:
        - For data manipulation and analysis.
    - Matplotlib [portmanteau of MATLAB (MATrix LABoratory), plot, and library] Python plotting
    library with the Pyplot (Py = Python) module as matplotlib.pyplot with the plt designation:
        - For plotting graphs.
    - Scientific Python (SciPy) Python library as scipy:
        - For scientific computations including linear algebra, integration, optimisation, and
        sparse matrix operations (sparse) and signal processing.
    - Importing the sparse linear algebra module from the SciPy library as scipy.sparse.linalg:
        - For sparse linear algebra operations with the spsolve (sparse solve) function. The spsolve
        function is used for solving sparse linear systems (Ax = b), where A is a sparse matrix, x
        is the unknown vector to be found (solution), and b may be a vector or matrix.
    - Importing the SG filter from the Signal Processing module of the SciPy library (scipy.signal)
    with savgol_filter:
        - For applying SG filtering.
    - Importing the Cholesky decomposition module from the linear algebra module of the SciPy
    library (scipy.linalg) as cho_factor and cho_solve:
        - For solving linear systems of equations using the Cholesky decomposition method.
    - SciPy Toolkit learn (Scikit-learn) machine learning Python library with the Metrics module as
    sklearn.metrics:
        - For calculating RMSE by using the mean_squared_error function.
    - NumPy with the Polynomial module as numpy.polynomial:
        - For polynomial fitting.

Functions and comparison methods:
    - load_microveneer_data(csv_file):
        - Loads MV spectral data from a comma-separated value (CSV) file generically.
    - load_blank_data(blank_csv_file):
        - Loads blank spectral data from a CSV file generically.
    - Load data for MVs and blank measurements.
        - Loads the relevant spectral data from a real CSV file.
    - assert len(wavenumbers):
        - Ensures that the lengths (len) of the wavenumbers and transmittance and corrected given
        absorbance arrays are the same for the MVs.
    - assert len(blank_wavenumbers):
        - Ensures that the lengths of the wavenumbers and transmittance and corrected given
        absorbance arrays are the same for the blank measurements.
    - transmittance_to_absorbance(trans):
        - Converts measured transmittance percentage to absorbance in arbitrary units.
    - Subtract the blank absorbance from the MV absorbance for the MVs:
        - Subtracts the blank absorbance from the experimental MV absorbance and the given corrected
        absorbance.
    - Dataclass for plotting the results of BL correction:
        - Contains the necessary parameters for plotting the results.
    - plot_results(fig_num, wavenumbers_local, absorbance_raw_local, baseline,
    absorbance_corrected_local, absorbance_corrected_given_local, method, title, rmse_value):
        - Plots the results of BL correction.
    - rubber_band_baseline_correction(y):
        - Performs RB BL correction.
    - polynomial_fit_baseline(wavenumbers_local, absorbance_raw_local, degree):
        - Performs polynomial fitting for BL correction.
    - baseline_als(y, lam, p, niter = 10):
        - Performs AsLS smoothing for BL correction.
    - savitzky_golay_baseline(absorbance_local, window_length, polyorder):
        - Performs SG filtering for BL correction.
    - whittaker_smooth(y, w, lambda_, differences):
        - Performs Whittaker smoothing for BL correction.
    - save_to_csv(wavenumbers_local, absorbance_corrected, original_filename, method_name):
        - Saves the corrected absorbance data to a CSV file.
    - optimal_rubber_band():
        - Finds the optimal parameters for RB BL correction, saves the results to a CSV file, and
        plots the results.
    - optimal_polynomial_fit():
        - Finds the optimal polynomial degree for BL correction, saves the results to a CSV file,
        and plots the results.
    - optimal_asls():
        - Finds the optimal parameters for AsLS BL correction, saves the results to a CSV file, and
        plots the results.
    - optimal_savitzky_golay():
        - Finds the optimal parameters for SG BL correction, saves the results to a CSV file, and
        plots the results.
    - optimal_whittaker():
        - Finds the optimal lambda for Whittaker BL correction, saves the results to a CSV file, and
        plots the results.
    - plt.show():
        - Pauses the script to prevent the closing of all figures when the script finishes.

Usage:
    - Import the necessary libraries.
    - Check the current working directory to ensure the correct path for the produced CSV files.
    - Define the load_data functions to load the MV and blank spectral data from a corresponding CSV
    file, respectively.
    - Ensure the lengths of the wavenumbers and transmittance arrays are the same.
    - Convert transmittance percentage to absorbance.
    - Convert raw transmittances to absorbances.
    - Subtract the blank absorbance from the MV absorbance.
    - Find optimal parameters for each BL correction method and plot the results.
"""

# Importing necessary libraries:
import math
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter
from scipy.linalg import cho_factor, cho_solve
from sklearn.metrics import mean_squared_error
from numpy.polynomial import Polynomial

# Check the current working directory by printing it:
print("Current working directory:", os.getcwd())

# Function to load MV spectral data from a CSV file:
def load_microveneer_data(csv_file):
    """
    Load MV spectral data from a CSV file.

    Parameters:
        - csv_file [strings of characters (str) of the path]: The path to the MV CSV file.

    Returns:
        - wavenumbers_local (np.ndarray): The wavenumbers local to the function.
        - transmittance_raw_local (np.ndarray): The raw transmittance data of the MVs local
        to the function.
        - absorbance_corrected_given_local (np.ndarray): The corrected given absorbance data of
        the MVs local to the function.
    """
    data = pd.read_csv(csv_file)
    wavenumbers_local = data['cm-1'].to_numpy()
    transmittance_raw_local = data['%T'].to_numpy()
    absorbance_corrected_given_local = data['Absorbance_Corrected_Given'].to_numpy()
    return wavenumbers_local, transmittance_raw_local, absorbance_corrected_given_local

# Function to load blank spectral data from a CSV file:
def load_blank_data(blank_csv_file):
    """
    Load blank spectral data from a CSV file.

    Parameters:
        - blank_csv_file (str of the path): The path to the blanks CSV file.

    Returns:
        - blank_wavenumbers_local (np.ndarray): The blank wavenumbers local to the function.
        - transmittance_raw_blank_local (np.ndarray): The raw transmittance data of the blanks local
        to the function.
        - absorbance_corrected_given_blank_local (np.ndarray): The corrected absorbance data of
        the blanks local to the function.
    """
    data = pd.read_csv(blank_csv_file)
    blank_wavenumbers_local = data['cm-1'].to_numpy()
    transmittance_raw_blank_local = data['%T'].to_numpy()
    absorbance_corrected_given_blank_local = data['Absorbance_Corrected_Given'].to_numpy()
    return blank_wavenumbers_local, transmittance_raw_blank_local, \
        absorbance_corrected_given_blank_local

# Load data for MVs and blank measurements:
wavenumbers, transmittance_raw, absorbance_corrected_given = \
    load_microveneer_data('light_water_0_edited.csv')
blank_wavenumbers, transmittance_raw_blank, absorbance_corrected_given_blank = \
    load_blank_data('light_water_empty_0_edited.csv')

# Ensure the lengths of the wavenumbers and transmittance and corrected given absorbance arrays are
# the same for the MVs and blank measurements, respectively:
assert len(wavenumbers) == len(transmittance_raw) == len(absorbance_corrected_given)
assert len(blank_wavenumbers) == len(transmittance_raw_blank) == \
    len(absorbance_corrected_given_blank)

# Convert transmittance percentage to absorbance:
# Formula: Absorbance = 2 - log_10(Transmittance-%).
def transmittance_to_absorbance(trans):
    """
    Convert transmittance percentage to absorbance in arbitrary units.

    Parameters:
        - Constant [integer (int)]: Two (2) for the absorbance calculation.
        - Conversion function (np.log10): The base-10 logarithm function for the absorbance
        calculation.
        - Use the above two parameters to calculate the absorbance as per the formula further above.
    Returns:
        - 2 - np.log10(trans): The calculated absorbance.
    """
    return 2 - np.log10(trans)

# Convert raw and corrected transmittance to absorbance for the MVs and blank measurements:
absorbance_raw = transmittance_to_absorbance(transmittance_raw)
absorbance_raw_blank = transmittance_to_absorbance(transmittance_raw_blank)

# Subtract the blank absorbance from the MV absorbance for the raw and corrected given absorbance:
absorbance_raw -= absorbance_raw_blank
absorbance_corrected_given -= absorbance_corrected_given_blank

@dataclass
class PlotData:
    """
    Dataclass for plotting the results of the BL corrections.

    Attributes:
        - fig_num (int): Figure number.
        - wavenumbers_local (np.ndarray): Wavenumbers local to the function.
        - absorbance_raw_local (np.ndarray): Raw absorbance local to the function.
        - baseline (np.ndarray): Baseline.
        - absorbance_corrected_local (np.ndarray): Corrected absorbance local to the function.
        - absorbance_corrected_given_local (np.ndarray): Corrected given absorbance local to the
        function.
        - rmse_value (float): Root mean square error.
        - title (str): Title of the plot.
        - method (str): Method used for BL correction.
    """
    fig_num: int
    wavenumbers_local: np.ndarray
    absorbance_raw_local: np.ndarray
    baseline: np.ndarray
    absorbance_corrected_local: np.ndarray
    absorbance_corrected_given_local: np.ndarray
    rmse_value: float
    title: str
    method: str

# Function to plot the results:
def plot_results(data: PlotData):
    """
    Plot the results of the BL correction.

    Parameters:
        - data (PlotData): Dataclass containing all necessary parameters for plotting.
        - plt.figure: Create a new figure with the specified figure number and size.
        - plt.plot: Plot the raw absorbance data, baseline, corrected absorbance data, and given
        corrected absorbance data as functions of wavenumber local to the function:
            - label: Set the label of the plot.
            - color: Set the colour of the plot.
            - linestyle: Set the line style of the plot.
        - plt.xlabel: Set the x-axis label.
        - plt.ylabel: Set the y-axis label.
        - plt.title: Set the title of the plot.
        - plt.gca().invert_xaxis(): Invert the x-axis.
        - plt.legend: Display the legend.
        - plt.text: Display the RMSE value in the plot:
            - transform: Set the transformation of the text.
            - fontsize: Set the font size of the text.
            - verticalalignment: Set the vertical alignment of the text.
            - bbox: Set the bounding box properties of the text.
        - plt.show(block = False): Display the plot without blocking the script:
            - block: Set to False Boolean (bool) value to prevent the script from blocking the
            display of the plot.
        - plt.show: Display the plot.
    """
    plt.figure(data.fig_num, figsize = (10, 6))
    plt.plot(data.wavenumbers_local, data.absorbance_raw_local, label = 'Non-Baseline Corrected', \
             color = 'blue')
    plt.plot(data.wavenumbers_local, data.baseline, label = f'Baseline ({data.method})', \
             color = 'green', linestyle = '--')
    plt.plot(data.wavenumbers_local, data.absorbance_corrected_local, label = f'{data.method} \
             Corrected', color = 'orange')
    plt.plot(data.wavenumbers_local, data.absorbance_corrected_given_local, label = \
             'Given Baseline Corrected', color = 'red', linestyle = ':')
    plt.xlabel(r'$\bar{\nu}$ (cm$^{-1}$)')
    plt.ylabel(r'$A_{\mathrm{corr}}$ (a.u.)')
    plt.title(f'Baseline Correction Using {data.title}')
    plt.gca().invert_xaxis()
    plt.legend()
    plt.text(0.05, 0.95, f'RMSE: {data.rmse_value:.3f}', transform = plt.gca().transAxes, fontsize \
             = 12, verticalalignment = 'top', bbox = dict(boxstyle = "round,pad = 0.3", \
                                                          edgecolor = "black", facecolor = "white"))
    plt.show(block = False)

# The RB BL correction function:
def rubber_band_baseline_correction(y):
    """
    Rubber-band BL correction algorithm.

    Parameters:
        - y (array-like): The input absorbance data.
        - v (array-like): Boolean array indicating vertices.
        - dtype (type): Data type of the array.
        - i (int): Index for the loop.
        - left (int): Index of the left vertex.
        - right (int): Index of the right vertex.
        - baseline_value (float): Value of the BL.
        - baseline (array-like): The estimated BL.
        - corrected_local (array-like): The BL-corrected data.

    Returns:
        - baseline (array-like): The estimated BL.
        - corrected (array-like): The BL-corrected data.
    """
    v = np.zeros_like(y, dtype = bool)  # Boolean array indicating vertices.
    v[0] = v[-1] = True  # Start with the first and last points as vertices.

    for i in range(1, len(y) - 1):
        # Linear interpolation between two adjacent vertices.
        left = np.where(v[:i])[0][-1]
        right = np.where(v[i:])[0][0] + i
        baseline_value = y[left] + (y[right] - y[left]) * (i - left) / (right - left)
        if y[i] < baseline_value:
            # Mark this point as a vertex.
            v[i] = True

    baseline = np.interp(np.arange(len(y)), np.where(v)[0], y[v])
    corrected_local = y - baseline
    return baseline, corrected_local

# The polynomial fitting BL correction function:
def polynomial_fit_baseline(wavenumbers_local, absorbance_raw_local, degree):
    """
    Polynomial fitting for BL correction.

    Parameters:
        - wavenumbers_local (np.ndarray): The local wavenumbers array.
        - absorbance_raw_local (np.ndarray): The local raw absorbance data.
        - degree (int): The degree of the polynomial.

    Returns:
        - baseline (np.ndarray): The estimated BL.
        - absorbance_corrected_local (np.ndarray): The BL-corrected data.
    """
    p = Polynomial.fit(wavenumbers_local, absorbance_raw_local, degree)
    baseline = p(wavenumbers_local)
    absorbance_corrected_local = absorbance_raw_local - baseline
    return baseline, absorbance_corrected_local

# The AsLS smoothing BL correction function:
def baseline_als(y, lam, p, niter = 10):
    """
    Asymmetric Least Squares smoothing for BL correction fitting.

    Parameters:
        - y (array-like): The input absorbance data.
        - lam (float): The smoothing parameter, lam is short for lambda.
        - p (float): The asymmetric parameter.
        - niter (int, optional): The number of iterations. Defaults to 10.
        - L (int): The length of the input array.
        - D (sparse matrix): The difference matrix.
        - w (array-like): The weights.
        - _ (int): The loop index.
        - W (sparse matrix): The diagonal matrix of the weights.
        - Z (sparse matrix): The matrix for the least squares problem.
        - z (array-like): The estimated BL.

    Returns:
        z (array-like): The estimated BL.
    """
    L = len(y) # pylint: disable=invalid-name
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape = (L - 2, L)) # pylint: disable=invalid-name
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L) # pylint: disable=invalid-name
        Z = W + lam * D.T @ D # pylint: disable=invalid-name
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z

# The SG filtering BL correction function:
def savitzky_golay_baseline(absorbance_local, window_length, polyorder):
    """
    Savitzky-Golay filtering for BL correction.

    Parameters:
        - baseline (array-like): The estimated BL.
        - absorbance (array-like): The input absorbance data local to the function.
        - window_length (int): The length of the window for the Savitzky-Golay filter.
        - polyorder (int): The order of the polynomial for the Savitzky-Golay filter.
        - absorbance_corrected_local (array-like): The BL-corrected data.

    Returns:
        - baseline (array-like): The estimated BL.
        - absorbance_corrected (array-like): The BL-corrected data.
    """
    baseline = savgol_filter(absorbance_local, window_length, polyorder)
    absorbance_corrected_local = absorbance_local - baseline
    return baseline, absorbance_corrected_local

# The Whittaker smoothing BL correction function:
def whittaker_smooth(y, w, lambda_, differences):
    """
    Whittaker smoothing for BL correction.

    Parameters:
        - y (array-like): The input absorbance data.
        - X (array-like): The identity matrix.
        - _ (int): The loop index.
        - differences (int): The number of differences for the Whittaker smoothing.
        - W (array-like): The diagonal matrix of the weights.
        - w (array-like): The weights for the Whittaker smoothing.
        - Z (array-like): The matrix for the least squares problem.
        - lambda_ (float): The smoothing parameter.
        - T (array-like): The transpose of the identity matrix.
        - z (array-like): The estimated BL.
        - cho_solve (function): The Cholesky decomposition solver.
        - cho_factor (function): The Cholesky decomposition function.

    Returns:
        z (array-like): The estimated BL.
    """
    y = np.asarray(y)
    X = np.eye(len(y)) # pylint: disable=invalid-name
    for _ in range(differences):
        X = np.diff(X, axis = 0) # pylint: disable=invalid-name
    W = np.diag(w) # pylint: disable=invalid-name
    Z = W + lambda_ * X.T.dot(X) # pylint: disable=invalid-name
    z = cho_solve(cho_factor(Z), w * y)
    return z

# Function to save data to a CSV file:
def save_to_csv(wavenumbers_local, absorbance_corrected, original_filename, method_name):
    """
    Save the corrected absorbance data to a CSV file.

    Parameters:
        - wavenumbers (np.ndarray): The wavenumbers array local to the function.
        - absorbance_corrected (np.ndarray): The BL-corrected absorbance data.
        - original_filename (str): The original filename to base the new filename on.
        - method_name (str): The method name to append to the new filename.
        - base_filename (str): The base filename without the extension.
        - new_filename (str): The new filename with the method name appended.
        - df (pd.DataFrame): The dataframe containing the corrected absorbance data.
        - df.to_csv(new_filename, index = False): Save the dataframe to a CSV file.
        - print(f"Saved baseline-corrected data to: {new_filename}"): Print the saved filename.

    Returns:
        - new_filename (str): The new filename with the method name appended.
        - df (pd.DataFrame): The dataframe containing the corrected absorbance data.
        - df.to_csv(new_filename, index = False): Save the dataframe to a CSV file.
        - print(f"Saved baseline-corrected data to: {new_filename}"): Print the saved filename.
    """
    base_filename = original_filename.replace(".csv", "")
    new_filename = f"{base_filename}_{method_name}.csv"
    df = pd.DataFrame({'cm-1': wavenumbers_local, 'Absorbance_Corrected': absorbance_corrected})
    df.to_csv(new_filename, index=False)
    print(f"Saved baseline-corrected data to: {new_filename}")

# Optimal parameters for each method:
def optimal_rubber_band():
    """
    Find the optimal parameters for the rubber-band BL correction and plot the results.

    Parameters:
        - baseline_rb (np.ndarray): The estimated BL.
        - absorbance_rb_corrected (np.ndarray): The BL-corrected absorbance data.
        - rmse_rb (float): The RMSE value for the rubber-band BL correction.

    Returns:
        - baseline_rb (np.ndarray): The estimated BL.
        - absorbance_rb_corrected (np.ndarray): The BL-corrected absorbance data.
        - rmse_rb (float): The RMSE value for the rubber-band BL correction.
        - save_to_csv(wavenumbers, absorbance_rb_corrected, 'light_water_0_edited.csv', \
        'RubberBand'): Function to save the results to a CSV file.
        - plot_data (PlotData): Dataclass containing all necessary parameters for plotting.
        - plot_results(plot_data): Function to plot the results.
    """
    baseline_rb, absorbance_rb_corrected = rubber_band_baseline_correction(absorbance_raw)
    rmse_rb = math.sqrt(mean_squared_error(absorbance_corrected_given, absorbance_rb_corrected))
    print(f'Rubber-Band Correction RMSE: {rmse_rb}')

    # Save the results to a CSV file:
    save_to_csv(wavenumbers, absorbance_rb_corrected, 'light_water_0_edited.csv', 'RubberBand')

    plot_data = PlotData(
        fig_num = 1,
        wavenumbers_local = wavenumbers,
        absorbance_raw_local = absorbance_raw,
        baseline = baseline_rb,
        absorbance_corrected_local = absorbance_rb_corrected,
        absorbance_corrected_given_local = absorbance_corrected_given,
        rmse_value = rmse_rb,
        title = 'Rubber-Band Correction',
        method = 'Rubber-Band'
    )

    plot_results(plot_data)

def optimal_polynomial_fit():
    """
    Find the optimal polynomial degree for the polynomial fitting BL correction and plot the
    results.

    Parameters:
        - best_rmse (float): The best RMSE value.
        - best_degree (int): The best polynomial degree.
        - degree (int): The polynomial degree.
        - baseline_poly (np.ndarray): The estimated BL.
        - absorbance_poly_corrected (np.ndarray): The BL-corrected absorbance data.
        - wavenumbers (np.ndarray): The wavenumbers array.
        - absorbance_raw (np.ndarray): The raw absorbance data.
        - rmse_poly (float): The RMSE value for the polynomial fitting BL correction.
        - absorbance_corrected_given (np.ndarray): The given corrected absorbance data.
        - absorbance_poly_corrected (np.ndarray): The BL-corrected absorbance data.

    Returns:
        - baseline_poly (np.ndarray): The estimated BL.
        - absorbance_poly_corrected (np.ndarray): The BL-corrected absorbance data.
        - rmse_poly (float): The RMSE value for the polynomial fitting BL correction.
        - save_to_csv(wavenumbers, absorbance_poly_corrected, 'light_water_0_edited.csv', \
        'PolyFit'): Function to save the results to a CSV file.
        - plot_data (PlotData): Dataclass containing all necessary parameters for plotting.
        - plot_results(plot_data): Function to plot the results.
    """
    best_rmse = float('inf')
    best_degree = 1
    for degree in range(1, 10):
        baseline_poly, absorbance_poly_corrected = polynomial_fit_baseline(wavenumbers, \
                                                                           absorbance_raw, degree)
        rmse_poly = math.sqrt(mean_squared_error(absorbance_corrected_given, \
                                                 absorbance_poly_corrected))
        if rmse_poly < best_rmse:
            best_rmse = rmse_poly
            best_degree = degree
        else:
            break
    print(f'Best Polynomial Degree: {best_degree} with RMSE: {best_rmse}')

    # Save the results to CSV:
    save_to_csv(wavenumbers, absorbance_poly_corrected, 'light_water_0_edited.csv', 'PolyFit')

    plot_data = PlotData(
        fig_num = 2,
        wavenumbers_local = wavenumbers,
        absorbance_raw_local = absorbance_raw,
        baseline = baseline_poly,
        absorbance_corrected_local = absorbance_poly_corrected,
        absorbance_corrected_given_local = absorbance_corrected_given,
        rmse_value = best_rmse,
        title = 'Polynomial Fitting',
        method = 'Poly'
    )

    plot_results(plot_data)

def optimal_asls():
    """
    Find the optimal parameters for the AsLS smoothing BL correction and plot the results.

    Parameters:
        - best_rmse (float): The best RMSE value.
        - best_params (tuple): The best parameters.
        - lam (float): The lambda parameter.
        - p (float): The asymmetric parameter.
        - baseline_als_corr (np.ndarray): The estimated BL.
        - absorbance_raw (np.ndarray): The raw absorbance data.
        - absorbance_als_corrected (np.ndarray): The BL-corrected absorbance data.
        - rmse_als (float): The RMSE value for the AsLS smoothing BL correction.
        - absorbance_corrected_given (np.ndarray): The given corrected absorbance data.

    Returns:
        - baseline_als_corr (np.ndarray): The estimated BL.
        - absorbance_als_corrected (np.ndarray): The BL-corrected absorbance data.
        - rmse_als (float): The RMSE value for the AsLS smoothing BL correction.
        - save_to_csv(wavenumbers, absorbance_als_corrected, 'light_water_0_edited.csv', 'AsLS'): \
        Function to save the results to a CSV file.
        - plot_data (PlotData): Dataclass containing all necessary parameters for plotting.
        - plot_results(plot_data): Function to plot the results.
    """
    best_rmse = float('inf')
    best_params = None
    for lam in [1e4, 1e5, 1e6, 1e7]:
        for p in [0.001, 0.01, 0.1]:
            baseline_als_corr = baseline_als(np.array(absorbance_raw), lam, p)
            absorbance_als_corrected = np.array(absorbance_raw) - baseline_als_corr
            rmse_als = math.sqrt(mean_squared_error(absorbance_corrected_given, \
                                                    absorbance_als_corrected))
            if rmse_als < best_rmse:
                best_rmse = rmse_als
                best_params = (lam, p)

    print(f'Best AsLS Parameters: Lambda={best_params[0]}, p={best_params[1]} with RMSE: \
          {best_rmse}')

    # Save the results to CSV:
    save_to_csv(wavenumbers, absorbance_als_corrected, 'light_water_0_edited.csv', 'AsLS')

    plot_data = PlotData(
        fig_num = 3,
        wavenumbers_local = wavenumbers,
        absorbance_raw_local = absorbance_raw,
        baseline = baseline_als_corr,
        absorbance_corrected_local = absorbance_als_corrected,
        absorbance_corrected_given_local = absorbance_corrected_given,
        rmse_value = best_rmse,
        title = 'Asymmetric Least Squares Smoothing',
        method = 'AsLS'
    )

    plot_results(plot_data)


def optimal_savitzky_golay():
    """
    Find the optimal parameters for the Savitzky-Golay filtering BL correction and plot the
    results.

    Parameters:
        - best_rmse (float): The best RMSE value.
        - best_params (tuple): The best parameters.
        - window_length (int): The window length.
        - polyorder (int): The polynomial order.
        - baseline_sg (np.ndarray): The estimated BL.
        - absorbance_sg_corrected (np.ndarray): The BL-corrected absorbance data.
        - rmse_sg (float): The RMSE value for the SG filtering BL correction.
        - absorbance_corrected_given (np.ndarray): The given corrected absorbance data.

    Returns:
        - baseline_sg (np.ndarray): The estimated BL.
        - absorbance_sg_corrected (np.ndarray): The BL-corrected absorbance data.
        - rmse_sg (float): The RMSE value for the SG filtering BL correction.
        - save_to_csv(wavenumbers, absorbance_sg_corrected, \
        'light_water_0_edited.csv', 'SGFilter'): Function to save the results to a CSV file.
        - plot_data (PlotData): Dataclass containing all necessary parameters for plotting.
        - plot_results(plot_data): Function to plot the results.
    """
    best_rmse = float('inf')
    best_params = None
    for window_length in range(5, 101, 2):  # Testing odd window lengths from five to 99.
        for polyorder in range(1, 5):  # Testing polynomial orders from one to four.
            baseline_sg, absorbance_sg_corrected = savitzky_golay_baseline(absorbance_raw, \
                                                                           window_length, polyorder)
            rmse_sg = math.sqrt(mean_squared_error(absorbance_corrected_given, \
                                                   absorbance_sg_corrected))
            if rmse_sg < best_rmse:
                best_rmse = rmse_sg
                best_params = (window_length, polyorder)

    print(f'Best SG Parameters: Window Length={best_params[0]}, Polyorder={best_params[1]} with \
          RMSE: {best_rmse}')

    # Save the results to CSV:
    save_to_csv(wavenumbers, absorbance_sg_corrected, 'light_water_0_edited.csv', 'SGFilter')

    plot_data = PlotData(
        fig_num = 4,
        wavenumbers_local = wavenumbers,
        absorbance_raw_local = absorbance_raw,
        baseline = baseline_sg,
        absorbance_corrected_local = absorbance_sg_corrected,
        absorbance_corrected_given_local = absorbance_corrected_given,
        rmse_value = best_rmse,
        title = 'Savitzky-Golay Filtering',
        method = 'SG'
    )

    plot_results(plot_data)

def optimal_whittaker():
    """
    Find the optimal parameters for the Whittaker smoothing BL correction and plot the
    results.
    Parameters:
        - best_rmse (float): The best RMSE value.
        - best_lambda (float): The best lambda value.
        - lambda_ (float): The lambda parameter.
        - w (np.ndarray): The weights.
        - baseline_whittaker (np.ndarray): The estimated BL.
        - absorbance_whittaker_corrected (np.ndarray): The BL-corrected absorbance data.
        - rmse_whittaker (float): The RMSE value for the Whittaker smoothing BL correction.
        - absorbance_corrected_given (np.ndarray): The given corrected absorbance data.
    Returns:
        - best_lambda (float): The best lambda value.
        - best_rmse (float): The best RMSE value.
        - baseline_whittaker (np.ndarray): The estimated BL.
        - absorbance_whittaker_corrected (np.ndarray): The BL-corrected absorbance data.
        - save_to_csv(wavenumbers, absorbance_whittaker_corrected, 'light_water_0_edited.csv', \
        'WhittakerSmooth'): Function to save the results to a CSV file.
        - plot_data (PlotData): Dataclass containing all necessary parameters for plotting.
        - plot_results(plot_data): Function to plot the results.
    """
    best_rmse = float('inf')
    best_lambda = None
    for lambda_ in [1e3, 1e4, 1e5, 1e6]:
        w = np.ones_like(absorbance_raw)
        baseline_whittaker = whittaker_smooth(absorbance_raw, w, lambda_, differences=2)
        absorbance_whittaker_corrected = absorbance_raw - baseline_whittaker
        rmse_whittaker = math.sqrt(mean_squared_error(absorbance_corrected_given, \
                                                      absorbance_whittaker_corrected))
        if rmse_whittaker < best_rmse:
            best_rmse = rmse_whittaker
            best_lambda = lambda_
    print(f'Best Whittaker Lambda: {best_lambda} with RMSE: {best_rmse}')

    # Save the results to CSV:
    save_to_csv(wavenumbers, absorbance_whittaker_corrected, 'light_water_0_edited.csv', \
                'WhittakerSmooth')

    plot_data = PlotData(
        fig_num = 5,
        wavenumbers_local = wavenumbers,
        absorbance_raw_local = absorbance_raw,
        baseline = baseline_whittaker,
        absorbance_corrected_local = absorbance_whittaker_corrected,
        absorbance_corrected_given_local = absorbance_corrected_given,
        rmse_value = best_rmse,
        title = 'Whittaker Smoothing',
        method = 'Whittaker'
    )

    plot_results(plot_data)

# Find optimal parameters for each method and plot results:
optimal_rubber_band()
optimal_polynomial_fit()
optimal_asls()
optimal_savitzky_golay()
optimal_whittaker()

# Pause to prevent the closing of all figures when the script finishes:
plt.show()
