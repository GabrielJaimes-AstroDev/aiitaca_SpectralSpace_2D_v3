import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
import re
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
from tqdm import tqdm
from glob import glob
import shutil
import zipfile
from astropy.io import fits

# Set page configuration
st.set_page_config(
    page_title="B.2D Spectral Space Analyzer",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header2 {
        font-size: 2.5rem; 
        color: #2ca02c; 
        margin-bottom: 1rem;
        text-align: center;  /* <-- AÑADE ESTA LÍNEA */
    }
    .section-header {font-size: 1.8rem; color: #1f77b4; border-bottom: 2px solid #1f77b4; padding-bottom: 0.3rem; margin-top: 1.5rem;}
    .info-box {background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;}
    .stButton>button {width: 100%;}
    .main-header {
        font-size: 1.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

def sanitize_filename(filename):
    invalid_chars = r'[<>:"/\\|?*]'
    return re.sub(invalid_chars, '_', filename)

def load_model(model_file):
    try:
        model = pickle.load(model_file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def extract_molecule_formula(header):
    pattern = r"molecules=['\"]([^,'\"]+)"
    match = re.search(pattern, header)
    if match:
        formula = match.group(1)
        if ',' in formula:
            formula = formula.split(',')[0]
        return formula
    return "Unknown"

def load_and_interpolate_spectrum(file_content, filename, reference_frequencies):
    try:
        try:
            lines = file_content.decode('utf-8').splitlines()
        except UnicodeDecodeError:
            lines = file_content.decode('latin-1').splitlines()
    except Exception as e:
        raise ValueError(f"Could not decode file {filename}: {e}")

    first_line = lines[0].strip()
    second_line = lines[1].strip() if len(lines) > 1 else ""
    
    formula = "Unknown"
    param_dict = {}
    data_start_line = 0
    
    # Format 1
    if first_line.startswith('//') and 'molecules=' in first_line:
        header = first_line[2:].strip()  # Remove the '//'
        formula = extract_molecule_formula(header)
        
        for part in header.split():
            if '=' in part:
                try:
                    key, value = part.split('=')
                    key = key.strip()
                    value = value.strip("'")
                    if key in ['molecules', 'sourcesize']:
                        continue
                    try:
                        param_dict[key] = float(value)
                    except ValueError:
                        param_dict[key] = value
                except:
                    continue
        data_start_line = 1
    
    # Format 2
    elif first_line.startswith('!') or first_line.startswith('#'):
        # Try to extract information from header if available
        if 'molecules=' in first_line:
            formula = extract_molecule_formula(first_line)
        data_start_line = 1
    
    # Format 3
    else:
        data_start_line = 0
        formula = filename.split('.')[0]  # Use filename as formula

    spectrum_data = []
    for line in lines[data_start_line:]:
        line = line.strip()
        if not line or line.startswith('!') or line.startswith('#'):
            continue
            
        try:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    freq = float(parts[0])
                    intensity = float(parts[1])
                except ValueError:
                    freq_str = parts[0].replace('D', 'E').replace('d', 'E')
                    intensity_str = parts[1].replace('D', 'E').replace('d', 'E')
                    freq = float(freq_str)
                    intensity = float(intensity_str)
                
                if np.isfinite(freq) and np.isfinite(intensity):
                    spectrum_data.append([freq, intensity])
        except Exception as e:
            st.warning(f"Could not parse line '{line}': {e}")
            continue

    if not spectrum_data:
        raise ValueError("No valid data points found in spectrum file")

    spectrum_data = np.array(spectrum_data)

    if np.max(spectrum_data[:, 0]) < 1e11:  # If frequencies are less than 100 GHz, probably in GHz
        spectrum_data[:, 0] = spectrum_data[:, 0] * 1e9  # Convert GHz to Hz
        st.info(f"Converted frequencies from GHz to Hz for {filename}")

    interpolator = interp1d(spectrum_data[:, 0], spectrum_data[:, 1],
                            kind='linear', bounds_error=False, fill_value=0.0)
    interpolated = interpolator(reference_frequencies)

    params = [
        param_dict.get('logn', np.nan),
        param_dict.get('tex', np.nan),
        param_dict.get('velo', np.nan),
        param_dict.get('fwhm', np.nan)
    ]

    return spectrum_data, interpolated, formula, params, filename

def find_knn_neighbors(training_embeddings, new_embeddings, k=5):
    if len(training_embeddings) == 0 or len(new_embeddings) == 0:
        return []
    
    k = min(k, len(training_embeddings))
    
    knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    knn.fit(training_embeddings)
    
    all_neighbor_indices = []
    for new_embedding in new_embeddings:
        distances, indices = knn.kneighbors([new_embedding])
        # Verify indices are within valid range
        valid_indices = [idx for idx in indices[0] if idx < len(training_embeddings)]
        all_neighbor_indices.append(valid_indices)
    
    return all_neighbor_indices

def truncate_filename(filename, max_length=20):
    if len(filename) > max_length:
        return filename[:max_length-3] + "..."
    return filename

def truncate_title(title, max_length=50):
    if len(title) > max_length:
        return title[:max_length-3] + "..."
    return title

def calculate_parameter_uncertainty(model, neighbor_indices):
    uncertainties = []
    expected_values = []
    
    for i in range(4):  # For each parameter (logn, tex, velo, fwhm)
        param_values = model['y'][neighbor_indices, i]
        valid_values = param_values[~np.isnan(param_values)]
        
        if len(valid_values) > 0:
            expected_value = np.mean(valid_values)
            uncertainty = np.std(valid_values)
        else:
            expected_value = np.nan
            uncertainty = np.nan
            
        expected_values.append(expected_value)
        uncertainties.append(uncertainty)
    
    return expected_values, uncertainties

def plot_parameter_vs_neighbors(model, results, selected_idx, max_neighbors=20, expected_values=None, expected_errors=None):
    if selected_idx >= len(results['umap_embedding_new']):
        return None
    
    new_embedding = results['umap_embedding_new'][selected_idx]
    filename = results['filenames_new'][selected_idx]
    
    neighbor_range = range(1, min(max_neighbors + 1, len(model['embedding']) + 1))
    
    param_data = {
        'n_neighbors': [],
        'logn_mean': [], 'logn_std': [],
        'tex_mean': [], 'tex_std': [],
        'velo_mean': [], 'velo_std': [],
        'fwhm_mean': [], 'fwhm_std': []
    }
    
    for k in neighbor_range:
        knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
        knn.fit(model['embedding'])
        distances, indices = knn.kneighbors([new_embedding])
        
        param_data['n_neighbors'].append(k)
        
        for i, param_name in enumerate(['logn', 'tex', 'velo', 'fwhm']):
            param_values = model['y'][indices[0], i]
            valid_values = param_values[~np.isnan(param_values)]
            
            if len(valid_values) > 0:
                param_data[f'{param_name}_mean'].append(np.mean(valid_values))
                param_data[f'{param_name}_std'].append(np.std(valid_values))
            else:
                param_data[f'{param_name}_mean'].append(np.nan)
                param_data[f'{param_name}_std'].append(np.nan)
    
    max_neighbors_avg = {}
    for i, param_name in enumerate(['logn', 'tex', 'velo', 'fwhm']):
        max_neighbors_avg[param_name] = param_data[f'{param_name}_mean'][-1] if len(param_data[f'{param_name}_mean']) > 0 else np.nan
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['log(N)', 'T_ex (K)', 'Velocity (km/s)', 'FWHM (km/s)']
    )
    
    param_names = ['logn', 'tex', 'velo', 'fwhm']
    param_labels = ['log(N)', 'T_ex (K)', 'Velocity (km/s)', 'FWHM (km/s)']
    
    for i, param in enumerate(param_names):
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        fig.add_trace(
            go.Scatter(
                x=param_data['n_neighbors'],
                y=param_data[f'{param}_mean'],
                mode='lines+markers',
                name=f'{param_labels[i]}',
                line=dict(color='blue')
            ),
            row=row, col=col
        )
        
       
        if expected_values is not None and expected_errors is not None and not np.isnan(expected_values[i]):
            fig.add_trace(
                go.Scatter(
                    x=[neighbor_range[0], neighbor_range[-1]],
                    y=[expected_values[i], expected_values[i]],
                    mode='lines',
                    name=f'{param_labels[i]} Expected',
                    line=dict(color='red'),
                    showlegend=False
                ),
                row=row, col=col
            )
            
            if not np.isnan(expected_errors[i]):
                fig.add_trace(
                    go.Scatter(
                        x=[neighbor_range[0], neighbor_range[-1], neighbor_range[-1], neighbor_range[0]],
                        y=[expected_values[i] - expected_errors[i], expected_values[i] - expected_errors[i], 
                           expected_values[i] + expected_errors[i], expected_values[i] + expected_errors[i]],
                        fill='toself',
                        fillcolor='rgba(255, 0, 0, 0.2)',
                        line=dict(color='rgba(255, 255, 255, 0)'),
                        name=f'{param_labels[i]} Error Band',
                        showlegend=False
                    ),
                    row=row, col=col
                )
        
        fig.update_xaxes(title_text='Number of Neighbors', row=row, col=col)
        fig.update_yaxes(title_text=param_labels[i], row=row, col=col)
    
    fig.update_layout(
        height=600,
        title_text=f"Parameter Convergence vs. Number of Neighbors: {truncate_filename(filename)}",
        showlegend=False
    )
    
    return fig, max_neighbors_avg

def plot_neighbors_logn_tex(model, results, selected_idx, knn_neighbors, expected_values=None, expected_errors=None):

    if selected_idx >= len(results['umap_embedding_new']):
        return None
    
    neighbor_indices = results['knn_neighbors'][selected_idx]
    
    if not neighbor_indices:
        return None
    
    neighbor_logn = model['y'][neighbor_indices, 0]
    neighbor_tex = model['y'][neighbor_indices, 1]
    neighbor_formulas = [model['formulas'][idx] for idx in neighbor_indices]
    
    avg_logn = np.nanmean(neighbor_logn)
    avg_tex = np.nanmean(neighbor_tex)
    std_logn = np.nanstd(neighbor_logn)
    std_tex = np.nanstd(neighbor_tex)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=neighbor_logn,
        y=neighbor_tex,
        mode='markers',
        marker=dict(color='blue', size=10),
        name='Neighbors',
        text=neighbor_formulas,
        hovertemplate='<b>Formula:</b> %{text}<br><b>log(N):</b> %{x:.2f}<br><b>T_ex:</b> %{y:.2f} K<extra></extra>'
    ))
    
    if not np.isnan(avg_logn) and not np.isnan(avg_tex):
        fig.add_trace(go.Scatter(
            x=[avg_logn],
            y=[avg_tex],
            mode='markers',
            marker=dict(color='red', size=15, symbol='star'),
            name='Average of Neighbors',
            hovertemplate='<b>Average</b><br><b>log(N):</b> %{x:.2f} ± %{customdata[0]:.2f}<br><b>T_ex:</b> %{y:.2f} ± %{customdata[1]:.2f} K<extra></extra>',
            customdata=[[std_logn, std_tex]]
        ))
        
        if not np.isnan(std_logn) and std_logn > 0:
            fig.add_trace(go.Scatter(
                x=[avg_logn - std_logn, avg_logn + std_logn],
                y=[avg_tex, avg_tex],
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                name='log(N) Std Dev',
                showlegend=False,
                hovertemplate='<b>log(N) Std Dev:</b> ±%{x:.2f}<extra></extra>'
            ))
        
        if not np.isnan(std_tex) and std_tex > 0:
            fig.add_trace(go.Scatter(
                x=[avg_logn, avg_logn],
                y=[avg_tex - std_tex, avg_tex + std_tex],
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                name='T_ex Std Dev',
                showlegend=False,
                hovertemplate='<b>T_ex Std Dev:</b> ±%{y:.2f} K<extra></extra>'
            ))
    
    if (expected_values is not None and 
        not np.isnan(expected_values[0]) and 
        not np.isnan(expected_values[1])):
        
        fig.add_trace(go.Scatter(
            x=[expected_values[0]],
            y=[expected_values[1]],
            mode='markers',
            marker=dict(color='green', size=15, symbol='diamond'),
            name='Expected Value',
            hovertemplate='<b>Expected</b><br><b>log(N):</b> %{x:.2f}<br><b>T_ex:</b> %{y:.2f} K<extra></extra>'
        ))
        
        if (expected_errors is not None and 
            not np.isnan(expected_errors[0]) and 
            not np.isnan(expected_errors[1]) and
            expected_errors[0] > 0 and expected_errors[1] > 0):
            
            fig.add_trace(go.Scatter(
                x=[expected_values[0] - expected_errors[0], expected_values[0] + expected_errors[0]],
                y=[expected_values[1], expected_values[1]],
                mode='lines',
                line=dict(color='green', width=3),
                name='log(N) Error',
                showlegend=False,
                hovertemplate='<b>log(N) Error:</b> ±%{x:.2f}<extra></extra>'
            ))
            fig.add_trace(go.Scatter(
                x=[expected_values[0], expected_values[0]],
                y=[expected_values[1] - expected_errors[1], expected_values[1] + expected_errors[1]],
                mode='lines',
                line=dict(color='green', width=3),
                name='T_ex Error',
                showlegend=False,
                hovertemplate='<b>T_ex Error:</b> ±%{y:.2f} K<extra></extra>'
            ))
    
    fig.update_layout(
        title=f"Neighbors in LogN vs T_ex Space (k={knn_neighbors})",
        xaxis_title="log(N)",
        yaxis_title="T_ex (K)",
        height=500,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def extract_filter_params(filename):
    velo_match = re.search(r'velo(-?[0-9]+(?:\.[0-9]+)?)', filename)
    fwhm_match = re.search(r'fwhm([0-9]+(?:\.[0-9]+)?)', filename)
    sigma_match = re.search(r'sigma([0-9]+(?:\.[0-9]+)?)', filename)

    try:
        velo = float(velo_match.group(1)) if velo_match else None
    except Exception:
        velo = None
    try:
        fwhm = float(fwhm_match.group(1)) if fwhm_match else None
    except Exception:
        fwhm = None
    try:
        sigma = float(sigma_match.group(1)) if sigma_match else None
    except Exception:
        sigma = None

    return velo, fwhm, sigma

def get_available_filter_params(filters_dir):
    filter_files = glob(os.path.join(filters_dir, "*.txt"))
    
    velocities = set()
    fwhms = set()
    sigmas = set()
    
    for filter_file in filter_files:
        velo, fwhm, sigma = extract_filter_params(os.path.basename(filter_file))
        if velo is not None:
            velocities.add(velo)
        if fwhm is not None:
            fwhms.add(fwhm)
        if sigma is not None:
            sigmas.add(sigma)
    
    return sorted(velocities), sorted(fwhms), sorted(sigmas)

def apply_filter_to_spectrum(spectrum_data, filter_path, output_dir):

    try:
        filter_data = np.loadtxt(filter_path, comments='/')
        freq_filter_hz = filter_data[:, 0]  # Hz
        intensity_filter = filter_data[:, 1]
        freq_filter = freq_filter_hz / 1e9  # Convert to GHz

        if np.max(intensity_filter) > 0:
            intensity_filter = intensity_filter / np.max(intensity_filter)

        freq_spectrum = spectrum_data[:, 0]  # GHz
        intensity_spectrum = spectrum_data[:, 1]  # K
        interp_spec = interp1d(freq_spectrum, intensity_spectrum, kind='cubic', bounds_error=False, fill_value=0)
        spectrum_on_filter = interp_spec(freq_filter)

        filtered_intensities = spectrum_on_filter * intensity_filter

        filtered_freqs_hz = freq_filter * 1e9

        return np.column_stack((filtered_freqs_hz, filtered_intensities))

    except Exception as e:
        st.error(f"Error applying filter {os.path.basename(filter_path)}: {str(e)}")
        return None

def generate_filtered_spectra(spectrum_data, filters_dir, selected_velo, selected_fwhm, selected_sigma, allow_negative=True):

    filter_files = glob(os.path.join(filters_dir, "*.txt"))
    filtered_spectra = []
    
    for filter_path in filter_files:
        filter_name = os.path.basename(filter_path)
        velo, fwhm, sigma = extract_filter_params(filter_name)
        

        if (velo == selected_velo and 
            fwhm == selected_fwhm and 
            sigma == selected_sigma):
            
            filtered_spectrum = apply_filter_to_spectrum(spectrum_data, filter_path, tempfile.gettempdir())
            if filtered_spectrum is not None:
                if not allow_negative:
                    filtered_spectrum[:, 1] = np.where(filtered_spectrum[:, 1] < 0, 0, filtered_spectrum[:, 1])
                filtered_spectra.append((filter_name, filtered_spectrum))
    
    return filtered_spectra

def main():

    st.image("NGC6523_BVO_2.jpg", use_column_width=True)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.empty()
        
    with col2:
        st.markdown('<p class="main-header">AI-ITACA | Artificial Intelligence Integral Tool for AstroChemical Analysis</p>', unsafe_allow_html=True)
    
    st.markdown("""
    A remarkable upsurge in the complexity of molecules identified in the interstellar medium (ISM) is currently occurring, with over 80 new species discovered in the last three years. A number of them have been emphasized by prebiotic experiments as vital molecular building blocks of life. Since our Solar System was formed from a molecular cloud in the ISM, it prompts the query as to whether the rich interstellar chemical reservoir could have played a role in the emergence of life. The improved sensitivities of state-of-the-art astronomical facilities, such as the Atacama Large Millimeter/submillimeter Array (ALMA) and the James Webb Space Telescope (JWST), are revolutionizing the discovery of new molecules in space. However, we are still just scraping the tip of the iceberg. We are far from knowing the complete catalogue of molecules that astrochemistry can offer, as well as the complexity they can reach.<br><br>
    <strong>Artificial Intelligence Integral Tool for AstroChemical Analysis (AI-ITACA)</strong>, proposes to combine complementary machine learning (ML) techniques to address all the challenges that astrochemistry is currently facing. AI-ITACA will significantly contribute to the development of new AI-based cutting-edge analysis software that will allow us to make a crucial leap in the characterization of the level of chemical complexity in the ISM, and in our understanding of the contribution that interstellar chemistry might have in the origin of life.
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h4>About GUAPOS</h4>
    <p>The G31.41+0.31 Unbiased ALMA sPectral Observational Survey (GUAPOS) project targets the hot molecular core (HMC) G31.41+0.31 (G31) to reveal the complex chemistry of one of the most chemically rich high-mass star-forming regions outside the Galactic center (GC).</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header2">🧪 2D Spectral Space Analyzer</h1>', unsafe_allow_html=True)
    
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'spectra_files' not in st.session_state:
        st.session_state.spectra_files = []
    if 'expected_values' not in st.session_state:
        st.session_state.expected_values = None
    if 'uncertainties' not in st.session_state:
        st.session_state.uncertainties = None
    if 'user_expected_values' not in st.session_state:
        st.session_state.user_expected_values = [np.nan, np.nan, np.nan, np.nan]
    if 'user_expected_errors' not in st.session_state:
        st.session_state.user_expected_errors = [np.nan, np.nan, np.nan, np.nan]
    if 'max_neighbors_avg' not in st.session_state:
        st.session_state.max_neighbors_avg = None
    if 'filtered_spectra' not in st.session_state:
        st.session_state.filtered_spectra = []
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Input Parameters")
        
        # Model upload
        st.subheader("1. Upload Model")
        model_file = st.file_uploader("Upload trained model (PKL file)", type=['pkl'])
        
        if model_file is not None:
            if st.button("Load Model") or st.session_state.model is None:
                with st.spinner("Loading model..."):
                    st.session_state.model = load_model(model_file)
                    if st.session_state.model is not None:
                        st.success("Model loaded successfully!")
        
        # Spectra upload
        st.subheader("2. Upload Spectrum")
        spectrum_file = st.file_uploader(
            "Upload spectrum file (TXT, FITS, SPEC, DAT)", 
            type=['txt', 'fits', 'spec', 'dat']
        )
        
        if spectrum_file:
            st.session_state.spectrum_file = spectrum_file
        
        st.subheader("3. Filter Parameters")
        
        filters_dir = "1.Filters"
        
        if os.path.exists(filters_dir):
            velocities, fwhms, sigmas = get_available_filter_params(filters_dir)
            
            if velocities and fwhms and sigmas:
                selected_velo = st.selectbox("Velocity", velocities, index=0)
                selected_fwhm = st.selectbox("FWHM", fwhms, index=0)
                selected_sigma = st.selectbox("Sigma", sigmas, index=0)
                
                st.session_state.selected_velo = selected_velo
                st.session_state.selected_fwhm = selected_fwhm
                st.session_state.selected_sigma = selected_sigma


                consider_absorption = st.checkbox("Consider absorption lines (allow negative values)", value=False)
                st.session_state.consider_absorption = consider_absorption
            else:
                st.error("No valid filters found in the '1.Filters' directory")
        else:
            st.error("Filters directory '1.Filters' not found")
        

        st.subheader("4. Analysis Parameters")
        knn_neighbors = st.slider("Number of KNN neighbors", min_value=1, max_value=50, value=5)
        max_neighbors_plot = st.slider("Max neighbors for convergence plot", min_value=5, max_value=100, value=20)
        
        st.subheader("5. Expected Values (Optional)")
        st.markdown("Enter expected values and errors for comparison:")
        
        param_labels = ['log(N)', 'T_ex (K)', 'Velocity (km/s)', 'FWHM (km/s)']
        user_expected_values = []
        user_expected_errors = []
        
        for i, label in enumerate(param_labels):
            col1, col2 = st.columns(2)
            with col1:
                value = st.number_input(f"{label} value", value=float('nan'), key=f"exp_val_{i}")
                user_expected_values.append(value)
            with col2:
                error = st.number_input(f"{label} error", value=float('nan'), min_value=0.0, key=f"exp_err_{i}")
                user_expected_errors.append(error)
        
        st.session_state.user_expected_values = user_expected_values
        st.session_state.user_expected_errors = user_expected_errors
        
        if st.button("Generate Filtered Spectra and Analyze") and st.session_state.model is not None and hasattr(st.session_state, 'spectrum_file'):
            with st.spinner("Generating filtered spectra and analyzing..."):
                try:

                    spectrum_content = st.session_state.spectrum_file.getvalue()
                    spectrum_filename = st.session_state.spectrum_file.name
                    
                    lines = spectrum_content.decode('utf-8').splitlines()
                    data_lines = [line for line in lines if not (line.strip().startswith('!') or line.strip().startswith('//'))]
                    spectrum_data = np.loadtxt(data_lines)
                    
                    filtered_spectra = generate_filtered_spectra(
                        spectrum_data, 
                        filters_dir, 
                        st.session_state.selected_velo, 
                        st.session_state.selected_fwhm, 
                        st.session_state.selected_sigma,
                        allow_negative=st.session_state.consider_absorption
                    )
                    
                    if not filtered_spectra:
                        st.error("No filters found matching the selected parameters")
                        return
                    
                    st.session_state.filtered_spectra = filtered_spectra
                    
                    spectra_files = []
                    for filter_name, filtered_data in filtered_spectra:
                        # Create a temporary file-like object
                        file_obj = tempfile.NamedTemporaryFile(delete=False, suffix='.txt')
                        np.savetxt(file_obj, filtered_data, delimiter='\t', fmt=['%.10f', '%.6e'])
                        file_obj.seek(0)
                        # Guardar el nombre descriptivo en el objeto
                        file_obj.name = filter_name  # <--- CAMBIO AQUÍ
                        spectra_files.append(file_obj)
                    
                    model = st.session_state.model
                    results = analyze_spectra(model, spectra_files, knn_neighbors)
                    st.session_state.results = results
                    
                    if len(results['umap_embedding_new']) > 0:
                        expected_values_list = []
                        uncertainties_list = []
                        
                        for i in range(len(results['umap_embedding_new'])):
                            if i < len(results['knn_neighbors']):
                                expected_values, uncertainties = calculate_parameter_uncertainty(
                                    model, results['knn_neighbors'][i]
                                )
                                expected_values_list.append(expected_values)
                                uncertainties_list.append(uncertainties)
                        
                        st.session_state.expected_values = expected_values_list
                        st.session_state.uncertainties = uncertainties_list
                    
                    st.success(f"Analysis completed! Generated {len(filtered_spectra)} filtered spectra.")
                    
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
    
    if (st.session_state.expected_values is not None and 
        st.session_state.uncertainties is not None and
        st.session_state.results is not None and
        len(st.session_state.results['filenames_new']) > 0):
        
        with st.sidebar:
            st.subheader("Estimated Parameters")
            
            selected_idx = st.selectbox(
                "Select spectrum for parameter estimates",
                range(len(st.session_state.results['filenames_new'])),
                format_func=lambda i: st.session_state.results['filenames_new'][i]
            )
            
            if selected_idx is not None:
                param_labels = ['log(N)', 'T_ex (K)', 'Velocity (km/s)', 'FWHM (km/s)']
                expected_vals = st.session_state.expected_values[selected_idx]
                uncertainties = st.session_state.uncertainties[selected_idx]
                
                for i, (label, exp_val, uncert) in enumerate(zip(param_labels, expected_vals, uncertainties)):
                    if not np.isnan(exp_val) and not np.isnan(uncert):
                        st.markdown(f"**{label}**: {exp_val:.2f} ± {uncert:.2f}")
                    else:
                        st.markdown(f"**{label}**: N/A")
    
    # Main content
    if st.session_state.model is None:
        st.info("Please upload a model file to get started.")
        return
    
    model = st.session_state.model
    
    with st.expander("Model Information", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training Samples", model.get('sample_size', 'N/A'))
        with col2:
            st.metric("PCA Components", model.get('n_components', 'N/A'))
        with col3:
            st.metric("Variance Threshold", f"{model.get('variance_threshold', 0.99)*100:.1f}%")
    
    if st.session_state.results is None:
        st.info("Upload a spectrum file, select filter parameters, and click 'Generate Filtered Spectra and Analyze' to see results.")
        return
    
    results = st.session_state.results
    
    st.header("Analysis Results")
    
    # UMAP Visualization
    st.subheader("A. UMAP Projection")
    
    train_df = pd.DataFrame({
        'umap_x': model['embedding'][:, 0],
        'umap_y': model['embedding'][:, 1],
        'formula': model['formulas'],
        'logn': model['y'][:, 0],
        'tex': model['y'][:, 1],
        'velo': model['y'][:, 2],
        'fwhm': model['y'][:, 3],
        'type': 'Training'
    })
    
    if len(results['umap_embedding_new']) > 0:
        # Create truncated filenames for legend
        truncated_filenames = [truncate_filename(fname) for fname in results['filenames_new']]
        
        new_df = pd.DataFrame({
            'umap_x': results['umap_embedding_new'][:, 0],
            'umap_y': results['umap_embedding_new'][:, 1],
            'formula': truncated_filenames,
            'full_filename': results['filenames_new'],
            'logn': results['y_new'][:, 0],
            'tex': results['y_new'][:, 1],
            'velo': results['y_new'][:, 2],
            'fwhm': results['y_new'][:, 3],
            'filename': results['filenames_new'],
            'type': 'Predicted'
        })
        
        combined_df = pd.concat([train_df, new_df], ignore_index=True)
    else:
        combined_df = train_df
    
    unique_formulas = combined_df['formula'].unique()
    color_cycle = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel + px.colors.qualitative.Dark24
    color_map = {formula: color_cycle[i % len(color_cycle)] for i, formula in enumerate(unique_formulas)}
    
    fig = go.Figure()
    
    for formula in train_df['formula'].unique():
        formula_data = train_df[train_df['formula'] == formula]
        fig.add_trace(go.Scatter(
            x=formula_data['umap_x'],
            y=formula_data['umap_y'],
            mode='markers',
            name=f"{formula} (Training)",
            marker=dict(
                color=color_map[formula],
                size=8,
                symbol='circle'
            ),
            hoverinfo='text',
            text=[f"Formula: {row['formula']}<br>log(N): {row['logn']:.2f}<br>T_ex: {row['tex']:.2f} K<br>Velocity: {row['velo']:.2f} km/s<br>FWHM: {row['fwhm']:.2f} km/s<br>Type: Training" 
                  for _, row in formula_data.iterrows()],
            legendgroup=formula,
            showlegend=True
        ))
    
    if len(results['umap_embedding_new']) > 0:
        for i, row in new_df.iterrows():
            fig.add_trace(go.Scatter(
                x=[row['umap_x']],
                y=[row['umap_y']],
                mode='markers',
                name=f"{row['formula']} (Predicted)",
                marker=dict(
                    color='black',
                    size=10,
                    symbol='diamond'
                ),
                hoverinfo='text',
                text=[f"File: {row['full_filename']}<br>Formula: {row['formula']}<br>log(N): {row['logn']:.2f}<br>T_ex: {row['tex']:.2f} K<br>Velocity: {row['velo']:.2f} km/s<br>FWHM: {row['fwhm']:.2f} km/s<br>Type: Predicted"],
                legendgroup=row['formula'],
                showlegend=True
            ))
    
    fig.update_layout(
        width=700,
        height=700,
        autosize=False,
        title="UMAP Projection of Spectral Data",
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2",
        legend=dict(
            itemsizing='constant',
            font=dict(size=10),
            tracegroupgap=0
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("B. Parameter Distributions")
    
    param_names = ['logn', 'tex', 'velo', 'fwhm']
    param_labels = ['log(N)', 'T_ex (K)', 'Velocity (km/s)', 'FWHM (km/s)']
    
    fig = make_subplots(rows=2, cols=2, subplot_titles=param_labels)
    
    for i, param in enumerate(param_names):
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        fig.add_trace(
            go.Histogram(x=train_df[param], name='Training', opacity=0.7, marker_color='orange'),
            row=row, col=col
        )
        
        if len(results['umap_embedding_new']) > 0:
            fig.add_trace(
                go.Histogram(x=new_df[param], name='Predicted', opacity=0.7, marker_color='black'),
                row=row, col=col
            )
    
    fig.update_layout(height=600, showlegend=False, title_text="Parameter Distributions")
    st.plotly_chart(fig, use_container_width=True)
    
    if len(results['umap_embedding_new']) > 0:
        st.subheader("C. Individual Spectrum Analysis")
        
        selected_idx = st.selectbox("Select a spectrum for detailed analysis", 
                                   range(len(results['filenames_new'])),
                                   format_func=lambda i: results['filenames_new'][i])
        
        if selected_idx is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Spectrum Visualization**")
                
                truncated_title = truncate_title(f"Spectrum: {results['filenames_new'][selected_idx]}")
                
                spectrum_fig = go.Figure()
                spectrum_fig.add_trace(go.Scatter(
                    x=model['reference_frequencies'],
                    y=results['X_new'][selected_idx],
                    mode='lines',
                    name=truncate_filename(results['filenames_new'][selected_idx]),
                    line=dict(color='blue', width=2)
                ))
                
                spectrum_fig.update_layout(
                    title={
                        'text': truncated_title,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top',
                        'font': {
                            'size': 11
                        }
                    },
                    xaxis_title='Frequency (Hz)',
                    yaxis_title='Intensity',
                    hovermode='x unified',
                    height=500,
                    width=600,
                    showlegend=False
                )
                
                spectrum_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                spectrum_fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                
                st.plotly_chart(spectrum_fig, use_container_width=True)
            
            with col2:
                if 'knn_neighbors' in results and selected_idx < len(results['knn_neighbors']):
                    neighbor_indices = results['knn_neighbors'][selected_idx]
                    
                    if neighbor_indices:

                        avg_params = [
                            np.nanmean(model['y'][neighbor_indices, 0]),
                            np.nanmean(model['y'][neighbor_indices, 1]),
                            np.nanmean(model['y'][neighbor_indices, 2]),
                            np.nanmean(model['y'][neighbor_indices, 3])
                        ]
                        
                        std_params = [
                            np.nanstd(model['y'][neighbor_indices, 0]),
                            np.nanstd(model['y'][neighbor_indices, 1]),
                            np.nanstd(model['y'][neighbor_indices, 2]),
                            np.nanstd(model['y'][neighbor_indices, 3])
                        ]
                        
                        neighbor_formulas = [model['formulas'][idx] for idx in neighbor_indices]
                        most_common_formula = max(set(neighbor_formulas), key=neighbor_formulas.count)
                    else:
                        avg_params = [np.nan, np.nan, np.nan, np.nan]
                        std_params = [np.nan, np.nan, np.nan, np.nan]
                        most_common_formula = "Unknown"
                else:
                    avg_params = [np.nan, np.nan, np.nan, np.nan]
                    std_params = [np.nan, np.nan, np.nan, np.nan]
                    most_common_formula = "Unknown"
                
                st.markdown("**Estimated Parameters**")
                param_data = {
                    'Parameter': param_labels,
                    'Value': [
                        f"{avg_params[0]:.2f}" if not np.isnan(avg_params[0]) else "N/A",
                        f"{avg_params[1]:.2f}" if not np.isnan(avg_params[1]) else "N/A",
                        f"{avg_params[2]:.2f}" if not np.isnan(avg_params[2]) else "N/A",
                        f"{avg_params[3]:.2f}" if not np.isnan(avg_params[3]) else "N/A"
                    ]
                }
                st.table(pd.DataFrame(param_data))
                
                st.markdown(f"**Molecule Formula**: {most_common_formula}")
            
            st.markdown("**Parameter Convergence vs. Number of Neighbors**")
            convergence_fig, max_neighbors_avg = plot_parameter_vs_neighbors(
                model, results, selected_idx, max_neighbors_plot,
                st.session_state.user_expected_values, st.session_state.user_expected_errors
            )
            st.session_state.max_neighbors_avg = max_neighbors_avg
            
            if convergence_fig:
                st.plotly_chart(convergence_fig, use_container_width=True)
                
                st.markdown("**Average Values for Maximum Neighbors**")
                avg_data = {
                    'Parameter': param_labels,
                    'Average Value': [
                        f"{max_neighbors_avg['logn']:.2f}" if not np.isnan(max_neighbors_avg['logn']) else "N/A",
                        f"{max_neighbors_avg['tex']:.2f}" if not np.isnan(max_neighbors_avg['tex']) else "N/A",
                        f"{max_neighbors_avg['velo']:.2f}" if not np.isnan(max_neighbors_avg['velo']) else "N/A",
                        f"{max_neighbors_avg['fwhm']:.2f}" if not np.isnan(max_neighbors_avg['fwhm']) else "N/A"
                    ]
                }
                st.table(pd.DataFrame(avg_data))
            
            st.markdown("**D. Neighbors in LogN vs T_ex Space**")
            logn_tex_fig = plot_neighbors_logn_tex(
                model, results, selected_idx, knn_neighbors,
                st.session_state.user_expected_values, st.session_state.user_expected_errors
            )
            
            if logn_tex_fig:
                st.plotly_chart(logn_tex_fig, use_container_width=True)
            
            st.markdown("**E. K-Nearest Neighbors Analysis**")
            
            if 'knn_neighbors' in results and selected_idx < len(results['knn_neighbors']):
                neighbor_indices = results['knn_neighbors'][selected_idx]
                
                if neighbor_indices:
                    neighbor_data = []
                    for idx in neighbor_indices:
                        neighbor_data.append({
                            'Formula': model['formulas'][idx],
                            'log(N)': f"{model['y'][idx, 0]:.2f}",
                            'T_ex (K)': f"{model['y'][idx, 1]:.2f}",
                            'Velocity': f"{model['y'][idx, 2]:.2f}",
                            'FWHM': f"{model['y'][idx, 3]:.2f}"
                        })
                    
                    st.table(pd.DataFrame(neighbor_data))
                    
                    fig = go.Figure()
                    
                    training_hover_text = [
                        f"Formula: {form}<br>log(N): {logn:.2f}<br>T_ex: {tex:.2f} K<br>Velocity: {velo:.2f} km/s<br>FWHM: {fwhm:.2f} km/s"
                        for form, logn, tex, velo, fwhm in zip(
                            train_df['formula'], train_df['logn'], train_df['tex'], 
                            train_df['velo'], train_df['fwhm']
                        )
                    ]
                    
                    fig.add_trace(go.Scatter(
                        x=train_df['umap_x'], y=train_df['umap_y'],
                        mode='markers',
                        marker=dict(color='lightgray', size=5),
                        name='Training Data',
                        text=training_hover_text,
                        hoverinfo='text'
                    ))
                    
                    neighbor_x = [model['embedding'][idx, 0] for idx in neighbor_indices]
                    neighbor_y = [model['embedding'][idx, 1] for idx in neighbor_indices]
                    neighbor_formulas = [model['formulas'][idx] for idx in neighbor_indices]
                    neighbor_logn = [model['y'][idx, 0] for idx in neighbor_indices]
                    neighbor_tex = [model['y'][idx, 1] for idx in neighbor_indices]
                    neighbor_velo = [model['y'][idx, 2] for idx in neighbor_indices]
                    neighbor_fwhm = [model['y'][idx, 3] for idx in neighbor_indices]
                    
                    neighbor_hover_text = [
                        f"Formula: {form}<br>log(N): {logn:.2f}<br>T_ex: {tex:.2f} K<br>Velocity: {velo:.2f} km/s<br>FWHM: {fwhm:.2f} km/s"
                        for form, logn, tex, velo, fwhm in zip(
                            neighbor_formulas, neighbor_logn, neighbor_tex, neighbor_velo, neighbor_fwhm
                        )
                    ]
                    
                    fig.add_trace(go.Scatter(
                        x=neighbor_x, y=neighbor_y,
                        mode='markers',
                        marker=dict(color='blue', size=10),
                        name='Neighbors',
                        text=neighbor_hover_text,
                        hoverinfo='text'
                    ))
                    
                    selected_hover_text = f"File: {results['filenames_new'][selected_idx]}<br>Formula: {most_common_formula}"
                    
                    fig.add_trace(go.Scatter(
                        x=[results['umap_embedding_new'][selected_idx, 0]],
                        y=[results['umap_embedding_new'][selected_idx, 1]],
                        mode='markers',
                        marker=dict(color='red', size=15, symbol='star'),
                        name='Selected Spectrum',
                        text=[selected_hover_text],
                        hoverinfo='text'
                    ))
                    
                    fig.update_layout(
                        title="K-Nearest Neighbors in UMAP Space",
                        width=700,
                        height=700,
                        autosize=False,
                        xaxis_title="UMAP 1",
                        yaxis_title="UMAP 2",
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No neighbors found for this spectrum.")
            else:
                st.info("KNN analysis not available for this spectrum.")
    
    st.subheader("Download Results")
    
    if st.button("Export Results to CSV"):

        if len(results['umap_embedding_new']) > 0:
            results_df = pd.DataFrame({
                'filename': results['filenames_new'],
                'formula': results['formulas_new'],
                'umap_x': results['umap_embedding_new'][:, 0],
                'umap_y': results['umap_embedding_new'][:, 1],
                'logn': results['y_new'][:, 0],
                'tex': results['y_new'][:, 1],
                'velo': results['y_new'][:, 2],
                'fwhm': results['y_new'][:, 3]
            })
            
            if (st.session_state.expected_values is not None and 
                st.session_state.uncertainties is not None):
                
                expected_array = np.array(st.session_state.expected_values)
                uncertainties_array = np.array(st.session_state.uncertainties)
                
                results_df['logn_expected'] = expected_array[:, 0]
                results_df['tex_expected'] = expected_array[:, 1]
                results_df['velo_expected'] = expected_array[:, 2]
                results_df['fwhm_expected'] = expected_array[:, 3]
                
                results_df['logn_uncertainty'] = uncertainties_array[:, 0]
                results_df['tex_uncertainty'] = uncertainties_array[:, 1]
                results_df['velo_uncertainty'] = uncertainties_array[:, 2]
                results_df['fwhm_uncertainty'] = uncertainties_array[:, 3]
            
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="spectrum_analysis_results.csv",
                mime="text/csv"
            )
        else:
            st.warning("No results to export.")

def analyze_spectra(model, spectra_files, knn_neighbors=5):
    results = {
        'X_new': [],
        'y_new': [],
        'formulas_new': [],
        'filenames_new': [],
        'pca_components_new': [],
        'umap_embedding_new': [],
        'knn_neighbors': []
    }
    
    scaler = model['scaler']
    pca = model['pca']
    umap_model = model['umap']
    ref_freqs = model['reference_frequencies']
    
    for spectrum_file in tqdm(spectra_files, desc="Processing spectra"):
        try:
            spectrum_data, interpolated, formula, params, filename = load_and_interpolate_spectrum(
                spectrum_file.read(), spectrum_file.name, ref_freqs
            )
            
            X_scaled = scaler.transform([interpolated])
            X_pca = pca.transform(X_scaled)
            X_umap = umap_model.transform(X_pca)
            
            results['X_new'].append(interpolated)
            results['formulas_new'].append(formula)
            results['y_new'].append(params)
            results['filenames_new'].append(filename)
            results['umap_embedding_new'].append(X_umap[0])
            results['pca_components_new'].append(X_pca[0])
            
        except Exception as e:
            st.warning(f"Error processing {spectrum_file.name}: {str(e)}")
            continue
    
    if not results['umap_embedding_new']:
        st.error("No valid spectra could be processed.")
        return results
    
    results['X_new'] = np.array(results['X_new'])
    results['y_new'] = np.array(results['y_new'])
    results['formulas_new'] = np.array(results['formulas_new'])
    results['umap_embedding_new'] = np.array(results['umap_embedding_new'])
    results['pca_components_new'] = np.array(results['pca_components_new'])
    
    results['knn_neighbors'] = find_knn_neighbors(
        model['embedding'], results['umap_embedding_new'], k=knn_neighbors
    )
    
    return results

def read_spectrum_file(file_obj_or_path, filename=None):

    import tempfile

    input_logn = None
    input_tex = None
    header = ""
    freq = np.array([])
    spec = np.array([])

    if filename is None:
        if isinstance(file_obj_or_path, str):
            filename = os.path.basename(file_obj_or_path)
        else:
            filename = getattr(file_obj_or_path, 'name', 'unknown')
    ext = filename.lower().split('.')[-1]

    try:

        if ext in ['txt', 'dat']:
            try:
                if isinstance(file_obj_or_path, str):
                    with open(file_obj_or_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                else:
                    content = file_obj_or_path.read().decode('utf-8')
                    lines = content.splitlines()
            except UnicodeDecodeError:
                if isinstance(file_obj_or_path, str):
                    with open(file_obj_or_path, 'r', encoding='latin-1') as f:
                        lines = f.readlines()
                else:
                    content = file_obj_or_path.read().decode('latin-1')
                    lines = content.splitlines()
            header = lines[0].strip() if lines else ""
            input_params = re.search(r'logn[\s=:]+([\d.]+).*tex[\s=:]+([\d.]+)', header.lower()) if header else None
            if input_params:
                try:
                    input_logn = float(input_params.group(1))
                    input_tex = float(input_params.group(2))
                except (ValueError, TypeError):
                    input_logn = None
                    input_tex = None
            data = []
            for line in lines[1:]:
                line = line.strip()
                if line and not line.startswith(("//", "#")):
                    parts = re.split(r'[\s,;]+', line)
                    if len(parts) >= 2:
                        try:
                            frequency = float(parts[0]) * 1e9  # Convertir a Hz
                            intensity = float(parts[1])
                            data.append((frequency, intensity))
                        except ValueError:
                            continue
            if len(data) >= 10:
                freq, spec = zip(*data)
                freq = np.array(freq)
                spec = np.array(spec)
                return freq, spec, header, input_logn, input_tex

        # FITS
        if ext == 'fits':
            if isinstance(file_obj_or_path, str):
                fits_file = file_obj_or_path
            else:
                # Guardar temporalmente si es file-like
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.fits')
                tmp.write(file_obj_or_path.read())
                tmp.close()
                fits_file = tmp.name
            with fits.open(fits_file) as hdul:
                if len(hdul) > 1:
                    table = hdul[1].data
                    all_freqs = []
                    all_intensities = []
                    for row in table:
                        spectrum = row['DATA']
                        crval3 = row['CRVAL3']
                        cdelt3 = row['CDELT3']
                        crpix3 = row['CRPIX3']
                        n = len(spectrum)
                        channels = np.arange(n)
                        frequencies = crval3 + (channels + 1 - crpix3) * cdelt3
                        all_freqs.append(frequencies)
                        all_intensities.append(spectrum)
                    combined_freqs = np.concatenate(all_freqs)
                    combined_intensities = np.concatenate(all_intensities)
                    sorted_indices = np.argsort(combined_freqs)
                    freq = combined_freqs[sorted_indices]
                    spec = combined_intensities[sorted_indices]
                    header = f"Processed from FITS file: {filename}"
                    if not isinstance(file_obj_or_path, str):
                        os.remove(fits_file)
                    return freq, spec, header, input_logn, input_tex

        # ZIP con FITS (.spec)
        if ext == 'spec' and zipfile.is_zipfile(file_obj_or_path if isinstance(file_obj_or_path, str) else file_obj_or_path.name):
            # Guardar temporalmente si es file-like
            if not isinstance(file_obj_or_path, str):
                tmp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.spec')
                tmp_zip.write(file_obj_or_path.read())
                tmp_zip.close()
                zip_path = tmp_zip.name
            else:
                zip_path = file_obj_or_path
            extract_folder = tempfile.mkdtemp(prefix="unzip_")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_folder)
                fits_files = [f for f in os.listdir(extract_folder) if f.endswith('.fits')]
                if fits_files:
                    fits_file_path = os.path.join(extract_folder, fits_files[0])
                    with fits.open(fits_file_path) as hdul:
                        table = hdul[1].data
                        all_freqs = []
                        all_intensities = []
                        for row in table:
                            spectrum = row['DATA']
                            crval3 = row['CRVAL3']
                            cdelt3 = row['CDELT3']
                            crpix3 = row['CRPIX3']
                            n = len(spectrum)
                            channels = np.arange(n)
                            frequencies = crval3 + (channels + 1 - crpix3) * cdelt3
                            all_freqs.append(frequencies)
                            all_intensities.append(spectrum)
                        combined_freqs = np.concatenate(all_freqs)
                        combined_intensities = np.concatenate(all_intensities)
                        sorted_indices = np.argsort(combined_freqs)
                        freq = combined_freqs[sorted_indices]
                        spec = combined_intensities[sorted_indices]
                        header = f"Processed from FITS file within {filename}"
                        return freq, spec, header, input_logn, input_tex
            finally:
                # Limpieza de archivos temporales
                shutil.rmtree(extract_folder)
                if not isinstance(file_obj_or_path, str):
                    os.remove(zip_path)

    except Exception as e:
        raise ValueError(f"Error al procesar el archivo {filename}: {str(e)}")

    raise ValueError("No se pudo procesar el archivo con ningún método conocido")

if __name__ == "__main__":
    main()

