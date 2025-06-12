##
## By Rucha Deshpande
## Code refactor by Elim Thompson (6/11/2025)
## 
## This script took a processed data (from process_datasets.py) and
## apply an isolation forest model to detect anomalies in the shape.
##
## To run this script, use the command:
##     python detect_shape_anomaly.py --data_path 'path/to/data.p' 
######################################################################

################################
## Import packages
################################
import os
import pickle
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from skimage import io
from sklearn.ensemble import IsolationForest

################################
## Define constants
################################
# Extreme percentiles that defines outliers
extreme_percentiles = [0.1, 99.9]

################################
## Define functions
################################
def plot_image (array2D, title, fullfilename, edge_pixels):
    
    ''' Plot a normalized 2D array and save it as an image file.

        Inputs
        ------
            array2D: np.ndarray
                2D array to be plotted.
            title: str
                Title of the plot.
            fullfilename: str
                Full filename of the image plotted including path.
            edge: np.ndarray, optional
                Coordinates of edge pixels. If provided, overlay on top of the image.
    '''

    ## If array2D is an array of boolean, convert it to int.
    if array2D.dtype == bool: array2D = array2D.astype(int)
    normalized_array2D = (array2D - np.min(array2D)) / (np.max(array2D) - np.min(array2D))

    plt.imshow (normalized_array2D, cmap='gray')

    ## Overlay edge pixels on top of the image
    plt.scatter(edge_pixels[:, 1], edge_pixels[:, 0], color='red', s=1)

    plt.title (title)
    plt.savefig (fullfilename, bbox_inches='tight', dpi=600)
    plt.close()

def iforest_anomalies (real, synthetic):

    ''' Fit an Isolation Forest model on real data and predict anomalies
        in synthetic data. Rank and percentile scores are computed for
        the synthetic data based on scores.

        Inputs
        ------
        real: np.ndarray    
            Real data array of shape (n_samples, n_features)
        synthetic: np.ndarray
            Synthetic data array of shape (n_samples, n_features)

        Output
        ------
        results: pd.DataFrame
            DataFrame containing predictions, scores, relative rank, and percentile
            for the synthetic data.
    '''

    ## Fit Isolation Forest on real data
    clf = IsolationForest(max_samples=256, random_state=0)
    clf.fit(real)
    
    ## Obtain predictions and decision function scores for synthetic data
    ##    For binary predictions, 1 means inliers, -1 means outliers
    predictions = clf.predict(synthetic)
    scores = clf.decision_function(synthetic)
    relative_rank = scores.argsort().argsort()
    percentile = relative_rank / relative_rank.max() * 100

    ## Create dataframe to hold results
    results = {'predictions': predictions,
               'scores': scores,
               'relative_rank': relative_rank,
               'percentile': percentile}
    return pd.DataFrame(results)

def plot_extreme_images (results, synthetic_data, save_dir,
                         thresholds=extreme_percentiles, plot_bad=False):

    ''' Plot extreme images based on percentile thresholds.

        Inputs
        ------
        results: pd.DataFrame
            DataFrame containing predictions, scores, relative rank, and percentile
            for the synthetic data.
        synthetic_data: dict
            Dictionary containing synthetic data with keys 'short_filename' and
            'pixels_along_edge'.
        save_dir: str
            Directory to save the plotted images.
        thresholds: list, optional
            List of percentile thresholds to determine extreme images.
            Default is [0.1, 99.9].
        plot_bad: bool, optional
            If True, plot images with low percentiles (bad images).
            If False, plot images with high percentiles (good images).
            Default is False (plot good images).
    '''

    ## Create directory if it does not exist
    os.makedirs(save_dir, exist_ok=True)

    ## Get the synthetic images with extreme results
    cut = results['percentile'] <= thresholds[0] if plot_bad else \
          results['percentile'] >= thresholds[1]
    sub_results = results[cut]

    ## Loop through each percentile threshold and plot images
    for _, row in sub_results.iterrows():

        percentile = row['percentile']
        breast_area = row['breast_area']
        short_filename = row['short_filename']

        ## Read the image from the path
        image = io.imread(row['image_path'])

        ## Find the pixels along edges
        data_index = np.where (np.array (synthetic_data['short_filename']) == short_filename)[0][0]
        pixels_along_edge = synthetic_data['pixels_along_edge'][data_index]
        
        ## Title & full filename for saving
        image_quality = 'Bad' if plot_bad else 'Good'
        title = f'{image_quality}; Percentile: {percentile:.1f}; Breast area: {breast_area:.3f}'
        fullfilename = os.path.join(save_dir, f'{image_quality}_{short_filename}.png')

        plot_image(image, title, fullfilename, pixels_along_edge)

def get_statistics_angle_gradients (data, is_synthetic=True):

    ''' Get mean and standard deviation of angle gradients

        Inputs
        ------
        data: dict
            Dictionary containing real and synthetic data.
        is_synthetic: bool, optional
            If True, compute statistics for synthetic data.
            If False, compute statistics for real data.
            Default is True.

        Output
        ------
        mean_gradients: np.ndarray
            Mean of angle gradients across all images.
        std_gradients: np.ndarray
            Standard deviation of angle gradients across all images.
    '''

    image_type = 'synthetic' if is_synthetic else 'real'
    angle_gradients = data[image_type]['angle_gradients']
    mean_gradients = np.mean (angle_gradients, axis=0)
    std_gradients = np.std (angle_gradients, axis=0)

    return mean_gradients, std_gradients

def plot_score_distributions (sub_results, data, save_dir):
        
    ''' Plot cumulative histogram of anomaly scores and distribution
        of angle gradients. 

        Inputs
        ------
        sub_results: pd.DataFrame
            DataFrame containing scores and dataset names.
        data: dict
            Dictionary containing real and synthetic data.
        save_dir: str
            Directory to save the plotted images.            
    '''

    fig = plt.figure(figsize=(15, 6))
    gs = GridSpec(1, 2, figure=fig)

    ## Left plot: Stacked histogram of anomaly scores
    ax1 = fig.add_subplot(gs[0])
    #    Define bins based on all scores
    all_scores = sub_results['scores']
    bins = np.linspace(all_scores.min(), all_scores.max(), 50)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    #    Build histogram for each group
    groups = sub_results.groupby('dataset_name')
    hist_values, labels = [], []
    for name, group in groups:
        counts, _ = np.histogram(group['scores'], bins=bins)
        hist_values.append(counts)
        labels.append(name)
    #    Stack the histogram arrays: shape (n_groups, n_bins)
    hist_array = np.vstack(hist_values)
    bottom = np.zeros_like(hist_array[0])
    colors = plt.get_cmap('tab10')
    for i in range (hist_array.shape[0]):
        ax1.bar(bin_centers, hist_array[i], bottom=bottom,
                width=np.diff(bins)[0], label=labels[i],
                color=colors(i))
        bottom += hist_array[i]

    ax1.set_title("Stacked histogram of anomaly scores by dataset")
    ax1.set_xlabel("Anomaly score")
    ax1.set_ylabel("Cumulative counts")
    ax1.legend()
    ax1.grid()

    ## Right plot: Distribution of angle gradient per (64) edge bin
    ax2 = fig.add_subplot(gs[1])
    xvalues = np.arange (0, len(data['synthetic']['angle_gradients'][0]))
    #    Real data statistics per edge bin
    mean, std = get_statistics_angle_gradients (data, is_synthetic=False)
    ax2.errorbar(xvalues, mean, yerr=std, color='orange',ecolor='orange',
                 elinewidth=0.6, label='Real', linestyle=':') 
    #    Synthetic data statistics per edge bin
    mean, std = get_statistics_angle_gradients (data, is_synthetic=True)
    ax2.errorbar(xvalues, mean, yerr=std, color='green',ecolor='green',
                 elinewidth=0.6, label='Synthetic', linestyle=':')

    ax2.set_xlabel('Bins along boundary (top to bottom)')
    ax2.set_ylabel('Angle gradients as feature')
    ax2.set_title('Feature distribution')
    ax2.legend()
    ax2.grid()

    plt.savefig(save_dir + '/anomaly_score_distribution.png', bbox_inches='tight', dpi=600)
    plt.close()

def analysis (data, csv_path, thresholds=extreme_percentiles, do_plots=False):

    ''' Perform analysis on the data to detect shape anomalies using
        Isolation Forest. 

        Inputs
        ------
        data: dict
            Dictionary containing real and synthetic data.
        csv_path: str
            Path to save the results as a CSV file and plots
        thresholds: list, optional
            List of percentile thresholds to determine extreme images.
            Default is [0.1, 99.9].
        do_plots: bool, optional
            If True, plot extreme images and score distributions.
            If False, only save results to a CSV file.
            Default is False.
    '''

    ## Fit Isolation Forest model on real data and predict anomalies
    real = np.stack (data['real']['angle_gradient_distributions'])
    synthetic = np.stack (data['synthetic']['angle_gradient_distributions'])
    results = iforest_anomalies (real, synthetic)

    ## Merge results with data's image_path, breast_area, short_name
    results['image_path'] = data['synthetic']['image_path']
    results['breast_area'] = data['synthetic']['breast_area']
    results['dataset_name'] = data['synthetic']['dataset_name']
    results['short_filename'] = data['synthetic']['short_filename']

    ## Save results to a csv file
    results.to_csv (csv_path, index=False)
    if not do_plots: return

    ## Visualize best and worst images
    plot_dir = os.path.dirname(csv_path)
    save_dir = os.path.join(plot_dir, 'extreme_images')
    plot_extreme_images (results, data['synthetic'], save_dir,
                         thresholds=thresholds, plot_bad=True)
    plot_extreme_images (results, data['synthetic'], save_dir,
                         thresholds=thresholds, plot_bad=False)                         
    ## Visualize score and feature distributions
    plot_score_distributions (results[['scores', 'dataset_name']], data, plot_dir)

################################
## Script starts here
################################
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Process an integer argument.")
    parser.add_argument('--verbose', action='store_true', default=False, help='Enable verbose output')
    parser.add_argument('--data_path', type=str, help='File path to the data dictionary')
    parser.add_argument('--do_plots', action='store_true', default=False, help='Enable plotting of results')
    parser.add_argument('--bad_percentile', type=float, default=extreme_percentiles[0], help='Percentile threshold for bad images')
    parser.add_argument('--good_percentile', type=float, default=extreme_percentiles[1], help='Percentile threshold for good images')
    args = parser.parse_args()

    verbose = args.verbose
    do_plots = args.do_plots
    data_path = args.data_path
    extreme_percentiles = [args.bad_percentile, args.good_percentile]

    with open (data_path, 'rb') as f:
        data = pickle.load(f)
    f.close()

    ## Define the path to save results
    csv_path = os.path.join(os.path.dirname(data_path), 'shape_anomaly_results.csv')
    analysis (data, csv_path, thresholds=extreme_percentiles, do_plots=do_plots)

