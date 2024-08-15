# -*- coding: utf-8 -*-
"""
Helper for anatomical receptive field (RF) analysis.
Iincludes custom-written fiunctions for analysis and plot.
Clean code for publication

@author: Sebastian Molina-Obando
"""

#%% 
#Importing packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, levene
from itertools import combinations
import math
from caveclient import CAVEclient
client = CAVEclient('flywire_fafb_production')


#%% Plotting functions
def add_mean_median_lines(data, ax, color_mean, color_median, vertical=True):
    import numpy as np
    
    #mean_value = np.nanmean(data)
    median_value = np.nanmedian(data)
    
    if vertical:
        #ax.axvline(mean_value, color=color_mean, linestyle='dashed', linewidth=2, label=f'Mean: {mean_value:.2f}')
        ax.axvline(median_value, color=color_median, linestyle='dashed', linewidth=2, label=f'Median: {median_value:.2f}')
    else:
        #ax.axhline(mean_value, color=color_mean, linestyle='dashed', linewidth=2, label=f'Mean: {mean_value:.2f}')
        ax.axhline(median_value, color=color_median, linestyle='dashed', linewidth=2, label=f'Median: {median_value:.2f}')

def replace_outliers_with_nan(df, multiplier=1.5):
    # Calculate the first quartile (Q1) and third quartile (Q3) for each column
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)

    # Calculate the IQR for each column
    iqr = q3 - q1

    # Determine the lower and upper bounds for outlier detection
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    # Replace outlier values with NaN
    df_filtered = df.mask((df < lower_bound) | (df > upper_bound))
    
    return df_filtered

#%% Analyisis functions

def combine_xyz(df):
    """
    Combines the x, y, and z columns into a single column, converts units, and adds new columns.

    Args:
        df (pandas.DataFrame): A pandas DataFrame containing 'x', 'y', and 'z' columns of the same length.

    Returns:
        pandas.DataFrame: The input DataFrame with new columns 'post_pt_position' and 'pre_pt_position' containing
        lists of [x/4, y/4, z/40] for both postsynaptic and presynaptic neurons.

    Note:
    - The function takes a DataFrame with 'x', 'y', and 'z' columns representing coordinates.
    - New columns 'post_pt_position' and 'pre_pt_position' are added with adjusted unit values.
    - The input DataFrame is modified in-place.
    """
    # Generating the single column

    post_pt_position = []
    for x,y,z in zip(df['post_x'].tolist(),df['post_y'].tolist(),df['post_z'].tolist()):
        temp_ls = [x/4,y/4,z/40]
        post_pt_position.append(temp_ls)

    pre_pt_position = []
    for x,y,z in zip(df['pre_x'].tolist(),df['pre_y'].tolist(),df['pre_z'].tolist()):
        temp_ls = [x/4,y/4,z/40]
        pre_pt_position.append(temp_ls)

    #Adding new columns and names
    df['post_pt_position'] = post_pt_position
    df['pre_pt_position'] = pre_pt_position
    #Changing column names
    df.rename(columns={'pre': 'pre_pt_root_id', 'post': 'post_pt_root_id'}, inplace=True)


def match_all_pre_to_single_post(up_to_date_post_ids, up_to_date_pre_ids, neuropile_mesh):
    """
    Match presynaptic neurons to a single postsynaptic neuron within a specified neuropile.

    Parameters:
    - up_to_date_post_ids (list): List of up-to-date postsynaptic neuron IDs.
    - up_to_date_pre_ids (list): List of up-to-date presynaptic neuron IDs.
    - neuropile_mesh (str): Neuropile mesh name for filtering synapses in the specified neuropile.

    Returns:
    pandas.DataFrame: A DataFrame containing the count of presynaptic contacts for each unique pre-post pair.
    
    Note:
    - The function uses the flywire module from the fafbseg package to fetch synapse data.
    - The `combine_xyz` function is assumed to be defined elsewhere and is used to combine pre- and postsynaptic XYZ values.
    - Synapses are filtered based on a minimum score of 50 and only those in the specified neuropile are retained.
    - The resulting DataFrame is aggregated to count the number of contacts for each unique pre-post pair.
    """    
    from fafbseg import flywire
    print('Matching all pre to single post')
    
    # Fetch the neuron's inputs
    post_inputs = flywire.synapses.fetch_synapses(
        up_to_date_post_ids, pre=False, post=True, attach=True,
        min_score=50, clean=True, transmitters=False,
        neuropils=True, batch_size=30,
        dataset='production', progress=True, mat="live"
    )

    # Combining pre- and postsynapses XYZ values in single columns
    combine_xyz(post_inputs)  # Assuming combine_xyz is a defined function that does the operation

    # Filtering: keeping only synapses in the medulla
    post_inputs = post_inputs[post_inputs['neuropil'] == neuropile_mesh].copy()

    # Filter connections just selected presynaptic cells
    pre_post_match_df = post_inputs[post_inputs['pre_pt_root_id'].isin(up_to_date_pre_ids)].copy()

    # Aggregating data frame based on unique post and pre segment IDs
    # While aggregating, counting the number of contacts for each pre-post pair
    pre_post_counts = pre_post_match_df.groupby(['post_pt_root_id', 'pre_pt_root_id'])['pre_pt_root_id'].count().reset_index(name='pre_syn_count')


    return pre_post_counts, post_inputs




def calculate_spatial_span(up_to_date_post_ids, up_to_date_pre_ids, post_ids_update_df, R_post_df, post_inputs, pre_post_counts, pre_inputs, single_column_area,single_column_diameter):
    """
    Calculate the total and individual spatial span of presynaptic neurons contacting the same postsynaptic neuron.

    Args:
        up_to_date_post_ids (list): List of up-to-date postsynaptic neuron IDs.
        up_to_date_pre_ids (list): List of up-to-date presynaptic neuron IDs.
        post_ids_update_df (pandas.DataFrame): DataFrame containing updated and old postsynaptic neuron IDs.
        R_post_df (pandas.DataFrame): DataFrame containing postsynaptic neuron coordinates.
        post_inputs (pandas.DataFrame): DataFrame containing postsynaptic synapse data.
        pre_post_counts (pandas.DataFrame): DataFrame with counts of presynaptic contacts for each pre-post pair.
        pre_inputs (pandas.DataFrame): DataFrame containing presynaptic synapse data.
        single_column_area (float): Area covered by a single column.
        single_column_diameter (float): Diameter of a single column.

    Returns:
        Tuple[pandas.DataFrame, pandas.DataFrame]: Two DataFrames - spatial_span_df for total spatial span and
        individual_spatial_span_df for individual spatial span of presynaptic neurons.
    """
   
    
    import pandas as pd
    import numpy as np
    from scipy.spatial import ConvexHull, distance
    from scipy import stats
    print('Calculating spatial span')
    
    #For all presynaptic neurons that togeter contact same postsynaptic neuron:
    pre_post_volumes = []
    pre_post_areas = []
    pre_post_diameters = []
    pre_post_diameters_projected = []
    pre_count = []
    pre_xzy_ls = []
    post_xzy_ls = []
    pre_center_ls = []
    num_pre_sites = []
    hull_ls = []
    pre_projected_points_ls = []
    
    #For individual presynaptic neurons:
    individual_pre_xzy_ls = []
    individual_post_xzy_ls = []
    individual_pre_center_ls = []
    individual_num_pre_sites = []
    individual_pre_post_volumes = []
    individual_pre_post_areas = []
    individual_pre_post_diameters = []
    individual_pre_post_diameters_projected = []
    individual_hull_ls = []
    individual_pre_count = []
    individual_curr_post = []
    individual_pre_projected_points_ls = []
    

    for i in range(0, len(up_to_date_post_ids)):
        curr_post = up_to_date_post_ids[i]

        # Getting single postynaptic cell's coordinates
        try:
            old_curr_post = update_df[update_df['new_id'] == curr_post]['old_id'].tolist()[0]
        except:
            old_curr_post = str(curr_post)

        single_post_coords = R_post_df[R_post_df['Updated_seg_id'] == old_curr_post]['XYZ-ME'].to_numpy(dtype=str, copy=True)
        post_xyz = np.zeros([np.shape(single_post_coords)[0], 3])
        new_post_coords = np.zeros([np.shape(single_post_coords)[0], 3])

        for idx, coordinate in enumerate(single_post_coords):
            post_xyz[idx, :] = np.array([coordinate.split(',')], dtype=float)
            new_post_coords[idx, :] = np.array([coordinate.split(',')], dtype=float)

        post_xyz *= [4, 4, 40]  # For plotting it using navis (correcting for data resolution)
        post_xzy_ls.append(post_xyz)
        

        # Getting presynaptic cells coordinates based on postsynaptic location
        curr_post_inputs = post_inputs[post_inputs['post_pt_root_id'] == curr_post].copy()

        
        ## For all presynaptic neurons that togeter contact same postsynaptic neuron:
        # Getting presynaptic cells coordinates based on postsynaptic location
        curr_pre_ls = pre_post_counts[pre_post_counts['post_pt_root_id'] == curr_post]['pre_pt_root_id'].tolist()
        curr_pre_inputs = pre_inputs[pre_inputs['post_pt_root_id'].isin(curr_pre_ls)].copy()

        if len(curr_pre_inputs) < 10:
            pre_post_volumes.append(None)
            pre_post_areas.append(None)
            pre_count.append(None)
            pre_xzy_ls.append(None)
            pre_center_ls.append(None)
            num_pre_sites.append(None)
            hull_ls.append(None)
            pre_post_diameters.append(None)
            pre_post_diameters_projected.append(None)
            pre_projected_points_ls.append(None)
        else:
            pre_count.append(len(curr_pre_ls))

            # Getting presynaptic cells coordinates
            temp_pre_coords = curr_pre_inputs['pre_pt_position'].tolist()

            # Correcting xyz positions for mesh plotting
            pre_xyz = np.array([list(np.array(l) * [4, 4, 40]) for l in temp_pre_coords])
            pre_xzy_ls.append(pre_xyz)
            num_pre_sites.append(len(pre_xyz))  # Total number of points in the presynaptic partner(s)

            # Calculate the center of the cloud of points
            pre_center = np.mean(pre_xyz, axis=0)
            pre_center_ls.append(pre_center)

            # Calculate the volume of the cloud using the convex hull method
            hull = ConvexHull(pre_xyz)
            volume = hull.volume
            pre_post_volumes.append(volume)
            
            # Calculate largest diameter
            largest_diameter = 0
            for simplex in hull.simplices:
                for i in range(len(simplex)):
                    for j in range(i+1, len(simplex)):
                        # Calculate distance between two points
                        d = distance.euclidean(pre_xyz[simplex[i]], pre_xyz[simplex[j]])
                        if d > largest_diameter:
                            largest_diameter = d

            # Convert largest diameter to micrometers
            largest_diameter_um = largest_diameter / 10**3
            pre_post_diameters.append(largest_diameter_um)

            # Calculate volume/area based on projections using PCA on presynaptic partner coordinates
            # PCA to get an approximate area of the volume
            pre_mean = np.mean(pre_xyz, axis=0)
            pre_centered_points = pre_xyz - pre_mean
            pre_cov_matrix = np.cov(pre_centered_points, rowvar=False)
            pre_eigenvalues, pre_eigenvectors = np.linalg.eigh(pre_cov_matrix)
            pre_normal_vector = pre_eigenvectors[:, [1, 2]]  # PC2 and PC3

            # Calculate volume/area based on projections using PCA on postsynaptic partner coordinates
            temp_post_coords = curr_post_inputs['pre_pt_position'].tolist()
            post_xyz = np.array([list(np.array(l) * [4, 4, 40]) for l in temp_post_coords])
            post_mean = np.mean(post_xyz, axis=0)
            post_centered_points = post_xyz - post_mean
            post_cov_matrix = np.cov(post_centered_points, rowvar=False)
            post_eigenvalues, post_eigenvectors = np.linalg.eigh(post_cov_matrix)
            post_normal_vector = post_eigenvectors[:, [0, 1]]  # PC1 and PC2

            # Project the points
            projected_points = pre_centered_points.dot(post_normal_vector)
            pre_projected_points_ls.append(projected_points)

            # Calculate area
            hull = ConvexHull(projected_points)
            hull_ls.append(hull)
            area = hull.volume  # Area is calculated as volume in 2D
            area_um2 = area / 10**6
            pre_post_areas.append(area_um2)
            
            # Calculate largest diameter
            largest_diameter = 0
            for simplex in hull.simplices:
                for i in range(len(simplex)):
                    for j in range(i+1, len(simplex)):
                        # Calculate distance between two points
                        d = distance.euclidean(projected_points[simplex[i]], projected_points[simplex[j]])
                        if d > largest_diameter:
                            largest_diameter = d

            # Convert largest diameter to micrometers
            largest_diameter_um = largest_diameter / 10**3
            pre_post_diameters_projected.append(largest_diameter_um)
            
            
        ## For individual presynaptic neurons:
        # Getting presynaptic cells coordinates based on postsynaptic location
        curr_pre_ls = pre_post_counts[pre_post_counts['post_pt_root_id'] == curr_post]['pre_pt_root_id'].tolist()
        for curr_pre in curr_pre_ls:
            individual_post_xzy_ls.append(post_xyz)
            individual_curr_post.append(curr_post)
            curr_pre_inputs = pre_inputs[pre_inputs['post_pt_root_id'].isin([curr_pre])].copy()
            if len(curr_pre_inputs) < 5:
                individual_pre_post_volumes.append(None)
                individual_pre_post_areas.append(None)
                individual_pre_count.append(None)
                individual_pre_xzy_ls.append(None)
                individual_pre_center_ls.append(None)
                individual_num_pre_sites.append(None)
                individual_hull_ls.append(None)
                individual_pre_post_diameters.append(None)
                individual_pre_post_diameters_projected.append(None)
                individual_pre_projected_points_ls.append(None)
            else:
                individual_pre_count.append(len([curr_pre]))

                # Getting presynaptic cells coordinates
                temp_pre_coords = curr_pre_inputs['pre_pt_position'].tolist()

                # Correcting xyz positions for mesh plotting
                pre_xyz = np.array([list(np.array(l) * [4, 4, 40]) for l in temp_pre_coords])
                individual_pre_xzy_ls.append(pre_xyz)
                individual_num_pre_sites.append(len(pre_xyz))  # Total number of points in the presynaptic partner(s)

                # Calculate the center of the cloud of points
                pre_center = np.mean(pre_xyz, axis=0)
                individual_pre_center_ls.append(pre_center)

                # Calculate the volume of the cloud using the convex hull method
                hull = ConvexHull(pre_xyz)
                volume = hull.volume
                individual_pre_post_volumes.append(volume)
                
                # Calculate largest diameter
                largest_diameter = 0
                for simplex in hull.simplices:
                    for i in range(len(simplex)):
                        for j in range(i+1, len(simplex)):
                            # Calculate distance between two points
                            d = distance.euclidean(pre_xyz[simplex[i]], pre_xyz[simplex[j]])
                            if d > largest_diameter:
                                largest_diameter = d

                # Convert largest diameter to micrometers
                largest_diameter_um = largest_diameter / 10**3
                individual_pre_post_diameters.append(largest_diameter_um)
                

                # Calculate volume/area based on projections using PCA on presynaptic partner coordinates
                # PCA to get an approximate area of the volume
                pre_mean = np.mean(pre_xyz, axis=0)
                pre_centered_points = pre_xyz - pre_mean
                pre_cov_matrix = np.cov(pre_centered_points, rowvar=False)
                pre_eigenvalues, pre_eigenvectors = np.linalg.eigh(pre_cov_matrix)
                pre_normal_vector = pre_eigenvectors[:, [1, 2]]  # PC2 and PC3

                # Calculate volume/area based on projections using PCA on postsynaptic partner coordinates
                temp_post_coords = curr_post_inputs['pre_pt_position'].tolist()
                post_xyz = np.array([list(np.array(l) * [4, 4, 40]) for l in temp_post_coords])
                post_mean = np.mean(post_xyz, axis=0)
                post_centered_points = post_xyz - post_mean
                post_cov_matrix = np.cov(post_centered_points, rowvar=False)
                post_eigenvalues, post_eigenvectors = np.linalg.eigh(post_cov_matrix)
                post_normal_vector = post_eigenvectors[:, [0, 1]]  # PC1 and PC2

                # Project the points
                projected_points = pre_centered_points.dot(post_normal_vector)
                individual_pre_projected_points_ls.append(projected_points)

                # Calculate area
                hull = ConvexHull(projected_points)
                individual_hull_ls.append(hull)
                area = hull.volume  # Area is calculated as volume in 2D
                area_um2 = area / 10**6
                individual_pre_post_areas.append(area_um2)
                
                # Calculate largest diameter
                largest_diameter = 0
                for simplex in hull.simplices:
                    for i in range(len(simplex)):
                        for j in range(i+1, len(simplex)):
                            # Calculate distance between two points
                            d = distance.euclidean(projected_points[simplex[i]], projected_points[simplex[j]])
                            if d > largest_diameter:
                                largest_diameter = d

                # Convert largest diameter to micrometers
                largest_diameter_um = largest_diameter / 10**3
                individual_pre_post_diameters_projected.append(largest_diameter_um)
                
                

            
            

    # Summary data frames
    spatial_span_df = pd.DataFrame()
    spatial_span_df['bodyId_post'] = up_to_date_post_ids
    spatial_span_df['Volume'] = pre_post_volumes
    spatial_span_df['Area'] = pre_post_areas
    spatial_span_df['Diameter'] = pre_post_diameters
    spatial_span_df['Diameter_projected'] = pre_post_diameters_projected
    spatial_span_df['Hull'] = hull_ls
    spatial_span_df['Pre_count'] = pre_count
    spatial_span_df['Pre_xyz'] = pre_xzy_ls
    spatial_span_df['Pre_center'] = pre_center_ls
    spatial_span_df['Post_xyz'] = post_xzy_ls
    spatial_span_df['pre_projected_points'] = pre_projected_points_ls
    spatial_span_df.set_index('bodyId_post', inplace=True)
    spatial_span_df['Area_zscore'] = (spatial_span_df['Area'] - spatial_span_df['Area'].mean()) / spatial_span_df['Area'].std()
    spatial_span_df['Num_pre_sites'] = num_pre_sites
    spatial_span_df['Num_columns'] = [round(area / single_column_area) if area is not None else None for area in pre_post_areas]
    spatial_span_df['Column_span'] = [round(diameter / single_column_diameter) if diameter is not None else None for diameter in pre_post_diameters]
    spatial_span_df['Column_span_projected'] = [round(diameter / single_column_diameter) if diameter is not None else None for diameter in pre_post_diameters_projected]
    
    individual_spatial_span_df = pd.DataFrame()
    individual_spatial_span_df['bodyId_post'] = individual_curr_post
    individual_spatial_span_df['Volume'] = individual_pre_post_volumes
    individual_spatial_span_df['Area'] = individual_pre_post_areas
    individual_spatial_span_df['Diameter'] = individual_pre_post_diameters
    individual_spatial_span_df['Diameter_projected'] = individual_pre_post_diameters_projected
    individual_spatial_span_df['Hull'] = individual_hull_ls
    individual_spatial_span_df['Pre_count'] = individual_pre_count
    individual_spatial_span_df['Pre_xyz'] = individual_pre_xzy_ls
    individual_spatial_span_df['Pre_center'] = individual_pre_center_ls
    individual_spatial_span_df['Post_xyz'] = individual_post_xzy_ls
    individual_spatial_span_df['pre_projected_points'] = individual_pre_projected_points_ls
    individual_spatial_span_df.set_index('bodyId_post', inplace=True)
    individual_spatial_span_df['Area_zscore'] = (individual_spatial_span_df['Area'] - individual_spatial_span_df['Area'].mean()) / individual_spatial_span_df['Area'].std()
    individual_spatial_span_df['Num_pre_sites'] = individual_num_pre_sites
    individual_spatial_span_df['Num_columns'] = [round(area / single_column_area) if area is not None else None for area in individual_pre_post_areas]
    individual_spatial_span_df['Column_span'] = [round(diameter / single_column_diameter) if diameter is not None else None for diameter in individual_pre_post_diameters]
    individual_spatial_span_df['Column_span_projected'] = [round(diameter / single_column_diameter) if diameter is not None else None for diameter in individual_pre_post_diameters_projected]

    return spatial_span_df, individual_spatial_span_df