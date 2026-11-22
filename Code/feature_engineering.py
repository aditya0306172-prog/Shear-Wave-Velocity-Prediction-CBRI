import pandas as pd
import numpy as np

def compute_physics_features(df_path, output_path):
    print(f"Loading data from {df_path}")
    df = pd.read_csv(df_path)
    
    # Sort just in case
    df = df.sort_values(by=['Station_ID', 'Depth_m']).reset_index(drop=True)
    
    # 1. Fines Content (FC) estimation
    # Based on Lithology typical mapping
    def estimate_fc(lithology):
        lithology = str(lithology).upper()
        if lithology in ['C', 'M', 'V', 'O', 'PT']: # Clay, Silt, Volcanic, Organic, Peat
            return 80.0
        elif 'F' in lithology and lithology != 'F': # SF (Sandy soil), GF (Gravelly soil)
            return 40.0
        elif lithology == 'S': # Sand
            return 10.0
        else: # G, F, R
            return 0.0
            
    df['FC'] = df['Lithology_Code'].apply(estimate_fc)
    
    # 2. qc (CPT Cone Resistance) estimation (MPa)
    # Empirical relation: qc = k * N60
    # k ≈ 0.2 for Clay, 0.4 for Sand, 0.6 for Gravel. We interpolate based on FC.
    def estimate_qc(row):
        n60 = row['N_Value'] # Assume N_Value = N60
        fc = row['FC']
        
        if fc >= 80:
            k = 0.2
        elif fc <= 10:
            k = 0.4
        elif fc == 0:
            k = 0.6
        else:
            # Interpolate between sand (10%) and clay (80%)
            # fc=10 -> 0.4, fc=80 -> 0.2
            k = 0.4 - ((fc - 10) / 70.0) * 0.2
            
        return k * n60
        
    df['qc_MPa'] = df.apply(estimate_qc, axis=1)
    
    # Group by station to compute depth-dependent features
    stations = df['Station_ID'].unique()
    
    sigma_v_list = []
    sigma_v_eff_list = []
    void_ratio_list = []
    
    print(f"Processing {len(stations)} stations for stress and void ratio...")
    
    for station in stations:
        st_df = df[df['Station_ID'] == station].copy()
        
        # Find water table depth (first depth where Vp >= 1500)
        saturated = st_df[st_df['Vp_m_s'] >= 1500]
        if len(saturated) > 0:
            zw = saturated['Depth_m'].min()
        else:
            zw = st_df['Depth_m'].max() + 10 # Assume dry
            
        cumulative_stress = 0.0
        prev_depth = 0.0
        
        for idx, row in st_df.iterrows():
            depth = row['Depth_m']
            dz = depth - prev_depth
            rho = row['Density_g_cm3']
            vp = row['Vp_m_s']
            
            # 1. Total stress = sum(rho * g * dz). rho in g/cm3 = t/m3. Stress in kPa.
            d_sigma = rho * 9.81 * dz
            cumulative_stress += d_sigma
            sigma_v_list.append(cumulative_stress)
            
            # 2. Pore water pressure u
            if depth > zw:
                # Hydrostatic pressure from water table to current layer midpoint/bottom
                # Roughly u = (depth - zw) * 1.0 * 9.81
                u = (depth - zw) * 1.0 * 9.81
            else:
                u = 0.0
                
            sigma_v_eff = cumulative_stress - u
            sigma_v_eff_list.append(sigma_v_eff)
            
            # 3. Void ratio e
            # Gs ≈ 2.65
            if vp >= 1500: # saturated, Sr = 1
                # rho = (2.65 + e) / (1 + e) => e = (2.65 - rho) / (rho - 1)
                if rho > 1.05:
                    e = (2.65 - rho) / (rho - 1.0)
                else:
                    e = 2.0 # Bound to avoid infinity
            else: # dry or partially saturated, assume Sr=0.5
                # rho = (2.65 + 0.5*e) / (1 + e) => rho + rho*e = 2.65 + 0.5*e => e(rho - 0.5) = 2.65 - rho
                if rho > 0.6:
                    e = (2.65 - rho) / (rho - 0.5)
                else:
                    e = 2.0
            
            # Bound e
            e = max(0.1, min(e, 3.0))
            void_ratio_list.append(e)
            
            prev_depth = depth
            
    df['Sigma_v_kPa'] = sigma_v_list
    df['Sigma_v_eff_kPa'] = sigma_v_eff_list
    df['Void_Ratio_e'] = void_ratio_list
    
    print(f"Saving enriched dataset to {output_path}")
    df.to_csv(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    compute_physics_features("clean_kiknet_dataset.csv", "clean_kiknet_dataset_physics.csv")
