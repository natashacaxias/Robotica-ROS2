import pandas as pd
import numpy as np
import os
import glob
import json

def get_metrics():
    base_dir = 'c:/dqn/atual/logs'
    scenarios = ['easy', 'medium', 'hard']
    results = {}
    
    for sc in scenarios:
        files = glob.glob(os.path.join(base_dir, sc, '*.csv'))
        if not files:
            continue
        
        # Read the CSV proper
        df = pd.read_csv(files[0])
        
        # Last step values
        last_row = df.iloc[-1]
        Lx = float(last_row['leader_x'])
        Ly = float(last_row['leader_y'])
        tx = float(last_row['goal_x'])
        ty = float(last_row['goal_y'])
        
        final_dist = np.hypot(tx - Lx, ty - Ly)
        
        # Steps and Time
        steps = len(df)
        time_s = steps * 0.05
        
        # Initial step values
        first_row = df.iloc[0]
        iLx = float(first_row['leader_x'])
        iLy = float(first_row['leader_y'])
        itx = float(first_row['goal_x'])
        ity = float(first_row['goal_y'])
        
        initial_dist = np.hypot(itx - iLx, ity - iLy)
        progress = initial_dist - final_dist
        
        results[sc.capitalize()] = {
            'Steps': steps,
            'Time_s': round(time_s, 2),
            'Final_Dist_m': round(final_dist, 3),
            'Init_Dist_m': round(initial_dist, 3),
            'Progress_m': round(progress, 3)
        }
        
    print(json.dumps(results, indent=2))

get_metrics()
