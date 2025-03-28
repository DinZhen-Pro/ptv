em1_params = {
    "max_iters": 50,  # Maximum iterations of EM
    "sigma": 2,  # Covariance of target point Gaussians
    "min_err": 1.0,  # Minimum change in error before optimization stops
    "max_distance": 1.0,  # Max distance for range search in EM1
    "centered": 4,  # Centering parameter
}

# Main parameters
params = {
    "em1_params": em1_params,  # Nested dictionary for EM1 parameters
    "window_size": 15,  # Window size
    "window_mag": 1.2,  # Window search size
    "min_events_for_em": 20,  # Minimum events for EM
    "max_events_per_window": 100000,  # Max events in one time window
    "min_distance": 4,  # Minimum distance
}