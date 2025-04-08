import numpy as np

def geodetic_to_ecef(latitude, longitude, altitude=0.0):
    """Convert latitude/longitude/altitude(HAE) to ECEF Coordinate Frame"""
    latitude = np.radians(latitude)
    longitude = np.radians(longitude)
    a = 6378.137 * 1000
    e_squared = 0.00669437999013
    N = a / np.sqrt(1 - (e_squared * np.sin(latitude) ** 2))
    x = (N + altitude) * np.cos(latitude) * np.cos(longitude)
    y = (N + altitude) * np.cos(latitude) * np.sin(longitude)
    z = (N * (1 - e_squared) + altitude) * np.sin(latitude)
    return np.array([x, y, z], dtype=float)

def ecef_to_geodetic(x, y, z):
    """Convert ECEF Coorindates to Geodetic (Lat, Lon, Altitude (HAE))"""
    epsilon_1 = 1e-6
    epsilon_2 = 1e-6 
    a = 6378.137 * 1000
    b = 6356.752 * 1000
    e_squared = 0.00669437999013
    N = a

    H = np.linalg.norm([x, y, z]) - np.sqrt(a * b)
    B = np.arctan2(z, np.linalg.norm([x, y]) * (1 - ((e_squared * N) / (N + H))))
    while True:
        N_i = a / np.sqrt(1 - e_squared * np.sin(B) ** 2)
        H_i = (np.linalg.norm([x, y]) / np.cos(B)) - N_i
        B_i = np.arctan2(z, np.linalg.norm([x, y]) * (1 - ((e_squared * N_i) / (N_i + H_i))))
        if (np.abs(H_i - H) < epsilon_1) and (np.abs(B_i - B) < epsilon_2):
            break 
        N = N_i
        H = H_i
        B = B_i

    lon = np.rad2deg(np.arctan2(y, x))
    lat = np.rad2deg(B)
    alt = H
    return lat, lon, alt # Reutrns in degrees

def ecef_to_topocentric(target_ecef, observer_ecef, observer_latitude, observer_longitude):
    """Given an observer location convert target coordinates from ECEF to Topocentric (Observer perspective coords)"""
    lat_rads = np.radians(observer_latitude)
    lon_rads = np.radians(observer_longitude)
    R = np.array([
        [-np.sin(lon_rads),                     np.cos(lon_rads),                      0],
        [-np.sin(lat_rads)*np.cos(lon_rads),    -np.sin(lat_rads)*np.sin(lon_rads),    np.cos(lat_rads)],
        [ np.cos(lat_rads)*np.cos(lon_rads),     np.cos(lat_rads)*np.sin(lon_rads),     np.sin(lat_rads)]
    ])
    observer_to_target = target_ecef - observer_ecef
    target_topo = R @ observer_to_target
    return target_topo

def topocentric_to_ecef(target_topo, observer_ecef, observer_latitude, observer_longitude):
    """Given Topocentric Coordinates of the target from the perspective of the observer convert back to ECEF coordinates"""
    lat_rads = np.radians(observer_latitude)
    lon_rads = np.radians(observer_longitude)
    R = np.array([
        [-np.sin(lon_rads),                     np.cos(lon_rads),                      0],
        [-np.sin(lat_rads)*np.cos(lon_rads),    -np.sin(lat_rads)*np.sin(lon_rads),    np.cos(lat_rads)],
        [ np.cos(lat_rads)*np.cos(lon_rads),     np.cos(lat_rads)*np.sin(lon_rads),     np.sin(lat_rads)]
    ]).T
    return (R @ target_topo) + observer_ecef