import numpy as np

class UnitConverter:
    """
    A basic unit-conversion helper class. 
    Supports scalar, list, and numpy array inputs.
    """

    FEET_TO_METERS = 0.3048
    METERS_TO_FEET = 1 / FEET_TO_METERS
    
    NM_TO_METERS = 1852.0
    METERS_TO_NM = 1 / NM_TO_METERS

    DEG_TO_RAD = np.pi / 180.0
    RAD_TO_DEGREE = 1 / DEG_TO_RAD

    @staticmethod
    def feet_to_meters(value):
        arr_value = np.asarray(value, dtype=float)
        converted = arr_value * UnitConverter.FEET_TO_METERS
        return converted.item() if np.isscalar(value) else converted

    @staticmethod
    def meters_to_feet(value):
        arr_value = np.asarray(value, dtype=float)
        converted = arr_value * UnitConverter.METERS_TO_FEET

        return converted.item() if np.isscalar(value) else converted

    @staticmethod
    def nm_to_meters(value):
        arr_value = np.asarray(value, dtype=float)
        converted = arr_value * UnitConverter.NM_TO_METERS

        return converted.item() if np.isscalar(value) else converted

    @staticmethod
    def meters_to_nm(value):
        arr_value = np.asarray(value, dtype=float)
        converted = arr_value * UnitConverter.METERS_TO_NM

        return converted.item() if np.isscalar(value) else converted

    @staticmethod
    def deg_to_rad(value):
        arr_value = np.asarray(value, dtype=float)
        converted = arr_value * UnitConverter.DEG_TO_RAD

        return converted.item() if np.isscalar(value) else converted

    @staticmethod
    def rad_to_deg(value):
        arr_value = np.asarray(value, dtype=float)
        converted = arr_value * UnitConverter.RAD_TO_DEGREE

        return converted.item() if np.isscalar(value) else converted
