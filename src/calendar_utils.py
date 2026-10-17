"""Holiday / festive-season calendar shared by the simulator and the features."""
import pandas as pd


def holiday_flag(dates):
    """1 on festive-season days (mid-Oct..mid-Nov, Sankranti) and national holidays."""
    d = pd.DatetimeIndex(dates)
    m, day = d.month, d.day
    festive = ((m == 10) & (day >= 15)) | ((m == 11) & (day <= 15))
    sankranti = (m == 1) & (day >= 10) & (day <= 16)
    national = (((m == 1) & (day == 26)) | ((m == 8) & (day == 15))
                | ((m == 10) & (day == 2)) | ((m == 12) & (day == 25)))
    return (festive | sankranti | national).astype(int)
