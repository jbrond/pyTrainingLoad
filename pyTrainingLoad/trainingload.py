import numpy as np
import pandas as pd

def bTRIM(tcx_df, restHR, maxHR, gender):
    """
    Calculates the Banister Training Impulse (TRIM) based on heart rate data.

    The Banister TRIM is a method to quantify training load by combining 
    exercise duration and relative heart rate intensity. It accounts for 
    gender-specific physiological differences.

    Args:
        tcx_df (pandas.DataFrame): A DataFrame with at least two columns:
            - 'time' (numeric): Timestamps in seconds or consistent time units.
            - 'heart_rate' (numeric): Heart rate values in beats per minute (BPM).
        restHR (float): Resting heart rate (in BPM).
        maxHR (float): Maximum heart rate (in BPM).
        gender (int): 0 for male, 1 for female. Affects the exponential weight factor.

    Returns:
        float: Estimated training load (TRIM) value.
    """
    D = float(tcx_df['time'].iloc[-1]) - float(tcx_df['time'].iloc[0])
    mhr = tcx_df['heart_rate'].mean(skipna=True)
    deltaHR = (mhr - restHR) / (maxHR - restHR)
    b = 1.92 if gender == 0 else 1.67
    Y = np.exp(b * deltaHR)
    tl = D * deltaHR * Y
    return tl

def eTRIM(tcx_df, restHR, maxHR, gender):
    """
    Calculates the Edwards Training Impulse (eTRIM) based on heart rate intensity zones.

    The Edwards TRIM method estimates training load by categorizing time spent 
    in various heart rate zones. Each zone has a different intensity factor, 
    and the total training load is a weighted sum of time spent in each zone.

    Args:
        tcx_df (pandas.DataFrame): A DataFrame containing heart rate data, with:
            - 'time' (numeric): Timestamps (in seconds or consistent time units).
            - 'heart_rate' (numeric): Heart rate values in beats per minute (BPM).
        restHR (float): Resting heart rate (in BPM).
        maxHR (float): Maximum heart rate (in BPM).
        gender (int): Not used in this implementation, but kept for consistency with similar functions.

    Returns:
        int: Total training load (TRIM score) based on Edwards intensity distribution.
    """
    time = tcx_df['time'].astype(float) - float(tcx_df['time'].iloc[0])
    time2 = np.arange(time.iloc[0], time.iloc[-1]+1)
    # Interpolate heart rate to each second
    yi = np.interp(time2, time, tcx_df['heart_rate'])
    yirel = (yi - restHR) / (maxHR - restHR) * 100
    # Intensity zones
    bins = [0, 50, 60, 70, 80, 90, 1000]
    counts, _ = np.histogram(yirel, bins)
    tl = np.sum(counts[1:6] * np.array([1, 2, 3, 4, 5]))
    return tl

def playerLoad(ax3):
    """
    Calculates external training load using PlayerLoad from triaxial accelerometer data.

    PlayerLoad is a commonly used metric in sports science to estimate external 
    load by summing the rate of change in acceleration across all three axes (x, y, z).
    This implementation calculates PlayerLoad using the square root of the sum of 
    squared differences between consecutive samples.

    Args:
        ax3 (dict): A dictionary containing accelerometer data, with:
            - 'data' (pandas.DataFrame): DataFrame with columns 'x', 'y', 'z',
              representing acceleration in each axis (usually in g or m/s²).

    Returns:
        float: Estimated PlayerLoad value representing total external load.
    """
    dax = np.diff(ax3['data']['x']) ** 2
    day = np.diff(ax3['data']['y']) ** 2
    daz = np.diff(ax3['data']['z']) ** 2
    tl = np.sum(np.sqrt((dax + day + daz) / 100))
    return tl

# ...existing code...

def eTRIMspecial(tcx_df, restHR, maxHR, gender):
    """
    Calculates a modified Edwards Training Impulse (eTRIMspecial) using six intensity zones.

    This special version of the Edwards TRIM method includes time spent in the 
    lowest heart rate zone (<50% HR reserve) and assigns it a non-zero weight. 
    The method divides heart rate relative intensity into six zones, each with 
    increasing weighting factors from 1 to 6.

    Args:
        tcx_df (pandas.DataFrame): A DataFrame containing heart rate data with:
            - 'time' (numeric): Timestamps (in seconds or consistent units).
            - 'heart_rate' (numeric): Heart rate values in beats per minute (BPM).
        restHR (float): Resting heart rate (in BPM).
        maxHR (float): Maximum heart rate (in BPM).
        gender (int): Not used in this implementation, included for consistency.

    Returns:
        int: Total training load (TRIM score) based on the 6-zone intensity distribution.
    """
    time = tcx_df['time'].astype(float) - float(tcx_df['time'].iloc[0])
    time2 = np.arange(time.iloc[0], time.iloc[-1]+1)
    yi = np.interp(time2, time, tcx_df['heart_rate'])
    yirel = (yi - restHR) / (maxHR - restHR) * 100
    bins = [0, 50, 60, 70, 80, 90, 1000]
    counts, _ = np.histogram(yirel, bins)
    tl = np.sum(counts[0:6] * np.array([1, 2, 3, 4, 5, 6]))
    return tl

def playerLoad2(ax3, frequency):
    """
    Calculates external training load (PlayerLoad) from filtered triaxial accelerometer data.

    This version applies a low-pass Butterworth filter to reduce high-frequency noise 
    before calculating PlayerLoad. The metric is computed by summing the square root 
    of the squared differences in acceleration across all three axes.

    Args:
        ax3 (dict): A dictionary containing accelerometer data with:
            - 'data' (pandas.DataFrame): DataFrame with columns 'x', 'y', 'z',
              representing raw acceleration in each axis (typically in g or m/s²).
        frequency (float): Sampling frequency of the accelerometer data in Hz.

    Returns:
        float: Filtered PlayerLoad value representing total external load.
    """
    from scipy.signal import butter, filtfilt
    b, a = butter(4, 10/(frequency/2), btype='low')
    x_filt = filtfilt(b, a, ax3['data']['x'])
    y_filt = filtfilt(b, a, ax3['data']['y'])
    z_filt = filtfilt(b, a, ax3['data']['z'])
    dax = np.diff(x_filt) ** 2
    day = np.diff(y_filt) ** 2
    daz = np.diff(z_filt) ** 2
    tl = np.sum(np.sqrt(dax + day + daz) / 100)
    return tl

def playerLoadMAG(ax3):
    """
    Calculates external training load using the magnitude of raw triaxial accelerometer data.

    This method computes the vector magnitude (Euclidean norm) of acceleration values 
    at each time point and sums them across the entire session.

    Args:
        ax3 (dict): A dictionary with:
            - 'data' (pandas.DataFrame): DataFrame with 'x', 'y', 'z' columns for raw acceleration.

    Returns:
        float: Total training load as the sum of vector magnitudes.
    """
    dax = ax3['data']['x'] ** 2
    day = ax3['data']['y'] ** 2
    daz = ax3['data']['z'] ** 2
    tl = np.sum(np.sqrt(dax + day + daz))
    return tl

def playerLoadSUM(ax3):
    """
    Calculates external training load by summing absolute values of acceleration in each axis.

    This method is a simple load estimate based on the sum of the absolute acceleration 
    values for each axis, treating each axis independently.

    Args:
        ax3 (dict): A dictionary with:
            - 'data' (pandas.DataFrame): DataFrame with 'x', 'y', 'z' columns for raw acceleration.

    Returns:
        float: Total training load as the sum of absolute accelerations.
    """
    dax = np.abs(ax3['data']['x'])
    day = np.abs(ax3['data']['y'])
    daz = np.abs(ax3['data']['z'])
    tl = np.sum(dax + day + daz)
    return tl

def impulseLoad(ax3):
    """
    Calculates external training load using impulse load based on vector magnitude.

    Impulse Load estimates external work by summing the vector magnitude of acceleration, 
    after filtering out near-zero values. The values are normalized by gravitational 
    acceleration (9.8067 m/s²) to approximate body-load-related impulses.

    Args:
        ax3 (dict): A dictionary with:
            - 'data' (pandas.DataFrame): DataFrame with 'x', 'y', 'z' columns for raw acceleration.

    Returns:
        float: Total impulse-based load estimate.
    """
    dax = ax3['data']['x'] ** 2
    day = ax3['data']['y'] ** 2
    daz = ax3['data']['z'] ** 2
    vm = np.sqrt(dax + day + daz)
    vm = vm[vm > 0.1]
    tl = np.sum(vm / 9.8067)
    return tl

def accelRate(ax3):
    """
    Calculates external training load as the accumulated rate of change in acceleration magnitude.

    This method estimates load by measuring how quickly total acceleration (vector magnitude) 
    changes over time, which reflects movement dynamics and intensity.

    Args:
        ax3 (dict): A dictionary with:
            - 'data' (pandas.DataFrame): DataFrame with 'x', 'y', 'z' columns for raw acceleration.

    Returns:
        float: Accumulated rate of acceleration change (acceleration rate load).
    """
    arr = ax3['data'][['x', 'y', 'z']].to_numpy()
    vm = np.sqrt(np.sum(arr ** 2, axis=1))
    tl = np.sum(np.abs(np.diff(vm)))
    return tl

def velocityLoad(ax3, frequency):
    """
    Calculates external training load using filtered ENMO (Euclidean Norm Minus One) with a moving average window.

    This method estimates load by:
      - Calculating the Euclidean norm (vector magnitude) of triaxial acceleration data.
      - Subtracting 1 g (gravity) to obtain ENMO, a metric sensitive to movement.
      - Applying a band-pass Butterworth filter (0.01–10 Hz) to isolate meaningful movement signals.
      - Computing the absolute value of the filtered signal.
      - Applying a moving average over a 3-second window (based on sampling frequency).
      - Downsampling and summing the averaged segments to get a total load estimate.

    Args:
        ax3 (dict): Dictionary with:
            - 'data' (pandas.DataFrame): DataFrame containing columns 'x', 'y', 'z' with raw acceleration.
        frequency (float): Sampling frequency of the accelerometer in Hz.

    Returns:
        float: Total velocity-based training load estimate.
    """
    from scipy.signal import butter, filtfilt
    from pandas import Series
    arr = ax3['data'][['x', 'y', 'z']].to_numpy()
    vm = np.sqrt(np.sum(arr ** 2, axis=1))
    wnds = int(frequency * 3)
    b, a = butter(4, [0.01/(frequency/2), 10/(frequency/2)], btype='band')
    vmf = filtfilt(b, a, vm - 1)
    # Moving average (left-aligned)
    vmf_ma = pd.Series(np.abs(vmf)).rolling(window=wnds, min_periods=1).mean()
    vmf_ma = vmf_ma.iloc[::wnds].to_numpy()
    tl = np.sum(vmf_ma)
    return tl

