# -*- coding: utf-8 -*-
import numpy as np
from scipy.constants import g as gravity

import warnings

try:
    from numpy import any as _any
except ImportError:
    def _any(arg):
        if arg is True:
            return True
        if arg is False:
            return False
        return any(arg)

def water_density(T=None, T0=None, units=None, a=None,
                  just_return_a=False, warn=True):
    """
    Density of water (kg/m3) as function of temperature (K)
    according to VSMOW model between 0 and 40 degree Celsius.
    Fitted using Thiesen's equation.

    Parameters
    ----------
    T: float
        Temperature (in Kelvin) (default: 298.15)
    T0: float
        Value of T for 0 degree Celsius (default: 273.15)
    units: object (optional)
        object with attributes: Kelvin, meter, kilogram
    a: array_like (optional)
        5 parameters to the equation.
    just_return_a: bool (optional, default: False)
        Do not compute rho, just return the parameters ``a``.
    warn: bool (default: True)
        Emit UserWarning when outside temperature range.

    Returns
    -------
    Density of water (float of kg/m3 if T is float and units is None)

    Examples
    --------
    >>> print('%.2f' % water_density(277.13))
    999.97

    References
    ----------
    TANAKA M., GIRARD G., DAVIS R., PEUTO A. and BIGNELL N.,
        "Recommanded table for the density of water between 0 °C and 40 °C
        based on recent experimental reports",
        Metrologia, 2001, 38, 301-309.
        http://iopscience.iop.org/article/10.1088/0026-1394/38/4/3
        doi:10.1088/0026-1394/38/4/3
    """
    if units is None:
        K = 1
        m = 1
        kg = 1
    else:
        K = units.Kelvin
        m = units.meter
        kg = units.kilogram
    if T is None:
        T = 298.15*K
    m3 = m**3
    if a is None:
        a = (-3.983035*K,  # C
             301.797*K,  # C
             522528.9*K*K,  # C**2
             69.34881*K,  # C
             999.974950*kg/m3)
    if just_return_a:
        return a
    if T0 is None:
        T0 = 273.15*K
    t = T - T0
    if warn and (_any(t < 0*K) or _any(t > 40*K)):
        warnings.warn("Temperature is outside range (0-40 degC)")
    return a[4]*(1-((t + a[0])**2*(t + a[1]))/(a[2]*(t + a[3])))


def celsius_to_kelvin(t_celsius):
    return t_celsius+273.15

def spillway_powerhouse_flows(q_s,b1,b3,q_p):
    """
    Weighted contribution of spillway plus powerhouse entrainment flows 
    (unitless) for adjusting dependency.

    Parameters
    ----------
    q_s: float
         Spillway Flows (m^3/s)
    b1: float
        Optimized coefficient unique to each dam (unitless)
    
    b3: float
        Optimized coefficient unique to each dam (unitless)
        
    q_p: float
         Powerhouse Flows (m^3/s)
        
    Returns
    -------
    Weighted contribution of spillway plus powerhouse entrainment flows 
    (unitless) for adjusting dependency.

    Examples
    --------
    

    References
    ----------
    Stewart, Kevin M., Adam Witt, and Boualem Hadjerioua. 
    "Total dissolved gas prediction and optimization in RIVERWARE." 
    Prepared for US Department of Energy Wind and Water Program by 
    Oakridge National Laboratory, Oak Ridge, TN (2015).
    
    https://info.ornl.gov/sites/publications/files/Pub59285.pdf
    """
    
    return ((q_s+b1*q_s+b3)/(q_s+q_p))



def tailwater(h_t, temp_water, p_atm):
    """
    TDG’s dependency on tailwater depth as referenced to atmospheric 
    pressure .
    (unitless)
    Parameters
    ----------
    h_t: float
         tailwater depth (m)
         
    temp_water: float
        Water temperature (C)
    
    p_atm: float
        atmospheric pressure (kg/ms^2)
            
        
        
    Returns
    -------
    TDG’s dependency on tailwater depth as referenced to atmospheric 
    pressure .
    (unitless)

    Examples
    --------
    

    References
    ----------
    Stewart, Kevin M., Adam Witt, and Boualem Hadjerioua. 
    "Total dissolved gas prediction and optimization in RIVERWARE." 
    Prepared for US Department of Energy Wind and Water Program by 
    Oakridge National Laboratory, Oak Ridge, TN (2015).
    
    https://info.ornl.gov/sites/publications/files/Pub59285.pdf
    """
    t_water_kelvin = celsius_to_kelvin(temp_water)
    rho = water_density(t_water_kelvin)
    g = gravity
    return (1+(rho*g*h_t)/(2*p_atm))

def powerhouse_entrainment(q_p,b1,q_s,b3):
    """
    Weighted contributions of the difference between the powerhouse and its 
    entrainment flows used to adjust the fraction of forebay TDG transferred 
    downstream. (unitless)
    Parameters
    ----------
    
    q_s: float
         Spillway Flows (m^3/s)
    b1: float
        Optimized coefficient unique to each dam (unitless)
    
    b3: float
        Optimized coefficient unique to each dam (unitless)
        
    q_p: float
         Powerhouse Flows (m^3/s)
        
        
    Returns
    -------
    Weighted contributions of the difference between the powerhouse and its 
    entrainment flows used to adjust the fraction of forebay TDG transferred 
    downstream. (unitless)

    Examples
    --------
    

    References
    ----------
    Stewart, Kevin M., Adam Witt, and Boualem Hadjerioua. 
    "Total dissolved gas prediction and optimization in RIVERWARE." 
    Prepared for US Department of Energy Wind and Water Program by 
    Oakridge National Laboratory, Oak Ridge, TN (2015).
    
    https://info.ornl.gov/sites/publications/files/Pub59285.pdf
    """
    return ((q_p-b1*q_s-b3)/(q_s+q_p))

def tailwater_tdg(q_s,q_p,x, h_t,temp_water,p_atm,tdg_f):
    """
    Tailwater TDG (%)
    
    Parameters
    ----------
    
    q_p: float
         Powerhouse Flows (m^3/s)
         
    q_s: float
         Spillway Flows (m^3/s)
         
    x:  list
         Optimized coefficients (b1,b2,b3) unique to each dam (unitless)
    
    h_t: float
         tailwater depth (m)
         
    temp_water: float
        Water temperature (C)
    
    p_atm: float
        atmospheric pressure (kg/ms^2)
    
    tdg_f: Forebay TDG (%)
            
        
    Returns
    -------
    
    Tailwater TDG (%)

    Examples
    --------
    

    References
    ----------
    Stewart, Kevin M., Adam Witt, and Boualem Hadjerioua. 
    "Total dissolved gas prediction and optimization in RIVERWARE." 
    Prepared for US Department of Energy Wind and Water Program by 
    Oakridge National Laboratory, Oak Ridge, TN (2015).
    
    https://info.ornl.gov/sites/publications/files/Pub59285.pdf
    """
    b1,b2,b3 = x
    A = spillway_powerhouse_flows(q_s,b1,b3,q_p)
    B = tailwater(h_t, temp_water, p_atm)
    C = powerhouse_entrainment(q_p,b1,q_s,b3)
    
    return 100*A*B*b2+tdg_f*C

def q_ge(q_p,q_s,b1,b3):
    """
    The volume of powerhouse flow entrained into the spillway (m^3/s)
    
    Parameters
    ----------
    
    q_p: float
         Powerhouse Flows (m^3/s)
         
    q_s: float
         Spillway Flows (m^3/s)
         
    b1: float
        Optimized coefficient unique to each dam (unitless)
    
    b3: float
        Optimized coefficient unique to each dam (unitless)
         
        
    Returns
    -------
    
    The volume of powerhouse flow entrained into the spillway (m^3/s)

    Examples
    --------
    

    References
    ----------
    Stewart, Kevin M., Adam Witt, and Boualem Hadjerioua. 
    "Total dissolved gas prediction and optimization in RIVERWARE." 
    Prepared for US Department of Energy Wind and Water Program by 
    Oakridge National Laboratory, Oak Ridge, TN (2015).
    
    https://info.ornl.gov/sites/publications/files/Pub59285.pdf
    """
    
    
    return min(q_p,(b1*q_s+b3))