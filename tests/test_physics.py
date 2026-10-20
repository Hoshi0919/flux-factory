"""
Physics Engine Verification Tests for Flux Factory.
Validates 2D field interactions, Lorentz deflections, damping, conservation laws,
fusion kinematics, and layout serialization.
"""
import math
import json
import base64
import pytest

# 1. Physical Constants and Models
EPSILON_SQ = 400.0  # Softening factor to prevent singularity at r -> 0

def calc_gravity_accel(px, py, nx, ny, power, radius=180.0):
    dx = nx - px
    dy = ny - py
    dist = math.hypot(dx, dy)
    if dist < 1e-4 or dist > radius:
        return 0.0, 0.0
    factor = power / (dist * dist + EPSILON_SQ)
    return (dx / dist) * factor, (dy / dist) * factor

def calc_repulsor_accel(px, py, nx, ny, power, radius=140.0):
    dx = px - nx
    dy = py - ny
    dist = math.hypot(dx, dy)
    if dist < 1e-4 or dist > radius:
        return 0.0, 0.0
    factor = power / (dist * dist + 200.0)
    return (dx / dist) * factor, (dy / dist) * factor

def calc_lorentz_accel(vx, vy, charge, b_field, dist, radius=120.0):
    """
    F = q * (v x B). In 2D plane:
    ax = q * vy * B
    ay = -q * vx * B
    Zero work theorem: ax * vx + ay * vy == 0 (speed magnitude conserved)
    """
    if dist > radius:
        return 0.0, 0.0
    falloff = (1.0 - dist / radius)
    effective_b = b_field * falloff
    ax = charge * vy * effective_b
    ay = -charge * vx * effective_b
    return ax, ay

def calc_damping(vx, vy, damp_coeff=0.08):
    return -vx * damp_coeff, -vy * damp_coeff

def calc_booster_accel(angle, power=1.5):
    return math.cos(angle) * power, math.sin(angle) * power


# 2. Test Cases
def test_gravity_softening_and_direction():
    # Test particle approaching from left
    ax, ay = calc_gravity_accel(px=0, py=100, nx=100, ny=100, power=200)
    assert ax > 0.0, "Gravity must pull towards positive x"
    assert abs(ay) < 1e-6, "No lateral acceleration on symmetric axis"
    
    # Test softening prevents division by zero
    ax_zero, ay_zero = calc_gravity_accel(px=100, py=100, nx=100, ny=100, power=200)
    assert ax_zero == 0.0 and ay_zero == 0.0

def test_repulsor_opposes_gravity():
    # Particle at (50, 100), node at (100, 100)
    gx, _ = calc_gravity_accel(50, 100, 100, 100, power=200)
    rx, _ = calc_repulsor_accel(50, 100, 100, 100, power=200)
    assert gx > 0, "Gravity pulls toward node"
    assert rx < 0, "Repulsor pushes away from node"

def test_lorentz_zero_work_and_charge_separation():
    vx, vy = 3.0, 0.0
    b_field = 0.5
    dist = 50.0
    
    # Positive charge (+1: Alpha)
    ax_alpha, ay_alpha = calc_lorentz_accel(vx, vy, charge=1.0, b_field=b_field, dist=dist)
    # Negative charge (-1: Beta)
    ax_beta, ay_beta = calc_lorentz_accel(vx, vy, charge=-1.0, b_field=b_field, dist=dist)
    # Neutral charge (0: Gamma)
    ax_gamma, ay_gamma = calc_lorentz_accel(vx, vy, charge=0.0, b_field=b_field, dist=dist)
    
    # Work done: dot product F . v
    work_alpha = ax_alpha * vx + ay_alpha * vy
    assert abs(work_alpha) < 1e-9, "Lorentz force must do exactly zero work (energy conservation)"
    
    # Opposite charges deflect in opposite lateral directions
    assert ay_alpha < 0.0, "Positive charge deflections downwards (clockwise)"
    assert ay_beta > 0.0, "Negative charge deflections upwards (counter-clockwise)"
    assert abs(ay_alpha + ay_beta) < 1e-9, "Equal and opposite lateral deflection"
    
    # Neutral particle does not deflect
    assert ax_gamma == 0.0 and ay_gamma == 0.0, "Neutral gamma particle unaffected by magnetic field"

def test_damping_reduces_speed():
    vx, vy = 5.0, 5.0
    initial_speed = math.hypot(vx, vy)
    ax, ay = calc_damping(vx, vy, damp_coeff=0.1)
    new_vx = vx + ax
    new_vy = vy + ay
    new_speed = math.hypot(new_vx, new_vy)
    assert new_speed < initial_speed, "Dampener must reduce kinetic velocity"

def test_booster_accelerates_in_vector_direction():
    angle = math.pi / 4.0  # 45 degrees
    ax, ay = calc_booster_accel(angle, power=2.0)
    assert math.isclose(ax, ay, rel_tol=1e-5), "45 deg booster gives equal x and y acceleration"
    assert math.isclose(math.hypot(ax, ay), 2.0, rel_tol=1e-5), "Magnitude matches power"

def test_collider_fusion_charge_and_momentum():
    """
    In the Tokamak collider:
    Alpha (charge +1, mass 1) + Beta (charge -1, mass 1) -> Gamma (charge 0, mass 2)
    """
    alpha = {"charge": 1, "mass": 1, "vx": 2.0, "vy": 1.0}
    beta = {"charge": -1, "mass": 1, "vx": 2.0, "vy": -1.0}
    
    # Net charge
    net_charge = alpha["charge"] + beta["charge"]
    assert net_charge == 0, "Fused product must be electrically neutral (Gamma)"
    
    # Total momentum before
    px_before = alpha["mass"] * alpha["vx"] + beta["mass"] * beta["vx"]
    py_before = alpha["mass"] * alpha["vy"] + beta["mass"] * beta["vy"]
    
    # Product mass and velocity
    gamma_mass = alpha["mass"] + beta["mass"]
    gamma_vx = px_before / gamma_mass
    gamma_vy = py_before / gamma_mass
    
    assert gamma_mass == 2
    assert math.isclose(gamma_vx, 2.0)
    assert math.isclose(gamma_vy, 0.0)

def test_state_serialization_roundtrip():
    nodes = [
        {"type": "gravity", "x": 350, "y": 240, "power": 150},
        {"type": "repulsor", "x": 500, "y": 400, "power": 120},
        {"type": "magnetic", "x": 600, "y": 300, "power": 1.8},
        {"type": "collider", "x": 750, "y": 360, "radius": 34}
    ]
    
    # Serialize to base64 JSON
    raw_json = json.dumps(nodes, separators=(',', ':'))
    encoded = base64.b64encode(raw_json.encode('utf-8')).decode('ascii')
    
    # Deserialize back
    decoded_json = base64.b64decode(encoded.encode('ascii')).decode('utf-8')
    recovered = json.loads(decoded_json)
    
    assert recovered == nodes, "Roundtrip serialization must be lossless"
