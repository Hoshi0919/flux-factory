"""
Physics Engine Verification Tests for Flux Factory.
Validates 2D field interactions, Lorentz deflections, damping, conservation laws,
fusion kinematics, booster vector rotation, power cycling, layout serialization,
kinetic resonance energy windows, subcritical bounces, supercritical shatters,
power grid brownouts, particle half-life decays, and byproduct magnetic separation.
"""
import math
import json
import base64
import pytest

# 1. Physical Constants and Models
EPSILON_SQ = 400.0  # Softening factor to prevent singularity at r -> 0
MIN_RESONANCE_SPEED = 1.8
MAX_RESONANCE_SPEED = 5.2

POWER_RATINGS = {
    'gravity': 12,
    'repulsor': 10,
    'lorentz': 15,
    'dampener': 8,
    'booster': 14,
    'collider': 25
}

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

def evaluate_collision_regime(speed_a, speed_b):
    avg_speed = (speed_a + speed_b) / 2.0
    if avg_speed < MIN_RESONANCE_SPEED:
        return "subcritical"
    elif avg_speed > MAX_RESONANCE_SPEED:
        return "supercritical"
    else:
        return "resonant"

def calc_subcritical_rebound(vx, vy, restitution=0.85):
    return -vx * restitution, -vy * restitution

def calc_grid_state(nodes, max_power):
    total = sum(POWER_RATINGS.get(n.get('type', ''), 0) for n in nodes)
    is_brownout = total > max_power
    factor = 0.55 if is_brownout else 1.0
    return total, is_brownout, factor

def update_particle_decay(half_life):
    if half_life is None:
        return None, False
    new_life = half_life - 1
    annihilated = new_life <= 0
    return new_life, annihilated


# 2. Test Cases
def test_gravity_softening_and_direction():
    ax, ay = calc_gravity_accel(px=0, py=100, nx=100, ny=100, power=200)
    assert ax > 0.0, "Gravity must pull towards positive x"
    assert abs(ay) < 1e-6, "No lateral acceleration on symmetric axis"
    
    ax_zero, ay_zero = calc_gravity_accel(px=100, py=100, nx=100, ny=100, power=200)
    assert ax_zero == 0.0 and ay_zero == 0.0

def test_repulsor_opposes_gravity():
    gx, _ = calc_gravity_accel(50, 100, 100, 100, power=200)
    rx, _ = calc_repulsor_accel(50, 100, 100, 100, power=200)
    assert gx > 0, "Gravity pulls toward node"
    assert rx < 0, "Repulsor pushes away from node"

def test_lorentz_zero_work_theorem():
    vx, vy = 3.0, 4.0
    charge = 1.0
    b_field = 2.5
    ax, ay = calc_lorentz_accel(vx, vy, charge, b_field, dist=30.0)
    
    work_rate = ax * vx + ay * vy
    assert abs(work_rate) < 1e-9, "Lorentz force must do strictly zero work"

def test_lorentz_charge_reflection_symmetry():
    vx, vy = 2.0, 1.5
    dist = 40.0
    b_field = 2.0
    
    ax_pos, ay_pos = calc_lorentz_accel(vx, vy, charge=1.0, b_field=b_field, dist=dist)
    ax_neg, ay_neg = calc_lorentz_accel(vx, vy, charge=-1.0, b_field=b_field, dist=dist)
    
    assert math.isclose(ax_pos, -ax_neg)
    assert math.isclose(ay_pos, -ay_neg)

def test_lorentz_neutral_particle_unperturbed():
    ax, ay = calc_lorentz_accel(2.0, 3.0, charge=0.0, b_field=5.0, dist=10.0)
    assert ax == 0.0 and ay == 0.0, "Neutral gamma particles must not deflect in magnetic field"

def test_dampener_deceleration():
    vx, vy = 5.0, -3.0
    ax, ay = calc_damping(vx, vy, damp_coeff=0.1)
    assert ax * vx < 0, "Damping must directly oppose velocity x"
    assert ay * vy < 0, "Damping must directly oppose velocity y"

def test_booster_impulse_vector_rotation():
    angles = [0, math.pi / 2, math.pi, -math.pi / 2]
    expected_dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    for angle, (exp_x, exp_y) in zip(angles, expected_dirs):
        ax, ay = calc_booster_accel(angle, power=2.0)
        assert math.isclose(ax, exp_x * 2.0, abs_tol=1e-6)
        assert math.isclose(ay, exp_y * 2.0, abs_tol=1e-6)

def test_tokamak_fusion_conservation_laws():
    alpha = {"charge": 1, "mass": 1, "vx": 2.0, "vy": 1.0}
    beta = {"charge": -1, "mass": 1, "vx": 2.0, "vy": -1.0}
    
    net_charge = alpha["charge"] + beta["charge"]
    assert net_charge == 0, "Fused product must be electrically neutral (Gamma)"
    
    px_before = alpha["mass"] * alpha["vx"] + beta["mass"] * beta["vx"]
    py_before = alpha["mass"] * alpha["vy"] + beta["mass"] * beta["vy"]
    
    gamma_mass = alpha["mass"] + beta["mass"]
    gamma_vx = px_before / gamma_mass
    gamma_vy = py_before / gamma_mass
    
    assert gamma_mass == 2
    assert math.isclose(gamma_vx, 2.0)
    assert math.isclose(gamma_vy, 0.0)

def test_state_serialization_roundtrip_all_node_types():
    nodes = [
        {"type": "gravity", "x": 350, "y": 240, "power": 450},
        {"type": "repulsor", "x": 500, "y": 400, "power": 200},
        {"type": "lorentz", "x": 600, "y": 300, "power": 3.5},
        {"type": "dampener", "x": 680, "y": 280, "power": None},
        {"type": "booster", "x": 720, "y": 320, "power": None, "angle": 0.785398},
        {"type": "collider", "x": 800, "y": 350, "power": None}
    ]
    
    export_payload = {
        "lvl": 3,
        "mode": "campaign",
        "nodes": nodes
    }
    
    raw_json = json.dumps(export_payload, separators=(',', ':'))
    encoded = base64.b64encode(raw_json.encode('utf-8')).decode('ascii')
    
    decoded_json = base64.b64decode(encoded.encode('ascii')).decode('utf-8')
    recovered = json.loads(decoded_json)
    
    assert recovered == export_payload, "Full payload roundtrip serialization must be lossless"
    assert math.isclose(recovered["nodes"][4]["angle"], 0.785398)

def test_all_eight_campaign_level_specifications():
    levels = {
        1: {"targetThroughput": 1.8, "requiredHoldTime": 2.5, "maxPower": 100},
        2: {"targetThroughput": 2.5, "requiredHoldTime": 2.5, "maxPower": 100},
        3: {"targetThroughput": 1.8, "requiredHoldTime": 2.5, "maxPower": 100},
        4: {"targetThroughput": 2.0, "requiredHoldTime": 2.5, "maxPower": 100},
        5: {"targetThroughput": 3.5, "requiredHoldTime": 3.0, "maxPower": 110},
        6: {"targetThroughput": 2.2, "requiredHoldTime": 2.5, "maxPower": 65},
        7: {"targetThroughput": 1.2, "requiredHoldTime": 2.5, "maxPower": 95},
        8: {"targetThroughput": 2.2, "requiredHoldTime": 3.0, "maxPower": 85}
    }
    assert len(levels) == 8
    for lvl, spec in levels.items():
        assert spec["targetThroughput"] > 0
        assert spec["requiredHoldTime"] >= 2.0
        assert spec["maxPower"] >= 60

def test_kinetic_energy_and_resonance_window():
    # Below window
    assert evaluate_collision_regime(1.0, 1.2) == "subcritical"
    # Within window
    assert evaluate_collision_regime(2.5, 3.5) == "resonant"
    assert evaluate_collision_regime(1.8, 5.2) == "resonant"
    # Above window
    assert evaluate_collision_regime(5.5, 6.0) == "supercritical"

def test_subcritical_reflection_kinematics():
    vx, vy = 1.2, -0.8
    rx, ry = calc_subcritical_rebound(vx, vy, restitution=0.85)
    assert rx == -1.2 * 0.85
    assert ry == 0.8 * 0.85
    # Velocity inverted
    assert rx * vx < 0 and ry * vy < 0

def test_supercritical_shatter_threshold():
    # If speed is excessive, it shatters without producing target
    regime = evaluate_collision_regime(6.2, 5.8)
    assert regime == "supercritical"

def test_power_grid_budget_and_brownout_factor():
    nodes = [
        {"type": "collider"},   # 25
        {"type": "lorentz"},    # 15
        {"type": "gravity"},    # 12
        {"type": "booster"}     # 14
    ]  # Total = 66 MW
    
    total, brownout, factor = calc_grid_state(nodes, max_power=70)
    assert total == 66
    assert not brownout
    assert factor == 1.0
    
    # Adding another collider pushes over 70 MW
    nodes.append({"type": "collider"})  # Total = 91 MW
    total_ovr, brownout_ovr, factor_ovr = calc_grid_state(nodes, max_power=70)
    assert total_ovr == 91
    assert brownout_ovr
    assert factor_ovr == 0.55, "Brownout must throttle field strengths to 55%"

def test_omega_particle_half_life_decay():
    half_life = 3
    life_1, dead_1 = update_particle_decay(half_life)
    assert life_1 == 2 and not dead_1
    life_2, dead_2 = update_particle_decay(life_1)
    assert life_2 == 1 and not dead_2
    life_3, dead_3 = update_particle_decay(life_2)
    assert life_3 == 0 and dead_3, "Particle must decay and annihilate when timer expires"

def test_byproduct_lorentz_separation_neutral_vs_charged():
    # Gamma (neutral, q=0) travels straight
    ax_gamma, ay_gamma = calc_lorentz_accel(3.0, 0.0, charge=0.0, b_field=2.0, dist=20.0)
    assert ax_gamma == 0.0 and ay_gamma == 0.0
    
    # Positron (byproduct, q=+1) deflects upwards
    ax_pos, ay_pos = calc_lorentz_accel(3.0, 0.0, charge=1.0, b_field=2.0, dist=20.0)
    assert ay_pos != 0.0, "Positron byproduct must be deflected away by Lorentz field"
