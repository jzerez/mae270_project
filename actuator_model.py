import csdl_alpha as csdl
import numpy as np


def actuator_mass_from_power(power,
                             base_mass=0.3,
                             mass_per_watt=0.004):
    """
    Simple differentiable actuator mass model.

    m = base_mass + mass_per_watt * power
    """
    return base_mass + mass_per_watt * power


def actuator_limits_from_power_and_gear(power,
                                        gear_ratio,
                                        efficiency=0.6,
                                        motor_base_speed=100.0,
                                        eps=1e-8):
    """
    First-pass actuator power / torque / speed model.

    Assumptions
    -----------
    mechanical power = efficiency * electrical power
    output speed = motor_base_speed / gear_ratio
    output torque = mechanical_power / output_speed
    """
    mech_power_max = efficiency * power
    omega_max = motor_base_speed / (gear_ratio + eps)
    tau_max = mech_power_max / (omega_max + eps)

    return tau_max, omega_max, mech_power_max


def actuator_power_usage(tau, omega):
    """
    Mechanical power magnitude.
    """
    return csdl.absolute(tau * omega)


def actuator_feasibility_metrics(tau_required,
                                 omega_required,
                                 tau_max,
                                 omega_max,
                                 mech_power_max,
                                 eps=1e-8):
    """
    Returns normalized utilization metrics.

    <= 1 : feasible
    > 1  : violation
    """
    power_required = actuator_power_usage(tau_required, omega_required)

    torque_ratio = csdl.absolute(tau_required) / (tau_max + eps)
    speed_ratio = csdl.absolute(omega_required) / (omega_max + eps)
    power_ratio = power_required / (mech_power_max + eps)

    max_ratio = csdl.maximum(
        torque_ratio,
        csdl.maximum(speed_ratio, power_ratio)
    )

    return {
        "torque_ratio": torque_ratio,
        "speed_ratio": speed_ratio,
        "power_ratio": power_ratio,
        "max_ratio": max_ratio,
    }


def actuator_constraint_residuals(tau_required,
                                  omega_required,
                                  tau_max,
                                  omega_max,
                                  mech_power_max):
    """
    Residual form useful for optimization constraints.

    residual <= 0 means feasible
    residual > 0 means violation
    """
    power_required = actuator_power_usage(tau_required, omega_required)

    return {
        "torque_residual": csdl.absolute(tau_required) - tau_max,
        "speed_residual": csdl.absolute(omega_required) - omega_max,
        "power_residual": power_required - mech_power_max,
    }


def build_actuator_bundle(power,
                          gear_ratio,
                          efficiency=0.6,
                          motor_base_speed=100.0,
                          base_mass=0.3,
                          mass_per_watt=0.004):
    """
    Convenience wrapper that returns all actuator properties.
    """
    mass = actuator_mass_from_power(
        power,
        base_mass=base_mass,
        mass_per_watt=mass_per_watt,
    )

    tau_max, omega_max, mech_power_max = actuator_limits_from_power_and_gear(
        power,
        gear_ratio,
        efficiency=efficiency,
        motor_base_speed=motor_base_speed,
    )

    return {
        "mass": mass,
        "tau_max": tau_max,
        "omega_max": omega_max,
        "mech_power_max": mech_power_max,
    }


if __name__ == "__main__":
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    power = csdl.Variable(value=120.0)
    gear = csdl.Variable(value=12.0)

    act = build_actuator_bundle(power, gear)

    tau_req = csdl.Variable(value=5.0)
    omega_req = csdl.Variable(value=6.0)

    metrics = actuator_feasibility_metrics(
        tau_req,
        omega_req,
        act["tau_max"],
        act["omega_max"],
        act["mech_power_max"],
    )

    residuals = actuator_constraint_residuals(
        tau_req,
        omega_req,
        act["tau_max"],
        act["omega_max"],
        act["mech_power_max"],
    )

    recorder.stop()

    print("mass =", act["mass"].value)
    print("tau_max =", act["tau_max"].value)
    print("omega_max =", act["omega_max"].value)
    print("mech_power_max =", act["mech_power_max"].value)

    print("torque_ratio =", metrics["torque_ratio"].value)
    print("speed_ratio =", metrics["speed_ratio"].value)
    print("power_ratio =", metrics["power_ratio"].value)
    print("max_ratio =", metrics["max_ratio"].value)

    print("torque_residual =", residuals["torque_residual"].value)
    print("speed_residual =", residuals["speed_residual"].value)
    print("power_residual =", residuals["power_residual"].value)