# Code from Google Gemini with minor tweaks for customization

import math
from xml.etree import ElementTree as ET
from xml.dom import minidom

def prettify(elem):
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def get_connector_params(xyz):
    """Calculates length, center position, and RPY to point a Z-axis cylinder to xyz."""
    x, y, z = xyz
    length = math.sqrt(x**2 + y**2 + z**2)
    
    if length < 1e-6:
        return 0, (0,0,0), (0,0,0)

    # Center of the cylinder is half-way to the next joint
    pos = (x/2.0, y/2.0, z/2.0)
    
    # Calculate orientation (Roll-Pitch-Yaw) to align Z-axis with the vector
    # Yaw (rotation around Z)
    yaw = math.atan2(y, x)
    # Pitch (rotation around Y)
    pitch = math.atan2(math.sqrt(x**2 + y**2), z)
    # Roll is not needed for a symmetric cylinder
    roll = 0.0
    
    return length, pos, (roll, pitch, yaw)

def create_urdf(joint_configs, output_file="robot.urdf", act_rad=0.15, act_len=0.3, link_rad=0.1):
    robot = ET.Element("robot", name="GeneratedRobot")

    # 1. Base Link
    base_link = ET.SubElement(robot, "link", name="base_link")
    
    prev_link_name = "base_link"

    for i, config in enumerate(joint_configs):
        curr_link_name = f"link_{i}"
        joint_name = f"joint_{i}"
        
        xyz = config['xyz']
        rpy = config['rpy']

        # --- CREATE JOINT ---
        joint = ET.SubElement(robot, "joint", name=joint_name, type="revolute")
        ET.SubElement(joint, "parent", link=prev_link_name)
        ET.SubElement(joint, "child", link=curr_link_name)
        ET.SubElement(joint, "origin", xyz=f"{xyz[0]} {xyz[1]} {xyz[2]}", rpy=f"{rpy[0]} {rpy[1]} {rpy[2]}")
        ET.SubElement(joint, "axis", xyz="0 0 1")
        # Added dummy limit since 'revolute' requires it
        ET.SubElement(joint, "limit", effort="1000", lower="-3.14", upper="3.14", velocity="0.5")

        # --- CREATE LINK ---
        link = ET.SubElement(robot, "link", name=curr_link_name)

        # Visual 1: Joint Cylinder (Dark Gray, Z-aligned)
        vis_joint = ET.SubElement(link, "visual")
        ET.SubElement(vis_joint, "origin", xyz="0 0 0", rpy="0 0 0")
        geom_joint = ET.SubElement(vis_joint, "geometry")
        ET.SubElement(geom_joint, "cylinder", radius=str(act_rad), length=str(act_len))
        mat_joint = ET.SubElement(vis_joint, "material", name="dark_gray")
        ET.SubElement(mat_joint, "color", rgba="0.3 0.3 0.3 1.0")

        # Visual 2: Connector (Light Gray)
        # We look ahead to the NEXT joint's xyz to see where to point the "bone"
        if i + 1 < len(joint_configs):
            next_xyz = joint_configs[i+1]['xyz']
            length, pos, angles = get_connector_params(next_xyz)
            
            if length > 1e-6:
                vis_conn = ET.SubElement(link, "visual")
                ET.SubElement(vis_conn, "origin", 
                               xyz=f"{pos[0]} {pos[1]} {pos[2]}", 
                               rpy=f"{angles[0]} {angles[1]} {angles[2]}")
                geom_conn = ET.SubElement(vis_conn, "geometry")
                ET.SubElement(geom_conn, "cylinder", radius=str(link_rad), length=str(length))
                mat_conn = ET.SubElement(vis_conn, "material", name="light_gray")
                ET.SubElement(mat_conn, "color", rgba="0.7 0.7 0.7 1.0")

        prev_link_name = curr_link_name

    # Save to file
    with open(output_file, "w") as f:
        f.write(prettify(robot))
    print(f"Successfully saved URDF to {output_file}")


# --- Example Usage ---
if __name__ == "__main__":
    # Define a simple 3-link arm
    # xyz/rpy is relative to the previous joint frame
    my_configs = [
        {'xyz': [0, 0, 0], 'rpy': [0, 0, 0]},        # Joint 0
        {'xyz': [1.25704309, 0.10060232, 0], 'rpy': [0, 0, 0]},  # Joint 1
        {'xyz': [0.10059914, 1.39927151, 0], 'rpy': [0, 0, 0]},        # Joint 2
    ]

    robot = create_urdf(my_configs)


    # robot.animate(cfg_)

    
    # # Save to file
    # robot.save("custom_robot.urdf")
    # print("URDF saved as custom_robot.urdf")
    
    # # Visualize (requires pyrender)
    # robot.show()
