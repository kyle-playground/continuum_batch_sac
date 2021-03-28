from xml.dom import minidom
import os


def make_link(name, shape, size, rpy, xyz):
    root = minidom.Document()
    link = root.createElement('link')
    link.setAttribute('name', name)
    visual = root.createElement('visual')
    link.appendChild(visual)
    geometry = root.createElement('geometry')
    visual.appendChild(geometry)
    if shape == "box":
        box = root.createElement('box')
        box.setAttribute('size', size)
        geometry.appendChild(box)
    elif shape == "cylinder":
        cylinder = root.createElement('cylinder')
        cylinder.setAttribute('length', size[0])
        cylinder.setAttribute('radius', size[1])
        geometry.appendChild(cylinder)
    origin = root.createElement('origin')
    origin.setAttribute('rpy', rpy)
    origin.setAttribute('xyz', xyz)
    visual.appendChild(origin)
    collision = root.createElement('collision')
    link.appendChild(collision)
    geometry_c = root.createElement('geometry')
    collision.appendChild(geometry_c)
    if shape == "box":
        box = root.createElement('box')
        box.setAttribute('size', size)
        geometry_c.appendChild(box)
    elif shape == "cylinder":
        cylinder = root.createElement('cylinder')
        cylinder.setAttribute('length', size[0])
        cylinder.setAttribute('radius', size[1])
        geometry_c.appendChild(cylinder)
    inertial = root.createElement('inertial')
    mass = root.createElement('mass')
    mass.setAttribute('value', "1")
    inertial.appendChild(mass)
    inertia = root.createElement('inertia')
    inertia.setAttribute("ixx", "0.1")
    inertia.setAttribute("ixy", "0.0")
    inertia.setAttribute("ixz", "0.0")
    inertia.setAttribute("iyy", "0.1")
    inertia.setAttribute("iyz", "0.0")
    inertia.setAttribute("izz", "0.1")
    inertial.appendChild(inertia)
    link.appendChild(inertial)

    return link


def make_virtual_link(name, xyz):
    root = minidom.Document()
    link = root.createElement('link')
    link.setAttribute('name', name)
    origin = root.createElement('origin')
    origin.setAttribute('xyz',xyz)
    link.appendChild(origin)
    inertial = root.createElement('inertial')
    mass = root.createElement('mass')
    mass.setAttribute('value', "1")
    inertial.appendChild(mass)
    inertia = root.createElement('inertia')
    inertia.setAttribute("ixx", "0.1")
    inertia.setAttribute("ixy", "0.0")
    inertia.setAttribute("ixz", "0.0")
    inertia.setAttribute("iyy", "0.1")
    inertia.setAttribute("iyz", "0.0")
    inertia.setAttribute("izz", "0.1")
    inertial.appendChild(inertia)
    link.appendChild(inertial)
    return link


def make_joint(name, type, originAtt, parentName, childName, limitAtt=None, axisAtt=None):
    root = minidom.Document()
    joint = root.createElement('joint')
    joint.setAttribute('name', name)
    joint.setAttribute('type', type)
    if type == "revolute" or type == "prismatic":
        limit = root.createElement('limit')
        limit.setAttribute('effort', limitAtt[0])
        limit.setAttribute('lower', limitAtt[1])
        limit.setAttribute('upper', limitAtt[2])
        limit.setAttribute('velocity', limitAtt[3])
        joint.appendChild(limit)
        axis = root.createElement('axis')
        axis.setAttribute('xyz', axisAtt)
        joint.appendChild(axis)
    origin = root.createElement('origin')
    origin.setAttribute('rpy', originAtt[0])
    origin.setAttribute('xyz', originAtt[1])
    parent = root.createElement('parent')
    parent.setAttribute('link', parentName)
    child = root.createElement('child')
    child.setAttribute('link', childName)

    joint.appendChild(origin)
    joint.appendChild(parent)
    joint.appendChild(child)

    return joint


length = "0.5"
radius = "0.25"
limit_for_joint = ["1000.0", "-0.5", "0.5", "0"]    # [unknown, pos limit, vel limit]
limit_for_slide = ["1000.0", "-0", "4", "0.5"]      # [unknown, pos limit, vel limit]
axis_x = "1 0 0"
axis_y = "0 1 0"
axis_z = "0 0 1"

root = minidom.Document()
robot = root.createElement('robot')
root.appendChild(robot)
robot.setAttribute('name', "RL_manipulator")
link1 = make_link("base_link", "box", "1 1 2", "0 0 0", "0 0 1")
link2 = make_link("slide_track", "box", "1 1 0.5", "0 0 0", "0 0 0")
joint1 = make_joint("base_link__slide_track", "fixed", ["0 0 0", "0 0 2"], "base_link", "slide_track")
link3 = make_link("mover", "box", "1 1 1", "0 0 0", "0 0 0")
joint2 = make_joint("slide_track__mover", "prismatic", ["0 0 0", "0 0 0.5"], "slide_track", "mover", limit_for_slide,
                    axis_x)

link4 = make_virtual_link("1_arm_virtual", "0 0 0")

joint_first_y = make_joint("mover__1_arm_v", "revolute", ["0 0 0", "0.75 0 0.125"], "mover", "1_arm_virtual",
                    limit_for_joint , axis_y) # origin =[(box lenght)/2 + (cylinder length)/2 0 radius/2]
link5 = make_link("1_arm_real", "cylinder", [length, radius], "0 1.571 0", "0 0 0")
joint_first_z = make_joint("1_arm_v__1_arm_r", "revolute", ["0 0 0", "0 0 0"], "1_arm_virtual", "1_arm_real",
                    limit_for_joint , axis_z)

robot.appendChild(link1)
robot.appendChild(link2)
robot.appendChild(joint1)
robot.appendChild(link3)
robot.appendChild(joint2)
robot.appendChild(link4)
robot.appendChild(joint_first_y)
robot.appendChild(link5)
robot.appendChild(joint_first_z)

for i in range(2,13):
    link_virtual_name = str(i)+"_arm_virtual"
    link_virtual = make_virtual_link(link_virtual_name, "0 0 0")
    joint_v_r_name = str(i-1)+"_arm_r__"+str(i)+"_arm_v"
    joint_origin = ["0 0 0", "0.5 0 0"]  # virtual joint origin = L
    joint_parent = str(i-1)+"_arm_real"
    joint_child = str(i)+"_arm_virtual"
    joint_v_r = make_joint(joint_v_r_name, "revolute", joint_origin, joint_parent, joint_child, limit_for_joint, axis_y)
    link_real_name = str(i)+"_arm_real"
    link_real_cylinder_size = [length, radius]
    link_real_origin = ["0 1.571 0", "0 0 0"]
    link_real = make_link(link_real_name, "cylinder", link_real_cylinder_size, link_real_origin[0], link_real_origin[1])
    joint_r_v_name = str(i)+"_arm_v__"+str(i)+"_arm_r"
    joint2_origin = ["0 0 0", "0 0 0"]
    joint2_parent = str(i) + "_arm_virtual"
    joint2_child = str(i) + "_arm_real"
    joint_r_v = make_joint(joint_r_v_name, "revolute", joint2_origin, joint2_parent, joint2_child, limit_for_joint, axis_z)
    robot.appendChild(link_virtual)
    robot.appendChild(joint_v_r)
    robot.appendChild(link_real)
    robot.appendChild(joint_r_v)


xml_str = root.toprettyxml(indent="\t")
save_path_file = "robot.xml"
with open(save_path_file, "w") as f:
    f.write(xml_str)