from vpython import *

"""Board: the ground that all the objects will be put on it"""
length = 200
width = 200
thickness = 0.2
board = box(pos=vec(0, 0, -thickness), size=vec(length, width, thickness), color=color.gray(0.8))

"""4 Obstacles (cylinders): in the middle of the board"""
offset = 15
obs_radius = 5
height = 15
orange = vec(1, 0.5, 0)
obs1 = cylinder(pos=vec(offset, offset, 0), axis=vec(0, 0, height), radius=obs_radius, color=orange)
obs2 = cylinder(pos=vec(-offset, offset, 0), axis=vec(0, 0, height), radius=obs_radius, color=orange)
obs3 = cylinder(pos=vec(-offset, -offset, 0), axis=vec(0, 0, height), radius=obs_radius, color=orange)
obs4 = cylinder(pos=vec(offset, -offset, 0), axis=vec(0, 0, height), radius=obs_radius, color=orange)

"""pedestrian: pedestrian and the velocity"""
ped = sphere(pos=vec(-95, 0, 0), radius=3, color=color.green)
# velocity of the pedestrian
ped.velocity = vector(5, 0, 0)
arrow_scale = 4
v_arrow = arrow(pos=ped.pos, axis=arrow_scale * ped.velocity, color=color.yellow)

# loop
deltat = 0.005
t = 0
ped.pos = ped.pos + ped.velocity * deltat
while True:
    rate(500)
    ped.pos = ped.pos + ped.velocity * deltat
    v_arrow.pos = ped.pos
    v_arrow.axis = ped.velocity * arrow_scale
    t = t + deltat