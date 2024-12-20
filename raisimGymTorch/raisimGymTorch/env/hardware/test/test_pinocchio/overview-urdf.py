import pinocchio
from sys import argv
from os.path import dirname, join, abspath

# Load the urdf model
model = pinocchio.buildModelFromUrdf("/home/ubuntu/hand/UR5_IK/rsc/ur5.urdf")
print("model name: " + model.name)
 
# Create data required by the algorithms
data = model.createData()
 
# Sample a random configuration
q = pinocchio.randomConfiguration(model)
print("q: %s" % q.T)
 
# Perform the forward kinematics over the kinematic tree
pinocchio.forwardKinematics(model, data, q)
 
# Print out the placement of each joint of the kinematic tree
for name, oMi in zip(model.names, data.oMi):
    print(("{:<24} : {: .2f} {: .2f} {: .2f}".format(name, *oMi.translation.T.flat)))