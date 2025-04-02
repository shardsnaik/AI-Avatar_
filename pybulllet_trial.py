import pybullet as pb
import pybullet_data
import time

pysics_client = pb.connect(pb.GUI)
pb.setGravity(0, 0, -9.81)
pb.setAdditionalSearchPath(pybullet_data.getDataPath())

planeID = pb.loadURDF("plane.urdf")

# loading the cube
cubeStartPos = [0, 0, 1]
cubeStartOrientation = pb.getQuaternionFromEuler([0, 0, 0])

cubeScale = 0.1


# Create collision shape and visual shape
colCubeId = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[cubeScale]*3)
visCubeId = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[cubeScale]*3, rgbaColor=[1,0,0,1])
# Create multi-body
cubeId = pb.createMultiBody(
    baseMass=1,  # 1 kg
    baseCollisionShapeIndex=colCubeId,
    baseVisualShapeIndex=visCubeId,
    basePosition=cubeStartPos
)
cubeId = pb.loadURDF("franka_panda/panda.urdf", cubeStartPos, cubeStartOrientation)

# Set camera view (add after connection)
pb.resetDebugVisualizerCamera(
    cameraDistance=1.5,
    cameraYaw=45,
    cameraPitch=-30,
    cameraTargetPosition=[0, 0, 0.5]
)

# # Draw coordinate frames
# pb.addUserDebugLine([0,0,0], [0.2,0,0], [1,0,0])  # X-axis (red)
# pb.addUserDebugLine([0,0,0], [0,0.2,0], [0,1,0])  # Y-axis (green)
# pb.addUserDebugLine([0,0,0], [0,0,0.2], [0,0,1])  # Z-axis (blue)

# # Show contact points
# pb.configureDebugVisualizer(pb.COV_ENABLE_CONTACT_POINTS, True)

# # Toggle wireframe mode
# pb.configureDebugVisualizer(pb.COV_ENABLE_WIREFRAME, True)

# Run the simulation for 5 seconds
for _ in range(1000):
    pb.stepSimulation()
    time.sleep(1/240)  # PyBullet runs at 240 Hz

# Disconnect from the physics server
pb.disconnect()