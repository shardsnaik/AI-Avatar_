# import os
# import numpy as np
# import fbx
# import pydub
# from scipy.io import wavfile

# class LipSyncAnimator:
#     def __init__(self, fbx_path, audio_path):
#         """
#         Initialize the lip-sync animator
        
#         :param fbx_path: Path to the FBX model file
#         :param audio_path: Path to the audio file
#         """
#         # Initialize FBX SDK
#         self.manager = fbx.FbxManager.Create()
#         self.scene = fbx.FbxScene.Create(self.manager, "LipSyncScene")
        
#         # Import FBX file
#         importer = fbx.FbxImporter.Create(self.manager, "")
#         if not importer.Initialize(fbx_path):
#             raise RuntimeError(f"Failed to import FBX file: {fbx_path}")
        
#         importer.Import(self.scene)
#         importer.Destroy()
        
#         # Find mouth-related nodes
#         self.mouth_nodes = self._find_mouth_nodes()
        
#         # Load and process audio
#         self.audio = pydub.AudioSegment.from_file(audio_path)
#         self.sample_rate, self.audio_data = self._process_audio()
    
#     def _find_mouth_nodes(self):
#         """
#         Identify mouth-related nodes in the FBX model
        
#         :return: List of mouth-related nodes
#         """
#         mouth_nodes = []
#         root_node = self.scene.GetRootNode()
        
#         def search_nodes(node):
#             if not node:
#                 return []
            
#             node_name = node.GetName().lower()
#             if any(keyword in node_name for keyword in ['jaw', 'mouth', 'lip']):
#                 mouth_nodes.append(node)
            
#             for i in range(node.GetChildCount()):
#                 search_nodes(node.GetChild(i))
        
#         search_nodes(root_node)
#         return mouth_nodes
    
#     def _process_audio(self):
#         """
#         Process audio for lip-sync analysis
        
#         :return: Sample rate and audio data
#         """
#         # Convert audio to numpy array
#         audio_array = np.array(self.audio.get_array_of_samples())
        
#         # If stereo, take the mean of channels
#         if self.audio.channels > 1:
#             audio_array = audio_array.reshape((-1, self.audio.channels)).mean(axis=1)
        
#         return self.audio.frame_rate, audio_array
    
#     def analyze_phonemes(self):
#         """
#         Analyze audio to extract phoneme information
        
#         :return: List of phoneme intensities over time
#         """
#         # Simple volume-based phoneme approximation
#         frame_size = int(self.sample_rate * 0.025)  # 25ms frames
#         hop_length = int(self.sample_rate * 0.01)   # 10ms hop
        
#         phoneme_intensities = []
#         for i in range(0, len(self.audio_data) - frame_size, hop_length):
#             frame = self.audio_data[i:i+frame_size]
#             intensity = np.abs(frame).mean()
#             phoneme_intensities.append(intensity)
        
#         return phoneme_intensities
    
#     def create_lip_sync_animation(self):
#         """
#         Generate lip-sync animation keyframes
#         """
#         phoneme_intensities = self.analyze_phonemes()
        
#         # Normalize intensities
#         max_intensity = max(phoneme_intensities)
#         normalized_intensities = [i / max_intensity for i in phoneme_intensities]
        
#         # Create animation stack and layer
#         anim_stack = fbx.FbxAnimStack.Create(self.scene, "LipSyncAnimation")
#         anim_layer = fbx.FbxAnimLayer.Create(self.scene, "LipSyncLayer")
#         anim_stack.AddMember(anim_layer)
        
#         # Create animation for each mouth node
#         for node in self.mouth_nodes:
#             # Create animation curve
#             curve = fbx.FbxAnimCurve.Create(self.scene, f"{node.GetName()}_LipSync")
#             anim_layer.AddCurve(node, curve)
            
#             # Set keyframes based on phoneme intensities
#             curve.KeyModifyBegin()
#             for time, intensity in enumerate(normalized_intensities):
#                 # Map intensity to rotation/translation
#                 # Adjust these values based on your specific model's requirements
#                 key = curve.KeyAdd(time * 0.01)  # 10ms intervals
#                 curve.KeySet(key, time * 0.01, intensity * 30, fbx.FbxAnimCurveDef.eInterpolationLinear)
#             curve.KeyModifyEnd()
    
#     def save_animated_model(self, output_path):
#         """
#         Save the animated FBX model
        
#         :param output_path: Path to save the animated FBX file
#         """
#         # Create exporter
#         exporter = fbx.FbxExporter.Create(self.manager, "")
        
#         # Initialize the exporter
#         if not exporter.Initialize(output_path):
#             raise RuntimeError(f"Failed to initialize exporter for {output_path}")
        
#         # Export the scene
#         exporter.Export(self.scene)
        
#         # Clean up
#         exporter.Destroy()
    
#     def __del__(self):
#         """
#         Clean up FBX SDK resources
#         """
#         if hasattr(self, 'manager'):
#             self.manager.Destroy()

# # Example usage
# def main():
#     # Paths to your files
#     fbx_model_path = "C:\\Users\\Admin\\Downloads\\uploads_files_4822046_Fbx\\Fbx\\Jake.fbx"
#     audio_file_path = 'image_src\\sample.wav'
#     output_model_path = 'path/to/output/lip_synced_model.fbx'
    
#     # Create lip sync animator
#     lip_sync = LipSyncAnimator(fbx_model_path, audio_file_path)
    
#     # Generate lip sync animation
#     lip_sync.create_lip_sync_animation()
    
#     # Save animated model
#     lip_sync.save_animated_model(output_model_path)

# if __name__ == "__main__":
#     main()


# Approch using pybullet libraries for 3D model animation

import os
import numpy as np
import librosa
import time
import pybullet as p
import pybullet_data
from scipy.io import wavfile

class LipSyncAnimator:
    def __init__(self, model_path, audio_path):
        """
        Initialize the lip-sync animator with PyBullet
        
        :param model_path: Path to the 3D model file (URDF or OBJ)
        :param audio_path: Path to the audio file
        """
        # Initialize PyBullet
        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        
        # Load model
        self.model_id = self.load_model(model_path)
        
        # Load and process audio
        self.audio_features = self.process_audio(audio_path)
        
        # Find joint indices for mouth/face
        self.mouth_joints = self.find_mouth_joints()
        
    def load_model(self, model_path):
        """
        Load the 3D model based on file extension
        
        :param model_path: Path to the model file
        :return: Model ID
        """
        extension = os.path.splitext(model_path)[1].lower()
        
        if extension == '.urdf':
            return p.loadURDF(model_path)
        elif extension == '.obj':
            return p.loadSoftBody(model_path)
        elif extension in ['.stl', '.dae']:
            visual_shape_id = p.createVisualShape(
                shapeType=p.GEOM_MESH,
                fileName=model_path
            )
            return p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=visual_shape_id
            )
        else:
            # Try to load as URDF as fallback
            try:
                return p.loadURDF(model_path)
            except p.error:
                raise ValueError(f"Unsupported model format: {extension}")
    
    def process_audio(self, audio_path):
        """
        Process audio file to extract features for lip-syncing
        
        :param audio_path: Path to the audio file
        :return: Audio features
        """
        # Load audio with librosa
        y, sr = librosa.load(audio_path, sr=None)
        
        # Extract features
        # We'll use RMS energy as a simple way to detect speech intensity
        hop_length = int(sr * 0.01)  # 10ms hop
        frame_length = int(sr * 0.025)  # 25ms frames
        
        # Calculate RMS energy for each frame
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Normalize
        rms_normalized = (rms - np.min(rms)) / (np.max(rms) - np.min(rms) + 1e-6)
        
        return {
            'rms': rms_normalized,
            'sr': sr,
            'hop_length': hop_length,
            'duration': len(y) / sr
        }
    
    def find_mouth_joints(self):
        """
        Find mouth-related joints in the model
        
        :return: List of mouth joint indices
        """
        num_joints = p.getNumJoints(self.model_id)
        mouth_joints = []
        
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.model_id, i)
            joint_name = joint_info[1].decode('utf-8').lower()
            
            # Check if joint name contains mouth-related keywords
            if any(keyword in joint_name for keyword in ['mouth', 'lip', 'jaw', 'face']):
                mouth_joints.append(i)
        
        if not mouth_joints:
            print("No mouth joints found. Using head joints or closest alternatives.")
            # Fallback to head joints or other face-related joints
            for i in range(num_joints):
                joint_info = p.getJointInfo(self.model_id, i)
                joint_name = joint_info[1].decode('utf-8').lower()
                if any(keyword in joint_name for keyword in ['head', 'neck', 'face']):
                    mouth_joints.append(i)
        
        return mouth_joints
    
    def animate_lip_sync(self):
        """
        Animate the model with lip-sync movements
        """
        if not self.mouth_joints:
            print("No suitable joints found for lip-sync animation.")
            return
        
        # Duration of each frame in seconds
        frame_duration = self.audio_features['hop_length'] / self.audio_features['sr']
        
        # Animate each frame
        for frame, intensity in enumerate(self.audio_features['rms']):
            # Calculate mouth opening based on intensity
            mouth_open_factor = intensity * 0.5  # Scale factor
            
            # Apply to mouth joints
            for joint_idx in self.mouth_joints:
                # Get joint info
                joint_info = p.getJointInfo(self.model_id, joint_idx)
                joint_name = joint_info[1].decode('utf-8').lower()
                
                # Different joints might need different transformations
                if 'jaw' in joint_name:
                    # Jaw typically rotates
                    p.setJointMotorControl2(
                        self.model_id,
                        joint_idx,
                        p.POSITION_CONTROL,
                        targetPosition=mouth_open_factor * 0.5,  # Adjust scaling
                        force=5
                    )
                elif 'lip' in joint_name:
                    # Lips might need to be moved differently
                    if 'upper' in joint_name:
                        p.setJointMotorControl2(
                            self.model_id,
                            joint_idx,
                            p.POSITION_CONTROL,
                            targetPosition=mouth_open_factor * 0.2,  # Less movement for upper lip
                            force=5
                        )
                    elif 'lower' in joint_name:
                        p.setJointMotorControl2(
                            self.model_id,
                            joint_idx,
                            p.POSITION_CONTROL,
                            targetPosition=mouth_open_factor * 0.4,  # More movement for lower lip
                            force=5
                        )
            
            # Step simulation
            p.stepSimulation()
            
            # Pause to match audio timing
            time.sleep(frame_duration)
    
    def export_animation(self, output_path):
        """
        Export the animation to a file
        
        :param output_path: Path to save the animation
        """
        # Unfortunately, PyBullet doesn't have a built-in exporter for animations
        # This is a placeholder for where you'd implement an export function
        # You'd need to capture the joint states at each frame and save them
        
        print(f"Animation export would go to: {output_path}")
        print("Animation export is not implemented in this version.")
    
    def cleanup(self):
        """
        Clean up PyBullet connection
        """
        p.disconnect()

def main():
    # File paths
    model_path = "C:\\AI-Avatar_\\image_src\\3d_models\\obj\\obj_file.obj"  # or .obj
    audio_path = 'image_src\\sample.wav'
    output_path = 'path/to/output/animation.fbx'
    
    # Create animator
    animator = LipSyncAnimator(model_path, audio_path)
    
    try:
        # Animate lip sync
        animator.animate_lip_sync()
        
        # Export animation (placeholder)
        animator.export_animation(output_path)
    finally:
        # Make sure to clean up
        animator.cleanup()

if __name__ == "__main__":
    main()