from constants import PROJECTOR_H, PROJECTOR_W, CAMERA_H, CAMERA_W

def camera_to_projector_coordinates(camera_x, camera_y):
    """
    Converts camera coordinates to projector coordinates.
    """
    return (camera_x * PROJECTOR_W // CAMERA_W, camera_y * PROJECTOR_H // CAMERA_H)


def projector_to_camera_coordinates(projector_x, projector_y):
    """
    Converts projector coordinates to camera coordinates.
    """
    return (
        projector_x * CAMERA_W // PROJECTOR_W,
        projector_y * CAMERA_H // PROJECTOR_H,
    )
