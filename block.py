import numpy as np
import cv2
import screeninfo
import time

from constants import (
    CAMERA_H,
    CAMERA_W,
    PROJECTOR_H,
    PROJECTOR_W,
    SCREEN_ID,
    WINDOW_NAME,
)


class Block:
    def __init__(self, min_x, min_y, max_x, max_y, parent: int) -> None:
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        self.parent = parent
        self.children = []

    def __str__(self) -> str:
        return f"Block(min_x={self.min_x}, min_y={self.min_y}, max_x={self.max_x}, max_y={self.max_y})"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other) -> bool:
        return (
            self.min_x == other.min_x
            and self.min_y == other.min_y
            and self.max_x == other.max_x
            and self.max_y == other.max_y
        )

    def __hash__(self) -> int:
        return hash((self.min_x, self.min_y, self.max_x, self.max_y))

    def __nonzero__(self) -> bool:
        return self.min_x != self.max_x and self.min_y != self.max_y

    def __contains__(self, point) -> bool:
        """
        Returns True if the point is contained in the block.
        """
        return (
            self.min_x <= point[0] < self.max_x and self.min_y <= point[1] < self.max_y
        )

    def subdivide(self, parent_num) -> list:
        """
        Subdivides the block into 4 blocks.
        """
        # Calculate the midpoints of the block
        mid_x = (self.min_x + self.max_x) // 2
        mid_y = (self.min_y + self.max_y) // 2

        # Create the 4 blocks
        blocks = [
            Block(self.min_x, self.min_y, mid_x, mid_y, parent_num),
            Block(self.min_x, mid_y, mid_x, self.max_y, parent_num),
            Block(mid_x, self.min_y, self.max_x, mid_y, parent_num),
            Block(mid_x, mid_y, self.max_x, self.max_y, parent_num),
        ]

        # Filter nonzero blocks
        blocks = [block for block in blocks if block]

        return blocks


def create_block_0():
    """
    Creates the first block (floodlit image)
    """
    return Block(0, 0, PROJECTOR_H, PROJECTOR_W, 0)


def create_blocks_window():
    screen = screeninfo.get_monitors()[SCREEN_ID]
    cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow(WINDOW_NAME, screen.x - 1, screen.y - 1)
    # cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(WINDOW_NAME, np.ones((PROJECTOR_H, PROJECTOR_W)))
    print(
        "Move window to projector and full-screen. Press any key on the window to continue."
    )
    cv2.waitKey(0)
    print(
        "OK Starting...get your cursor outta there! Press enter in the terminal to continue."
    )
    input()


def destroy_blocks_window():
    cv2.destroyAllWindows()


def display_blocks(*blocks):
    # Create image with the union of all the blocks
    im = np.zeros((PROJECTOR_H, PROJECTOR_W))

    for block in blocks:
        im[block.min_x : block.max_x, block.min_y : block.max_y] = 1

    cv2.imshow(WINDOW_NAME, im)
    cv2.waitKey(100)
    time.sleep(.1)


def clear_blocks():
    # Clears screen
    cv2.imshow(WINDOW_NAME, np.zeros((PROJECTOR_H, PROJECTOR_W)))
    cv2.waitKey(100)
    time.sleep(.1)


def floodlight_blocks():
    # Sets screen to white
    cv2.imshow(WINDOW_NAME, np.ones((PROJECTOR_H, PROJECTOR_W)))
    cv2.waitKey(100)
    time.sleep(.1)
