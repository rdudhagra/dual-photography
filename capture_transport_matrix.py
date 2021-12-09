import cv2
import pickle
import numpy as np
import time
import networkx as nx
from itertools import combinations
from camera import camera_close, camera_init, camera_read, camera_read_gray

from constants import CAMERA_H, CAMERA_W, SUBDIVISION_THRESHOLD, ZERO_THRESHOLD
from block import (
    create_block_0,
    create_blocks_window,
    destroy_blocks_window,
    display_blocks,
)

blocks = []
T = {}


def initialization():
    global blocks

    create_blocks_window()
    camera_init()

    blocks.append(create_block_0())

    blocks_affected_by_each_pixel = np.tile(frozenset([0]), (CAMERA_H, CAMERA_W))

    return blocks_affected_by_each_pixel


def construct_conflict_free_lists(blocks_affected_by_each_pixel):
    # Get all blocks that at least one pixel is influenced by
    all_blocks = set().union(*blocks_affected_by_each_pixel.flat)

    # Conflicts cache to avoid redundant computation
    conflicts_cache = set()

    conflict_graph = nx.Graph()
    conflict_graph.add_nodes_from(all_blocks)

    for blocks_influenced_by_pixel in blocks_affected_by_each_pixel.flat:
        if blocks_influenced_by_pixel in conflicts_cache:
            continue

        conflicts = combinations(blocks_influenced_by_pixel, 2)
        conflict_graph.add_edges_from(conflicts)
        conflicts_cache.add(blocks_influenced_by_pixel)

    # Color the graph to get conflict-free nodes
    blocks_to_colors_dict = nx.coloring.greedy_color(conflict_graph)

    # Flip keys and values of dictionary to get conflict-free lists
    colors_to_blocks_dict = {}
    for k, v in blocks_to_colors_dict.items():
        if v in colors_to_blocks_dict:
            colors_to_blocks_dict[v].add(k)
        else:
            colors_to_blocks_dict[v] = set([k])

    conflict_free_lists = list(colors_to_blocks_dict.values())
    return conflict_free_lists


def acquire_images(conflict_free_lists):

    captured_images = []

    for l in conflict_free_lists:
        # Display blocks in list
        display_blocks(*[blocks[b] for b in l])

        # Capture image
        im = camera_read_gray()
        captured_images.append(im)

    return captured_images


def process_results(blocks_affected_by_each_pixel, conflict_free_lists, new_images):
    global blocks
    global T_energies
    global T_corresponding_block

    subdivided_at_least_one_block = False

    new_blocks_affected_by_each_pixel = np.tile(None, (CAMERA_H, CAMERA_W))

    for x in range(CAMERA_H):
        for y in range(CAMERA_W):
            blocks_affected_by_this_pixel = blocks_affected_by_each_pixel[x, y]
            new_blocks_affected_by_this_pixel = set()

            for conflict_free_set, captured_image in zip(
                conflict_free_lists, new_images
            ):
                if captured_image[x, y] < ZERO_THRESHOLD:
                    continue  # No value measured, do nothing

                current_block_intersection_set = (
                    conflict_free_set & blocks_affected_by_this_pixel
                )
                if len(current_block_intersection_set) == 0:
                    continue  # This pixel is not influenced by any block in this conflict-free set

                assert len(current_block_intersection_set) == 1
                current_block = current_block_intersection_set.pop()
                if (
                    captured_image[x, y] < SUBDIVISION_THRESHOLD
                    or not blocks[current_block]
                ):
                    # Energy is not enough to subdivide or the block cannot be subdivided further
                    if current_block not in T:
                        T[current_block] = {}
                    T[current_block][x * CAMERA_W + y] = captured_image[x, y]
                else:
                    if not blocks[current_block].children:
                        # Subdivide the block
                        subdivided_at_least_one_block = True
                        new_blocks = blocks[current_block].subdivide(current_block)
                        start_block_num = len(blocks)
                        blocks.extend(new_blocks)
                        end_block_num = len(blocks)
                        blocks[current_block].children = list(
                            range(start_block_num, end_block_num)
                        )

                    new_blocks_affected_by_this_pixel.update(
                        blocks[current_block].children
                    )

            new_blocks_affected_by_each_pixel[x, y] = frozenset(
                new_blocks_affected_by_this_pixel
            )

    return subdivided_at_least_one_block, new_blocks_affected_by_each_pixel


def main():
    blocks_affected_by_each_pixel = initialization()
    time.sleep(0.1)
    cv2.imwrite("prime_image.jpg", camera_read())
    time.sleep(0.1)

    levels = {0: range(0, 1)}

    for level in range(14):
        print(f"Capturing level {level}...")

        conflict_free_lists = construct_conflict_free_lists(
            blocks_affected_by_each_pixel
        )

        new_images = acquire_images(conflict_free_lists)

        level_blocks_start = len(blocks)
        subdivided_blocks, blocks_affected_by_each_pixel = process_results(
            blocks_affected_by_each_pixel, conflict_free_lists, new_images
        )
        level_blocks_end = len(blocks)
        levels[level + 1] = range(level_blocks_start, level_blocks_end)

        if not subdivided_blocks:
            break

    print("Done! Saving results...")
    camera_close()
    time.sleep(1)

    # Save T to file
    with open("T.pkl", "wb") as f:
        pickle.dump(T, f)

    # Save blocks to file
    with open("blocks.pkl", "wb") as f:
        pickle.dump(blocks, f)

    # Save levels to file
    with open("levels.pkl", "wb") as f:
        pickle.dump(levels, f)



if __name__ == "__main__":
    main()
