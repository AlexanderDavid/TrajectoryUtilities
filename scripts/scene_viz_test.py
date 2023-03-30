from ardi.visualization import Scene

debug = False

if debug:
    # Perpendicular
    s = Scene(
        [[2, 0], [6, 0], [10, 0], [-4, 7]],
        [[2, 15], [6, 15], [10, 15], [14, 7]],
        3,
        0.05,
    )
    s.show()

else:
    # Perpendicular
    s = Scene(
        [[2, 0], [6, 0], [10, 0], [-3, 7]],
        [[2, 15], [6, 15], [10, 15], [14, 7]],
        3,
        0.05,
    )
    s.save("perpendicular.png", "png")

    # Adjacent
    s = Scene(
        [[2, 0], [6, 0], [10, 0], [-2, 3]],
        [[2, 15], [6, 15], [10, 15], [15, 12]],
        3,
        0.05,
    )
    s.save("adjacent.png", "png")

    # Opposite
    s = Scene(
        [[2, 0], [6, 0], [10, 0], [-3, 7]],
        [[2, 15], [6, 15], [10, 15], [15, 0]],
        3,
        0.05,
    )
    s.save("opposite.png", "png")

    # Intersection
    s = Scene([[0, 0], [3, 15], [-5, 7.5]], [[0, 15], [3, 0], [7, 7.5]], 2, 0.01)
    s.save("intersection.png", "png")
