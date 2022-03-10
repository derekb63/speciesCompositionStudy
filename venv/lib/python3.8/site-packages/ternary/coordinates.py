
axes_dict = dict(zip("brl", "012"))
reverse_coordinate_dict = dict(zip("012", "brl"))

corners_dict = dict(zip("LRT", "012"))
reverse_corners_dict = dict(zip("012", "LRT"))

corners = {'T', 'L', 'R'}
axes = {'b', 'r', 'l'}


def represent_permutation(permutation, reverse=False, translator=None):
    """
    Maps a permutation from the 'brl' specification to or from the '012'
    internal representation.
    """

    if not translator:
        translator = axes_dict
    if reverse:
        translator = reverse_translation_dict
    else:
        translator = translation_dict

    p = ''
    for i in permutation:
        p += translator[i]
    return p


def compose_permutations(inner, outer):
    """
    Composes the two permutations p1 and p2 as if they were functions in the
    symmetric group. That is, p3 = p1 o p2.

    Parameters
    ----------
    inner: string
        The inner (first applied) permutation, a string of 'brl'
    outer: string, None
        The outer (second applied) permutation, a string of 'brl'

    Returns
    -------
    composite, string
        The permutaton composite = outer o inner
    """

    if inner not in PERMUTATIONS:
        inner = represent_permutation(inner)
    if outer not in PERMUTATIONS:
        outer = represent_permutation(outer)

    d_inner = dict(zip("012", inner))
    d_outer = dict(zip("012", outer))
    composite = "".join(d_outer[d_inner[i]] for i in "012")
    return represent_permutation(composite, reverse=True)


def coordinate_system(self, coordinates=None, clockwise=False):
    """
    Returns the permutation corresponding to the coordinate system.
    """

    if not coordinates:
        return "012"

    # Find +/- for counterclockwise/clockwise
    if '-' in coordinates:
        clockwise = True
    coordinates = coordinates.replace('+', '')
    coordinates = coordinates.replace('-', '')

    # Check coordinates
    coordinates_set = set(coordinates)

    if coordinates_set == axes:
        rep = represent_permutation(coordinates, translator=axes_dict)
        if clockwise:
            rep = compose_permutations("210", rep)
        return (rep, clockwise)
    elif coordinates_set == corners:
        rep = represent_permutation(coordinates, translator=corners_dict)
        return (rep, clockwise)
    else:
        raise Exception("`coords` must specifies all corners or all axes.")


