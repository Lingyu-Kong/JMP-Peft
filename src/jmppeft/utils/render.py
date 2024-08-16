import time

import ase
import nglview
from tqdm.auto import trange


def render_trajectory(atoms_list: list[ase.Atoms], disable_tqdm: bool = False):
    view = nglview.show_asetraj(atoms_list, default=False)
    view.add_unitcell()
    view.add_spacefill()
    view.camera = "orthographic"
    view.parameters = {"clipDist": 5}
    view.center()
    view.update_spacefill(radiusType="covalent", radiusScale=0.5, color_scale="rainbow")

    for frame in trange(0, len(atoms_list), disable=disable_tqdm):
        # set frame to update coordinates
        view.frame = frame
        # make sure to let NGL spending enough time to update coordinates
        time.sleep(0.5)
        view.download_image(
            filename="/workspaces/repositories/jmp-peft/docs/public-notebooks/problematic-samples_files/0image{}.png".format(
                frame
            )
        )
        # make sure to let NGL spending enough time to render before going to next frame
        time.sleep(2.0)
