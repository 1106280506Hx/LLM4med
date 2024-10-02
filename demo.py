import tqdm, tqdm.notebook
tqdm.tqdm = tqdm.notebook.tqdm  # notebook-friendly progress bars
from pathlib import Path

from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_exhaustive, match_dense, pairs_from_retrieval
from hloc.visualization import plot_images, read_image
from hloc.utils import viz_3d
import gc

images = Path('E:\\kaggle\\input\\image-matching-challenge-2024\\test\\church\\images')
outputs = Path('/kaggle/working/demo')
#!rm -rf $outputs
sfm_pairs = outputs / 'pairs-sfm.txt'
loc_pairs = outputs / 'pairs-loc.txt'
sfm_dir = outputs / 'sfm'
features = outputs / 'keypoints.h5'
matches = outputs / 'matches.h5'
reference_sfm = outputs / "sfm_loftr"

# feature_conf = extract_features.confs['superpoint_aachen']
# matcher_conf = match_features.confs['superpoint+lightglue']

if __name__ == "__main__":

    retrieval_conf = extract_features.confs["disk"]
    matcher_conf = match_dense.confs["loftr"]

    references = [str(p.relative_to(images)) for p in (images).iterdir()]
    print(len(references), "mapping images")
    plot_images([read_image(images / r) for r in references], dpi=25)

    # extract_features.main(retrieval_conf, images, image_list=references, feature_path=features)
    global_descriptors = extract_features.main(retrieval_conf, images, outputs)
    pairs_from_exhaustive.main(sfm_pairs, image_list=references)
    match_dense.main(matcher_conf, sfm_pairs, images, outputs, features=features, matches=matches);

    model = reconstruction.main(sfm_dir, images, sfm_pairs, features, matches, image_list=references, mapper_options={"min_model_size": 3, # By default colmap does not generate a reconstruction if less than 10 images are registered. Lower it to 3.
            "max_num_models": 2})
    gc.collect()

    model

    fig = viz_3d.init_figure()
    viz_3d.plot_reconstruction(fig, model, color='rgba(255,0,0,0.5)', name="mapping", points_rgb=True)
    fig.show()