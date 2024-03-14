import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--ckpt",
    type=str,
    default="./models/Stable-diffusion/v1-5-pruned-emaonly.ckpt",
    help="path to checkpoint of model",
)
parser.add_argument(
    "--n_samples",
    type=int,
    default=4,
    help="how many samples to produce for each given prompt. A.k.a. batch size",
)
parser.add_argument(
    "--n_iter",
    type=int,
    default=1,
    help="sample this often",
)
parser.add_argument(
    "--prompt",
    type=str,
    nargs="?",
    default="a painting of a virus monster playing guitar",
    help="the prompt to render"
)
parser.add_argument(
    "--outdir",
    type=str,
    nargs="?",
    help="dir to write results to",
    default="outputs/20230905"
)
parser.add_argument(
    "--skip_grid",
    action='store_true',
    help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    default=True
)
parser.add_argument(
    "--skip_save",
    action='store_true',
    help="do not save individual samples. For speed measurements.",
)
parser.add_argument(
    "--ddim_steps",
    type=int,
    default=20,
    help="number of ddim sampling steps",
)
parser.add_argument(
    "--plms",
    action='store_true',
    help="use plms sampling",
)
parser.add_argument(
    "--dpm_solver",
    action='store_true',
    help="use dpm_solver sampling",
)
parser.add_argument(
    "--laion400m",
    action='store_true',
    help="uses the LAION400M model",
)
parser.add_argument(
    "--fixed_code",
    action='store_true',
    help="if enabled, uses the same starting code across samples ",
)
parser.add_argument(
    "--ddim_eta",
    type=float,
    default=0.0,
    help="ddim eta (eta=0.0 corresponds to deterministic sampling",
)

parser.add_argument(
    "--H",
    type=int,
    default=512,
    help="image height, in pixel space",
)
parser.add_argument(
    "--W",
    type=int,
    default=512,
    help="image width, in pixel space",
)
parser.add_argument(
    "--C",
    type=int,
    default=4,
    help="latent channels",
)
parser.add_argument(
    "--f",
    type=int,
    default=8,
    help="downsampling factor",
)

parser.add_argument(
    "--n_rows",
    type=int,
    default=0,
    help="rows in the grid (default: n_samples)",
)
parser.add_argument(
    "--scale",
    type=float,
    default=7.5,
    help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
)
parser.add_argument(
    "--from-file",
    type=str,
    help="if specified, load prompts from this file",
)
parser.add_argument(
    "--config",
    type=str,
    default="./configs/stable-diffusion/v1-inference.yaml",
    help="path to config which constructs model",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="the seed (for reproducible sampling)",
)
parser.add_argument(
    "--precision",
    type=str,
    help="evaluate at this precision",
    choices=["full", "autocast"],
    default="autocast"
)
parser.add_argument(
    "--from_file",
    type=str,
    default=None
)
opt = parser.parse_args()