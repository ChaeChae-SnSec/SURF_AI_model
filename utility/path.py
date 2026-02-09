from pathlib import Path

path_dir_root = Path(__file__).parent.parent

path_period_data = path_dir_root.joinpath('dataset/period_data_no_underscore')
path_dga_scheme = path_dir_root.joinpath('dataset/dga_scheme')
assert path_period_data.exists(), '토큰이 없는데?'

path_tokenizer = path_dir_root.joinpath('artifacts/tokenizer')
path_model = path_dir_root.joinpath('artifacts/model')
path_model.mkdir(exist_ok=True, parents=True)

path_figure = path_dir_root.joinpath(f'figure')
path_figure.mkdir(exist_ok=True, parents=True)