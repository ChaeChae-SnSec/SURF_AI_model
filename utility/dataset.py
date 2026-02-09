import polars as pl
pl.Config.set_engine_affinity(engine="streaming")

def get_train_set_pretrain_tld() :
    files = [
        'dataset/tld/T17_benign_tld.parquet',
        'dataset/tld/T18_benign_tld.parquet',
        'dataset/tld/T19_benign_tld.parquet',
        'dataset/tld/T17_dga_tld.parquet',
        'dataset/tld/T18_dga_tld.parquet',
        'dataset/tld/T19_dga_tld.parquet',
    ]

    return pl.read_parquet(files).unique()

def get_train_set_tld():
    files = [
        'dataset/tld/T17_benign_tld_test.parquet',
        'dataset/tld/T17_dga_tld_test.parquet',
        'dataset/tld/T18_benign_tld_test.parquet',
        'dataset/tld/T18_dga_tld_test.parquet',
        'dataset/tld/T19_benign_tld_test.parquet',
        'dataset/tld/T19_dga_tld_test.parquet',

        'dataset/tld/T17_benign_tld_train.parquet',
        'dataset/tld/T17_dga_tld_train.parquet',
        'dataset/tld/T18_benign_tld_train.parquet',
        'dataset/tld/T18_dga_tld_train.parquet',
        'dataset/tld/T19_benign_tld_train.parquet',
        'dataset/tld/T19_dga_tld_train.parquet',
        ]

    return pl.read_parquet(files).unique(), files

def get_val_set_tld():
    files = [
        'dataset/tld/T17_benign_tld_val.parquet',
        'dataset/tld/T17_dga_tld_val.parquet',
        'dataset/tld/T18_benign_tld_val.parquet',
        'dataset/tld/T18_dga_tld_val.parquet',
        'dataset/tld/T19_benign_tld_val.parquet',
        'dataset/tld/T19_dga_tld_val.parquet',
        ]

    return pl.read_parquet(files).unique()

def get_test_set_20_tld():
    files = [
        'dataset/tld/T20_benign_tld.parquet',
        'dataset/tld/T20_dga_tld.parquet',
        ]

    return pl.read_parquet(files).unique()

def get_test_set_21_tld():
    files = [
        'dataset/tld/T21_benign_tld.parquet',
        'dataset/tld/T21_dga_tld.parquet',
        ]

    return pl.read_parquet(files).unique()

def get_test_set_22_tld():
    files = [
        'dataset/tld/T22_benign_tld.parquet',
        'dataset/tld/T22_dga_tld.parquet',
        ]

    return pl.read_parquet(files).unique()

def get_test_set_23_tld():
    files = [
        'dataset/tld/T23_benign_tld.parquet',
        'dataset/tld/T23_dga_tld.parquet',
        ]

    return pl.read_parquet(files).unique()

def get_test_set_24_tld():
    files = [
        'dataset/tld/T24_benign_tld.parquet',
        ]

    return pl.read_parquet(files).unique()

def get_test_set_25_tld():
    files = [
        'dataset/tld/T25_benign_tld.parquet',
        ]

    return pl.read_parquet(files).unique()



def get_train_set_pretrain() :
    files = [
        'dataset/period_data_no_underscore/T17_benign.parquet',
        'dataset/period_data_no_underscore/T18_benign.parquet',
        'dataset/period_data_no_underscore/T19_benign.parquet',
        'dataset/period_data_no_underscore/T17_dga.parquet',
        'dataset/period_data_no_underscore/T18_dga.parquet',
        'dataset/period_data_no_underscore/T19_dga.parquet',
    ]

    return pl.read_parquet(files).unique()

def get_train_set():
    files = [
        'dataset/period_data_no_underscore/T17_benign_test.parquet',
        'dataset/period_data_no_underscore/T17_dga_test.parquet',
        'dataset/period_data_no_underscore/T18_benign_test.parquet',
        'dataset/period_data_no_underscore/T18_dga_test.parquet',
        'dataset/period_data_no_underscore/T19_benign_test.parquet',
        'dataset/period_data_no_underscore/T19_dga_test.parquet',

        'dataset/period_data_no_underscore/T17_benign_train.parquet',
        'dataset/period_data_no_underscore/T17_dga_train.parquet',
        'dataset/period_data_no_underscore/T18_benign_train.parquet',
        'dataset/period_data_no_underscore/T18_dga_train.parquet',
        'dataset/period_data_no_underscore/T19_benign_train.parquet',
        'dataset/period_data_no_underscore/T19_dga_train.parquet',
        ]

    return pl.read_parquet(files).unique()

def get_val_set():
    files = [
        'dataset/period_data_no_underscore/T17_benign_val.parquet',
        'dataset/period_data_no_underscore/T17_dga_val.parquet',
        'dataset/period_data_no_underscore/T18_benign_val.parquet',
        'dataset/period_data_no_underscore/T18_dga_val.parquet',
        'dataset/period_data_no_underscore/T19_benign_val.parquet',
        'dataset/period_data_no_underscore/T19_dga_val.parquet',
        ]

    return pl.read_parquet(files).unique()

def get_test_set_20():
    files = [
        'dataset/period_data_no_underscore/T20_benign.parquet',
        'dataset/period_data_no_underscore/T20_dga.parquet',
        ]

    return pl.read_parquet(files).unique()

def get_test_set_21():
    files = [
        'dataset/period_data_no_underscore/T21_benign.parquet',
        'dataset/period_data_no_underscore/T21_dga.parquet',
        ]

    return pl.read_parquet(files).unique()

def get_test_set_22():
    files = [
        'dataset/period_data_no_underscore/T22_benign.parquet',
        'dataset/period_data_no_underscore/T22_dga.parquet',
        ]

    return pl.read_parquet(files).unique()

def get_test_set_23():
    files = [
        'dataset/period_data_no_underscore/T23_benign.parquet',
        'dataset/period_data_no_underscore/T23_dga.parquet',
        ]

    return pl.read_parquet(files).unique()

def get_test_set_24():
    files = [
        'dataset/period_data_no_underscore/T24_benign.parquet',
        ]

    return pl.read_parquet(files).unique()

def get_test_set_25():
    files = [
        'dataset/period_data_no_underscore/T25_benign.parquet',
        ]

    return pl.read_parquet(files).unique()


if __name__ == '__main__':
    print(get_train_set())
    for function in [
        get_train_set,
        get_val_set,
        get_test_set_20,
        get_test_set_21,
        get_test_set_22,
        get_test_set_23,
        get_test_set_24,
        get_test_set_25,
    ]:
        df = function()
        print(function.__name__, df.shape, )