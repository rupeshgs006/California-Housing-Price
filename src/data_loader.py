from sklearn.datasets import fetch_california_housing


def download_dataset():

    dataset = fetch_california_housing(
        as_frame=True
    )

    return dataset.frame