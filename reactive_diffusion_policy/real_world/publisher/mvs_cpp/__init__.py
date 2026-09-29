__all__ = [
    "MVSImageCamera",
    "MVSCamControllerConfig",
    "get_all_mvs_dev_serial",
]


def __getattr__(name):
    if name in __all__:
        from .mvs import MVSImageCamera, MVSCamControllerConfig, get_all_mvs_dev_serial

        return {
            "MVSImageCamera": MVSImageCamera,
            "MVSCamControllerConfig": MVSCamControllerConfig,
            "get_all_mvs_dev_serial": get_all_mvs_dev_serial,
        }[name]
    raise AttributeError(name)
