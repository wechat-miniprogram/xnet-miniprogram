class VersionAgnosticNames:
    RELU = "RELU"


class TorchOpInfo:
    def __init__(self, torch_version: str, version_agnostic_name: str):
        self.torch_version = torch_version
        self.version_agnostic_name = version_agnostic_name


OPERATOR_NAME_LOOKUP_TABLE = {
    "relu_"     : TorchOpInfo("1.1.0", VersionAgnosticNames.RELU),
    "relu"      : TorchOpInfo("unknown", VersionAgnosticNames.RELU)
}


def get_version_agnostic_name(version_specific_name: str):
    if version_specific_name not in OPERATOR_NAME_LOOKUP_TABLE:
        return version_specific_name

    return OPERATOR_NAME_LOOKUP_TABLE[version_specific_name].version_agnostic_name
