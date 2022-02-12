import yaml
import os


class CVMetaData:
    def __init__(self, argv):
        print("Initializing data...")
        # Read the data
        if len(argv) > 1:
            self._filename = argv[1]
        else:
            self._filename = "./control_volume_analysis.csv"
        self._working_dir = os.getcwd()

        # Read config file
        self._open_glottis = [0.0, 1.0]
        if len(argv) > 2:
            config_filename = argv[2]
        else:
            config_filename = self._filename.replace(
                "control_volume_analysis.csv", "plot_settings.yaml")
        with open(config_filename) as plot_configs:
            self._documents = yaml.full_load(plot_configs)
            self._open_glottis[0] = self._documents["open phase"]
            self._open_glottis[1] = self._documents["close phase"]
            self._output_dir = os.path.join(
                self._working_dir, self._documents["output directory"])
            # Create dir if not exist
            if not os.path.exists(self._output_dir):
                print("Output directory doesn't exist, will create the directory...")
                os.mkdir(self._output_dir)
            self._timespan = self._documents["time span"]
            self._n_period = 1.0
            if "period" in self._documents:
                self._n_period = self._documents["period"]

            if "size" in self._documents:
                self._size = self._documents["size"]
            else:
                self._size = {"height": 11.25,
                              "width": 22, "left": 0.2, "right": 0.8}

            # Average behavior reader
            if "average" in self._documents:
                self._cases = self._documents["average"]

            for item, doc in self._documents.items():
                print(item, ":", doc)

    @property
    def documents(self):
        return self._documents

    @property
    def open_glottis(self):
        return self._open_glottis

    @property
    def timespan(self):
        return self._timespan

    @property
    def filename(self):
        return self._filename

    @property
    def output_dir(self):
        return self._output_dir

    @property
    def working_dir(self):
        return self._working_dir

    @property
    def n_period(self):
        return self._n_period

    @property
    def cases(self):
        return self._cases

    @property
    def size(self):
        return self._size
