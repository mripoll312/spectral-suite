#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read plate reader data, subtract background, perform path length correction, plot the results, rearrange data table
structures and finally store them in a tab separated csv file.

See options via data_toolbox.py -h

Author: Niels Krausch
"""

try:
    import glob
    import logging
    import os
    import sys
    import hashlib
    import io
    import matplotlib

    import configargparse

    # Import matplotlib, but turn plotter off to not require a graphical connection
    import matplotlib as mpl

    mpl.use('Agg')

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy import optimize
except ModuleNotFoundError as e:
    print("Important modules not installed. Please install the required modules with: pip install",
          str(e).split()[-1].strip("'"))
    sys.exit(2)

logging.basicConfig(format="%(asctime)s [%(levelname)-5.5s]  %(message)s", level=logging.INFO)

# Turn interactive plotting off to not require a graphical connection
plt.ioff()
plt.switch_backend('agg')

def self_log():
    ## Log myself with hash value
    def sha256sum(src, digest_size_in_bytes=64, length=io.DEFAULT_BUFFER_SIZE):
        sha2 = hashlib.new("sha256", digest_size=digest_size_in_bytes)
        with io.open(src, mode="rb") as fd:
            for chunk in iter(lambda: fd.read(length), b''):
                sha2.update(chunk)
        return sha2.hexdigest()
    path = os.path.abspath(__file__)
    logger = logging.getLogger(__name__)

    _mypath = os.path.abspath(__file__)
    _mysha256sum = sha256sum(_mypath)
    _myfilename = os.path.basename(__file__)

    logger.info("Hi, this is: '{}' as '{}'".format(_myfilename, __name__))
    logger.info("I am located here:")
    logger.info(_mypath)
    logger.info("My sha256sum hash is:")
    logger.info(_mysha256sum)
self_log()

## end of header


class Data:
    """Class for obtaining and storing data"""

    def __init__(self):

        # Add logging
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # define list of dataframes
        self.df_list = []

        # define list of files
        self.file_list = []

        # define list of pathlength correction dataframes
        self.diff_lst = []

        # define rearranged dataframe
        self.df_rearr = pd.DataFrame(
            columns=["Time_Point", "Unique_Sample_ID", "Wavelength", "Abs", "Replicate",
                     "Condition_Name",
                     "Description"], data=None)

        # define vector with blank values
        self.blank_vals = pd.DataFrame()

        # define if path lengths should be corrected or only plotted
        self._plot_pathlength = True

        self._do_pathlength_correction = False

        # define time points of measurement
        self.timepoints = np.array([])

        # define if plots should be drawn from grouped data
        self.number_of_replicates = 1

        # define data frame for mean values
        self.df_mean = pd.DataFrame(
            columns=["Time_Point", "Unique_Sample_ID", "Wavelength", "Abs", "Condition_Name", "Description"],
            data=None)

        # define data frame for std values
        self.df_std = pd.DataFrame(
            columns=["Time_Point", "Unique_Sample_ID", "Wavelength", "Abs", "Condition_Name", "Description"],
            data=None)

        # define data frame for conversion rates
        self.conv_rates = pd.DataFrame()

        # define if raw data files should be plotted
        self._plot_raw_files = False

        # set initial concentration of the substrates
        self.initial_concs = np.array([], dtype="float")

        # flag for switching to standard preparation mode
        self.prep = False

        # set file extension of graphs
        self.graph_file_ext = ".png"

        # define number of experimental conditions
        self.number_of_conditions = 0

        self._twopoint = False

        # Try to use ggplot style
        try:
            plt.style.use('ggplot')
        except FileNotFoundError:
            print("'ggplot' style not found, will use default")
            pass

        # set flag for debugging
        self._debug = False

        # set flag for no plotting at all
        self._no_plot = False

        # store meta data
        self._meta_data = pd.DataFrame()

    def parse_args(self, argv=None):
        """
        Parse initial arguments.

        :return:
        """

        logger = logging.getLogger(__name__)

        if argv is None:
            argv = sys.argv[1:]

        p = configargparse.ArgParser(default_config_files=['~/Uni/test_config.txt'])
        p.add_argument('-c', '--config_file',
                            metavar="config_file",
                            is_config_file=True,
                            help='URI to config file')
        p.add_argument('--consider_pathlengths', dest='_do_pathlength_correction',
                            help='Consider pathlengths (default: ignore them completely)',
                            action='store_false')
        p.add_argument('--correct_pathlength',
                            help='Correct pathlength instead of only plotting them',
                            action='store_false')
        p.add_argument('-m', '--metadata_file',
                            help='URI to metadata file')
        p.add_argument('--no_plot',
                            help='Disable plotting',
                            action='store_true')
        p.add_argument('--plot_raw',
                            help='Plot raw datafiles',
                            action='store_true')
        p.add_argument('--prepare_standard',
                            help='If set, reads data to prepare a normalized \
                            standard absorption file which can be used in subsequent analysis',
                            action='store_true')
        p.add_argument('-s', '--svg',
                            help='Plot graphs as svg instead of png',
                            action='store_true')
        p.add_argument('-t', '--twopoint',
                            help='Do a two-point calibration.',
                            action='store_true')
        p.add_argument('-v', '--verbose',
                            help='Show verbose information',
                            action='store_true')

        args = p.parse_args(argv)

        if args.verbose:
            logger.setLevel(logging.DEBUG)

        self._do_pathlength_correction

        self._plot_pathlength = args.correct_pathlength

        self._plot_raw_files = args.plot_raw

        self.prep = args.prepare_standard

        if args.svg:
            self.graph_file_ext = ".svg"

        if args.no_plot:
            self._no_plot = True

        if args.twopoint:
            self._twopoint = True

        if not args.metadata_file:
            error_message = "Please link the metadata file!"
            logger.critical(error_message)
            raise(Exception(error_message))
        else:
            self._meta_data = pd.read_csv(args.metadata_file)
            self._meta_data = self._meta_data.dropna(axis='index', how='all')


        ## extract the meta data
        logger.debug("Setting the total concentrations automatically from entries found in metadata...")
        all_concentrations = self._meta_data.Total_Concentration
        self.initial_concs = np.array(all_concentrations, dtype="float")

        logger.debug("Setting the timepoints automatically from entries found in metadata...")
        all_timepoints = self._meta_data.Time_Point.unique()
        self.timepoints = np.array([_ for _ in all_timepoints if not np.isnan(_)], dtype="float")

        logger.debug("Setting the number of conditions automatically from entries found in metadata...")
        self.names_of_conditions = self._meta_data.Condition_Name.unique()
        # remove every entry with "blank" in its name
        self.names_of_conditions = [n for n in self.names_of_conditions if not "blank" in n]
        self.number_of_conditions = len(self.names_of_conditions)
        logger.debug("I set the number of conditions to {}.".format(self.number_of_conditions))

        logger.debug("Setting the number of replicates automatically from entries found in metadata...")
        self.number_of_replicates = len(self._meta_data.Replicate.unique())
        logger.debug("I set the number of replicates to {}.".format(self.number_of_replicates))

        ## check meta data for plausibility
        _tmp_error_reporter = 0
        for unique_sample_id in self._meta_data.Unique_Sample_ID.unique():
            test_results = self._meta_data[self._meta_data["Unique_Sample_ID"] == unique_sample_id]
            if len(test_results) != 1:
                logger.critical("Your metadata is inplausible, the Sample_IDs are NOT unique!")
                logger.critical("Please check these entries:")
                logger.critical(test_results)
                _tmp_error_reporter += 1
        if _tmp_error_reporter != 0:
            sys.exit(2)


        # Create folder graphs
        if not os.path.exists("Graphs"):
            os.makedirs("Graphs")

        # create folder for csv files
        if not os.path.exists("csv_files"):
            os.makedirs("csv_files")

        return


    def read_data(self):
        """Read data from csv files and store them as dataframes in a list."""

        logger = logging.getLogger(__name__)
        logger.info("Reading files...")

        # Get all tab separated files00
        all_files = self._meta_data.Data_File.unique()

        for progress_indexer, file in enumerate(all_files):
            #self.file_list.append(file)

            logger.info(f"Reading file {progress_indexer+1} of a total of {len(all_files)} files.")
            logger.info(f"Reading all data from file '{file}'")

            tmp = pd.read_csv(file, delimiter=r"\t")

            # Replace Overflow values with NaN
            tmp = tmp.replace("OVRFLW", np.nan)

            if np.any(tmp.isna()):
                logger.debug(
                    f"The data frame contains {tmp.isna().sum().sum()} Overflow values, which were replaced by NaN.")

            # Set Wavelength as index
            tmp = tmp.set_index("Wavelength")

            # Workaround where some numbers are not recognized as float
            tmp = tmp.astype(float)

            ## rename columns from Well IDs (e.g. A1, A2, ..., B9, B10, ... H11, H12) to Unique_Sample_ID
            rename_rule_dict = dict()
            all_entries_to_map = self._meta_data.query(f"Data_File == '{file}'")

            for _, i in all_entries_to_map.iterrows():
                map_from = i.Well_Number
                map_to = i.Unique_Sample_ID
                rule = { map_from : map_to }
                rename_rule_dict.update( rule )

            logger.debug("I will map with the following rule set:")
            logger.debug(rename_rule_dict)
            tmp = tmp.rename(mapper = rename_rule_dict, axis="columns")
            logger.debug("Renaming finished. Find the result below:")
            logger.debug(tmp)

            self.df_list.append(tmp)

        self.df_complete = pd.concat(self.df_list, axis=1, ignore_index=False)
        self.df_complete.to_csv("./csv_files/00_df_complete.csv")

        logger.info("Done reading files!")


    def sub_background(self):
        """
        Subtracting the background from the measured data if blank is given.

        :return:
        """
        logger = logging.getLogger(__name__)

        logger.info("Subtracting background signal from samples...")

        # Get all tab separated files
        all_blanks = self._meta_data.Blank_Unique_Sample_ID.unique()

        for name_of_current_blank in all_blanks:
            which_samples_to_blank = self._meta_data.query(f"Blank_Unique_Sample_ID == '{name_of_current_blank}'").Unique_Sample_ID
            subtract_df = self.df_complete.copy()
            subtract_df[:] = 0
            for current_sample in which_samples_to_blank:
                subtract_df[current_sample] = self.df_complete[name_of_current_blank]
            self.df_complete = self.df_complete.sub(subtract_df)

        # Drop the columns containing blank measurements
        self.df_complete = self.df_complete.drop(columns=[_ for _ in self.df_complete.columns if "blank" in _])

        self.df_complete.to_csv("./csv_files/00b_df_complete_background_subtracted_and_dropped.csv")

        logger.info("Done subtracting background signal!")



    def pathlength_correction(self):
        """
        Correct the pathlength of the sample based on Abs. at 900/977 nm.

        :return:
        """
        logger = logging.getLogger(__name__)

        logger.info("Reading files for pathlength correction...")

        # Get all tab separated files
        all_files = self._meta_data.Data_File.unique()

        pathlength_df_list = []
        for progress_indexer, file in enumerate(all_files):
            logger.debug(f"Checking pathlength correction files for {file}...")
            prefix, suffix = file.split("Read ")
            cutoff_left = prefix.rfind("/") +1
            prefix = prefix[cutoff_left:]

            all_977_files = glob.glob("./Pathlength_corr/*977.tsv")
            identified_correct_977 = [_ for _ in all_977_files if prefix in _]

            if len(identified_correct_977) != 1:
                logger.critical("Couldn't identify correct 977 file for path length correction. It is about this entry:")
                logger.critical(file)
                sys.exit(2)
            else:
                logger.debug(f"Identified 977 file for this entry -- it is '{identified_correct_977}'")

            all_900_files = glob.glob("./Pathlength_corr/*900.tsv")
            identified_correct_900 = [_ for _ in all_900_files if prefix in _]

            if len(identified_correct_900) != 1:
                logger.critical("Couldn't identify correct 900 file for path length correction. It is about this entry:")
                logger.critical(file)
                sys.exit(2)
            else:
                logger.debug(f"Identified 900 file for this entry -- it is '{identified_correct_900}'")

            f977 = pd.read_csv(identified_correct_977[0], delimiter="\t")
            f900 = pd.read_csv(identified_correct_900[0], delimiter="\t")


            # Subtract Abs_900 from Abs_977 and divide by 0.179 = deltaAbs H2O
            # this yields the pathlength in cm
            pathlength_df = f977.copy()
            pathlength_df["Mean"] = ( f977.Mean - f900.Mean ) / 0.179


            ## rename columns from Well IDs (e.g. A1, A2, ..., B9, B10, ... H11, H12) to Unique_Sample_ID
            rename_rule_dict = dict()
            all_entries_to_map = self._meta_data.query(f"Data_File == '{file}'")

            for _, i in all_entries_to_map.iterrows():
                map_from = i.Well_Number
                map_to = i.Unique_Sample_ID
                rule = { map_from : map_to }
                rename_rule_dict.update( rule )

            logger.debug("I will map with the following rule set:")
            logger.debug(rename_rule_dict)
            pathlength_df = pathlength_df.set_index('Well')
            pathlength_df = pathlength_df.rename(mapper = rename_rule_dict, axis="index")
            pathlength_df["Unique_Sample_ID"] = pathlength_df.index
            logger.debug("Renaming finished. Find the result below:")
            logger.debug(pathlength_df)

            pathlength_df_list.append(pathlength_df)

        full_pathlength_df = pd.concat(pathlength_df_list)
        full_pathlength_df = full_pathlength_df[["Unique_Sample_ID", "Mean"]]

        logger.info("Writing out the results from pathlength correction ...")
        full_pathlength_df.to_csv("./csv_files/00_sample_pathlengths.csv")
        logger.info("... done.")



        logger.info("Performing pathlength correction...")

        for current_column in self.df_complete.columns:
            m = full_pathlength_df.query(f"Unique_Sample_ID == '{current_column}'").Mean
            if len(m) != 1:
                logger.critical(f"Did not find a pathlength equivalent to {current_column}.")
            else:
                self.df_complete[current_column] = self.df_complete[current_column] / float(m)

        logger.info("... done.")


        # Print a warning when pathlength differs too much from median
        median = full_pathlength_df.Mean.median(axis=0)
        if np.any( abs(full_pathlength_df.Mean.values - median) > 0.15):
            logger.warning(
                f"Warning: The following samples differed strongly (> 0.15) in filling height from the "
                f"others:")
            for _, i in full_pathlength_df.iterrows():
                if np.any( abs(i.Mean - median) > 0.15):
                    logger.warning(i)


        ## plotting, if wished for...
        if not self._no_plot:
            df_list_complete = pd.read_csv("./csv_files/00_sample_pathlengths.csv")
            median = df_list_complete.Mean.median(axis=0)
            logger.info("Performing plotting of pathlength...")
            df_list_complete.Mean.plot(figsize=(9.5 * 0.7, 4.46 * 0.7), style="o")
            plt.xlabel("Sample No.")
            plt.ylabel("Pathlength [cm]")
            plt.title("Pathlength of the different samples", fontsize=10, fontweight="bold")
            if not min(df_list_complete.Mean) < 0.7:
                plt.ylim(median - 0.2, median + 0.2)
            else:
                plt.ylim(min(df_list_complete.Mean) - 0.1, 1)
            plt.axhline(y=median, color="black", linestyle="dashed")
            plt.axhline(y=median + 0.15, color="black", linestyle="dotted", linewidth=1)
            plt.axhline(y=median - 0.15, color="black", linestyle="dotted", linewidth=1)
            #plt.tight_layout()
            plt.savefig("./Graphs/00_Pathlength" + self.graph_file_ext, dpi=300)
            plt.close("all")


    def plot_data(self):
        """
        Plot all stored data frames and save them as PNG.

        :return:
        """
        logger = logging.getLogger(__name__)

        if self._plot_raw_files and not self._no_plot:
            logger.info("Plotting raw files...")

            # Find highest Abs for y axis
            max_abs = []
            for df in self.df_list:
                max_abs.append(df.max().max())

            highest_abs = float(round(max(max_abs), 1))

            for no, df in enumerate(self.df_list):
                df.iloc[-61:, :].plot(figsize=(9.5 * 0.7, 4.46 * 0.7), style=".")
                plt.title("Spectrum of file " + self.file_list[no][:-4], fontsize=10, fontweight="bold")
                plt.ylim((-0.025, highest_abs + 0.1))
                plt.xlabel("Wavelength [nm]")
                plt.ylabel("Abs.")
                #plt.tight_layout()
                plt.savefig("./Graphs/" + self.file_list[no][:-4] + self.graph_file_ext, dpi=300)
                plt.close("all")

            logger.info("Done plotting raw files!")

    def rearrange_data(self):
        """
        Rearrange dataframes for better handling.

        :return:
        """
        logger = logging.getLogger(__name__)

        logger.info("Start rearranging dataframes...")

        df_collector = []
        # Loop over samples in meta data, create tmp data frame, fill with data from meta data and append to df rearr
        for unique_sample_id in self._meta_data["Unique_Sample_ID"]:

            # Do not include the blanks in the rearranged data frame.
            # If "blank" is in the sample ID, we will skip to the next entry.
            if "blank" in unique_sample_id.lower():
                continue

            current_row_vals = self._meta_data[self._meta_data["Unique_Sample_ID"] == unique_sample_id]

            # Create tmp data frame
            wavelength_start = float(current_row_vals["Wavelength_Start"])
            wavelength_end   = float(current_row_vals["Wavelength_End"])
            #number_of_wavelength_entries = len(np.arange(wavelength_start, wavelength_end + 1., 1))
            tmp_df = pd.DataFrame(columns=self._meta_data.columns)

            # Fill tmp data frame
            tmp_df["Wavelength"] = np.arange(wavelength_start, wavelength_end + 1., 1)

            tmp_df["Unique_Sample_ID"] = unique_sample_id

            for col in current_row_vals.columns:
                tmp_df[col] = current_row_vals[col].values[0]

            foobar = self.df_complete[unique_sample_id].to_frame()
            foobar["Wavelength"] = foobar.index
            foobar = foobar.astype("float")
            foobar = foobar.rename(mapper = {unique_sample_id : "Abs"}, axis="columns")
            foobar["Index"] = foobar.Wavelength
            foobar = foobar.set_index("Index")


            tmp_df = tmp_df.merge(foobar, how="left", on="Wavelength")

            df_collector.append(tmp_df)

        ## join all collected dataframes into one
        self.df_rearr = pd.concat(df_collector, ignore_index=True)

        # Save rearranged data in tab-separated csv
        self.df_rearr.to_csv("./csv_files/01_rearranged_data.csv")

        logger.info("Done rearranging dataframes!")

    def query_wavelength(self, wavelength):
        """
        Returns the Absorption of a queried wavelength for all stored data.

        :param wavelength: Desired wavelength.
        :return: Abs. at given wavelength.
        """
        logger = logging.getLogger(__name__)

        abs_at_wavelegnth = self.df_rearr.loc[self.df_rearr["Wavelength"] == wavelength]

        logger.debug(abs_at_wavelegnth)

        return abs_at_wavelegnth

    def group_data(self, groupno=None):
        """
        Groups replicate experiments together.

        :param groupno: Define how many columns should be grouped.
        :return:
        """
        logger = logging.getLogger(__name__)

        if not groupno and self.number_of_replicates > 0:
            groupno = self.number_of_replicates

        logger.info(
            f"Start grouping the {groupno} replicate experiment(s) for {self.number_of_conditions} different "
            f"conditions.")

        if self.number_of_replicates > 1:

            for timepoint in self.timepoints:
                for i, name_of_condition in enumerate(self.names_of_conditions):
                    tmp_df_to_append = pd.DataFrame(columns=self.df_mean.columns.tolist())

                    # Define query command
                    query = f"Condition_Name == '{name_of_condition}' & Time_Point == {timepoint}"

                    #logger.debug("I am considering the following entries for grouping:")
                    #logger.debug(self.df_rearr.query(query))

                    # Query wavelength and same range in index no
                    wavelength_range = np.unique(self.df_rearr.query(query).loc[:, "Wavelength"])
                    wavelength_len = len(wavelength_range)

                    # First query all columns, which store identical data in every replicate
                    tmp_df_to_append = tmp_df_to_append.append(self.df_rearr.query(query).loc[:,
                                                               ["Time_Point", "Wavelength",
                                                                "Condition_Name",
                                                                "Description"]].iloc[:wavelength_len, :], sort=False,
                                                               ignore_index=True)

                    # Join sample names and put them to sample column
                    tmp_df_to_append.loc[:, "Unique_Sample_ID"] = "_".join(
                        np.unique(self.df_rearr.query(query).loc[:, "Unique_Sample_ID"].values))

                    # Add mean of Abs of replicates

                    tmp_list_means = []
                    tmp_list_std = []
                    for wavelength in self.df_rearr.query(query).Wavelength.unique():
                        all_entries_for_current_wavelength = self.df_rearr.query(query).query(f"Wavelength == '{wavelength}'")
                        #logger.debug("Considering these entries for this wavelength averaging:")
                        #logger.debug(all_entries_for_current_wavelength)
                        averaged_absorbance_for_current_wavelength = all_entries_for_current_wavelength.Abs.mean(skipna=True)
                        tmp_list_means.append( averaged_absorbance_for_current_wavelength )

                        standarddeviation_absorbance_for_current_wavelength = all_entries_for_current_wavelength.Abs.std(skipna=True)

                        if np.isnan(standarddeviation_absorbance_for_current_wavelength):
                            standarddeviation_absorbance_for_current_wavelength = 0
                        tmp_list_std.append( standarddeviation_absorbance_for_current_wavelength)



                    # Append tmp_df to df_mean
                    tmp_df_to_append.loc[:, "Abs"] = tmp_list_means
                    self.df_mean = self.df_mean.append(tmp_df_to_append, sort=False, ignore_index=True)

                    # Do the same for df_std
                    # First change Abs from mean to std
                    tmp_df_to_append.loc[:, "Abs"] = tmp_list_std
                    self.df_std = self.df_std.append(tmp_df_to_append, sort=False, ignore_index=True)

        else:
            self.df_mean = self.df_rearr.copy()
            self.df_std  = self.df_rearr.copy()
            self.df_std["Abs"] = 0


        # Store mean and std as csv files
        self.df_mean.to_csv("./csv_files/03_df_mean.csv")
        self.df_std.to_csv("./csv_files/04_df_std.csv")

    def plot_all_timepoints(self):
        """
        Plot data to every time point.

        :return:
        """
        logger = logging.getLogger(__name__)

        # Plot the samples at every time point for every experimental condition
        if not self._no_plot:
            logger.info("Start plotting...")

            for cond, condition_name in enumerate(self.names_of_conditions):
                logger.info(f"I am plotting now condition {condition_name}")
                plt.figure(figsize=(9.5 * 0.7, 4.46 * 0.7))

                logger.debug(f"I am extracting timepoints for this condition. These are:")
                entries_for_current_condition = self._meta_data.query(f"Condition_Name == '{condition_name}'")
                timepoints_for_this_condition = entries_for_current_condition.Time_Point.unique()
                logger.debug(timepoints_for_this_condition)

                for timepoint in timepoints_for_this_condition:

                    query = f"Condition_Name == '{condition_name}' & Time_Point == {timepoint} & Wavelength >= 260 & Wavelength <= 320"

                    df_mean_queried = self.df_mean.query(query)
                    if len(df_mean_queried) == 0:
                        pass
                    else:
                        logger.debug(f"... plotting timepoint {timepoint}")
                        plt.errorbar(df_mean_queried["Wavelength"], df_mean_queried["Abs"],
                                     1.96 * self.df_std.query(query)["Abs"], fmt=".", label=f"t = {timepoint}")



                plt.ylim((-0.025, 0.7))
                plt.xlabel("Wavelength [nm]")
                plt.ylabel("Abs.")
                plt.title(f"Spectrum of condition '{condition_name}'", fontsize=10, fontweight="bold")
                plt.legend(loc="best")
                #plt.tight_layout()
                plt.savefig(f"./Graphs/_cond_{cond + 1}" + self.graph_file_ext, dpi=300)
                plt.close("all")

    def standard_preparation(self):
        """
        Calculates standard absorptions for substrate and product and stores them in csv files.

        :return:
        """
        logger = logging.getLogger(__name__)

        logger.info("Starting preparation of standard...")

        df = pd.read_csv("./csv_files/00b_df_complete_background_subtracted_and_dropped.csv")
        _meta_data = self._meta_data

        which_columns_to_consider = _meta_data.Unique_Sample_ID.unique()
        which_columns_to_consider = [_ for _ in which_columns_to_consider if "blank" not in _]
        colsel = [col for col in df.columns if col in which_columns_to_consider]
        spectra_df = df.loc[: , colsel]

        #calibrate_which_wavelength = spectra_df[spectra_df.columns[0]].idxmax()
        #median_value_corrected = spectra_df.loc[calibrate_which_wavelength].median()

        for col in spectra_df.columns:
            logger.debug(f"Processing entry {col}...")

            entry_for_current_column = _meta_data.query(f"Unique_Sample_ID == '{col}'")
            assert(len(entry_for_current_column) == 1)
            ### concentration of product?
            concentration_of_substance_in_millimolar = float(entry_for_current_column.Total_Concentration)

            #absorbance_for_current_column_at_normalization_reference = spectra_df.loc[:, col].iloc[calibrate_which_wavelength]
            #spectra_df.loc[: , col] /=  absorbance_for_current_column_at_normalization_reference
            #spectra_df.loc[: , col] *=  median_value_corrected
            spectra_df.loc[: , col] /=  concentration_of_substance_in_millimolar
            ## spectra_df is now normalized to 1mM


        averaged_product_spectrum = spectra_df.mean(axis="columns")
        averaged_product_spectrum.name = "Abs"
        averaged_product_spectrum = averaged_product_spectrum.to_frame()
        averaged_product_spectrum["Wavelength"] = df.Wavelength
        averaged_product_spectrum["Abs_standard_deviation"] = spectra_df.std(axis="columns")

        #select_from = 260
        #select_to   = 320

        #averaged_product_spectrum = averaged_product_spectrum.query("{} <= Wavelength <= {}".format(select_from, select_to) )

        _filename = "out__compound_spectrum_normalized_to_1mM__xyz_uL_reaction_solution_in_xyz_uL_NaOH_xyz_mM.csv"

        averaged_product_spectrum.to_csv(_filename)

        logger.info(f"Done preparing standard. I wrote it to '{_filename}'.")

    def conversion_rate_from_twopoint_calibration(self, reference_wavelength = 277, observe_wavelength = 300):
        """
        Calculates the conversion rates, at every time point.

        :return:
        """
        logger = logging.getLogger(__name__)

        logger.info("Reading df_mean and df_std ...")
        self.df_mean = pd.read_csv("./csv_files/03_df_mean.csv")
        self.df_std  = pd.read_csv("./csv_files/04_df_std.csv")
        logger.info("... done.")

        logger.info("Starting calculation of the conversion rate...")


        ########################################################################
        ## automatic generation of reference values; currently turned off ######
        ########################################################################
        #refspec_deoxythymidine = self.df_mean.query(
        #    f"Unique_Sample_ID == 'FEK-x1817' & Wavelength >= 250 & Wavelength <= 370")
        #refspec_deoxythymidine = refspec_deoxythymidine[["Wavelength","Abs"]]
        #refspec_deoxythymidine = refspec_deoxythymidine.set_index("Wavelength")
        #refspec_thymine = self.df_mean.query(
        #    f"Unique_Sample_ID == 'FEK-x1827' & Wavelength >= 250 & Wavelength <= 370")
        #refspec_thymine = refspec_thymine[["Wavelength","Abs"]]
        #refspec_thymine = refspec_thymine.set_index("Wavelength")
        #print(refspec_thymine)
        #obs = observe_wavelength
        #ref = reference_wavelength
        #absorption_proportion_for_deoxythymidine = refspec_deoxythymidine.loc[obs] / refspec_deoxythymidine.loc[ref]
        #absorption_proportion_for_thymine        =        refspec_thymine.loc[obs] /        refspec_thymine.loc[ref]
        ########################################################################

        ## data from Niels (279,295)
        #assert(reference_wavelength == 275)
        #assert(observe_wavelength   == 295)
        #absorption_proportion_for_deoxythymidine    = 0.081416
        #absorption_proportion_for_thymine           = 0.938517

        ## data from Felix (277,300):
        assert(reference_wavelength == 277)
        assert(observe_wavelength   == 300)
        absorption_proportion_for_deoxythymidine    = 0.005115
        absorption_proportion_for_thymine           = 0.772973


        with open("csv_files/absorption_proportion_for_deoxythymidine.txt", "w") as f:
            f.write(str(absorption_proportion_for_deoxythymidine))
        with open("csv_files/absorption_proportion_for_thymine.txt", "w") as f:
            f.write(str(absorption_proportion_for_thymine))

        def model_of_absorptions(*args, **kwargs):
            params                  = kwargs.pop("p", None)
            spectrum_to_fit         = kwargs.pop("xdat")
            ref                     = kwargs.pop("ref")
            obs                     = kwargs.pop("obs")

            import uncertainties

            absorption_proportion_max_to_isosbestic  =        spectrum_to_fit.loc[obs] /        spectrum_to_fit.loc[ref]

            modelled_absorption_proportion =   params["deoxythymidine"] * absorption_proportion_for_deoxythymidine \
                                             + params["thymine"]        * absorption_proportion_for_thymine

            logger.debug("modelled_absorption_proportion:")
            logger.debug(modelled_absorption_proportion)

            modelled_absorption                     = spectrum_to_fit[:]
            modelled_absorption                     = modelled_absorption.loc[[ref,obs]]
            modelled_absorption["Std"]              = 0
            modelled_absorption["Std"].loc[ref]     = 0


            predicted                               = modelled_absorption.loc[ref]["Abs"] * modelled_absorption_proportion
            predicted                               = float(predicted)
            modelled_absorption["Abs"].loc[obs]     = predicted
            modelled_absorption["Std"].loc[obs]     = 0

            logger.debug("modelled_absorption")
            logger.debug(modelled_absorption)

            return modelled_absorption


        def fitting_eq(params, spectrum_to_fit, ref, obs, *args, **kwargs):
            absorption_proportion_max_to_isosbestic  =        spectrum_to_fit.loc[obs] /        spectrum_to_fit.loc[ref]

            modelled_absorption_proportion =   params["deoxythymidine"] * absorption_proportion_for_deoxythymidine \
                                             + params["thymine"]        * absorption_proportion_for_thymine

            logger.debug("modelled_absorption_proportion with uncertainties:")
            logger.debug(modelled_absorption_proportion)

            value           = float(modelled_absorption_proportion)
            uncertainty     = 1

            diff_to_data = ( absorption_proportion_max_to_isosbestic - value )/uncertainty

            logger.debug("diff_to_data:")
            logger.debug(diff_to_data)

            return diff_to_data

        def callback_iter(params, iter, resid, *fcn_args, **fcn_kws):
            logger.debug("callback_iter")
            r = resid**2
            logger.debug("iter: {}, residual: {}, params: {}".format(iter, r, params.valuesdict()))
            return

        # Perform curve fitting and plot results
        if not self._no_plot:
            logger.info("Performing curve fitting and plotting...")

        else:
            logger.info("Performing curve fitting...")


        for condition, condition_name in enumerate(self.names_of_conditions):
            logger.info(f"Fitting condition '{condition_name}'...")
            logger.debug(f"I am extracting timepoints for this condition. These are:")
            entries_for_current_condition = self._meta_data.query(f"Condition_Name == '{condition_name}'")
            timepoints_for_this_condition = entries_for_current_condition.Time_Point.unique()
            logger.debug(timepoints_for_this_condition)
            for timepoint in timepoints_for_this_condition:
                logger.debug(f"... fitting at timepoint {timepoint}...")

                _metadata_for_current_timepoint = entries_for_current_condition.query(f"Time_Point == {timepoint}")

                assert(len(_metadata_for_current_timepoint) == 1)

                fit_from_wavelength = int(_metadata_for_current_timepoint.Wavelength_Start.iloc[0])
                fit_to_wavelength   = int(_metadata_for_current_timepoint.Wavelength_End.iloc[0])

                curve_to_fit = self.df_mean.query(
                    f"Condition_Name == '{condition_name}' & Time_Point == {timepoint} & Wavelength >= {fit_from_wavelength} & Wavelength "
                    f"<= {fit_to_wavelength}")
                curve_to_fit = curve_to_fit[["Wavelength", "Abs"]]
                curve_to_fit = curve_to_fit.set_index("Wavelength")

                standard_deviations = self.df_std.query(
                    f"Condition_Name == '{condition_name}' & Time_Point == {timepoint} & Wavelength >= {fit_from_wavelength} & Wavelength "
                    f"<= {fit_to_wavelength}")[["Wavelength", "Abs"]]
                #standard_deviations = [std if std!= 0 else np.finfo(float).eps for std in standard_deviations]

                logger.debug("curve_to_fit:")
                logger.debug(curve_to_fit)

                import lmfit
                params = lmfit.Parameters()
                params.add("deoxythymidine", value=0.5, min=0, max=1)
                params.add("thymine",   expr= '1 - deoxythymidine')


                minimizer        = lmfit.Minimizer(fitting_eq, params, fcn_args=(curve_to_fit, reference_wavelength, observe_wavelength), iter_cb = callback_iter)
                minimizer_result = minimizer.minimize(method="least_squares")

                logger.info(lmfit.fit_report(minimizer_result))

                ## stopped working at some point ...
                #model         = lmfit.Model(model_of_absorptions, nan_policy="propagate")
                #modelled_data = model.eval(p=minimizer_result.params, xdat=curve_to_fit, ref=reference_wavelength, obs=observe_wavelength)
                ## workaround:
                modelled_data = model_of_absorptions(p=minimizer_result.params, xdat=curve_to_fit, ref=reference_wavelength, obs=observe_wavelength)

                if not self._no_plot:
                    fig = plt.figure(figsize=(9.5 * 0.7, 4.46 * 0.7))
                    ax  = plt.subplot(111)
                    ax.plot( curve_to_fit.index,  curve_to_fit["Abs"], 'b.')
                    ax.plot(modelled_data.index, modelled_data["Abs"], 'rs') #yerr=1.96*modelled_data["Std"], marker='s', mfc='red', linewidth=None, capsize=12, elinewidth=2 )
                    plt.xlabel("Wavelength [nm]")
                    plt.ylabel("Abs.")
                    plt.title(
                        f"Fit of condition '{condition_name}' and time point t{timepoint}\nwith parameters"
                        f"{minimizer_result.params.valuesdict()}", fontsize=10, fontweight="bold")
                    plt.legend(
                        [f"+Enz, condition {condition + 1}, t{timepoint}", "Fitted curve"],
                        loc="best")
                    #plt.tight_layout()
                    plt.savefig(
                        f"./Graphs/fit_cond_{condition + 1}_t_{timepoint}" + self.graph_file_ext,
                        dpi=300)
                    plt.close("all")


                # Store conversion rates
                tmp_df = pd.DataFrame(columns=["Condition_Name", "Time_Point"]+list(params.keys())+["stderr_{}".format(key) for key in params.keys()], index=[0])

                tmp_df.iloc[0, :] = condition_name, timepoint, *[ minimizer_result.params[key].value for key in params.keys() ],\
                                                       *[ minimizer_result.params[key].stderr for key in params.keys() ]

                logger.debug(f"Current data frame to append: {tmp_df}")

                self.conv_rates = self.conv_rates.append(tmp_df, ignore_index=True)

        if not self._no_plot:
            logger.info("Done with curve fitting and plotting!")
        else:
            logger.info("Done with curve fitting!")

        plt.close("all")

        # Store conversion rates and covariances
        self.conv_rates.to_csv("./csv_files/08_conv_rates.csv")


    def conversion_rate(self):
        self.conv_rates = pd.DataFrame()
        self.conversion_rate_from_spectral_unmixing()
        return

    def conversion_rate_from_spectral_unmixing(self):
        """
        Calculates the conversion rates, at every time point.

        :return:
        """
        logger = logging.getLogger(__name__)

        logger.info("Reading df_mean and df_std ...")
        self.df_mean = pd.read_csv("./csv_files/03_df_mean.csv")
        self.df_std  = pd.read_csv("./csv_files/04_df_std.csv")
        logger.info("... done.")

        logger.info("Starting calculation of the conversion rate...")

        def model_of_absorptions(*args, **kwargs):
            params                  = kwargs.pop("p", None)
            spectrum_to_fit         = kwargs.pop("xdat")
            reference_spectra_dict  = kwargs.pop("refspec")
            if params is None:
                params = lmfit.Parameters()
                for k in kwargs.keys():
                    params.add(k, value=kwargs[k])

            modelled_absorption           = spectrum_to_fit[:]
            modelled_absorption["Abs"]    = 0

            for compound_name in params.keys() - ["scaling"]:
                for wavelength_index in modelled_absorption.index:
                    modelled_absorption.loc[wavelength_index] += reference_spectra_dict[compound_name]["Abs"].loc[wavelength_index] * params[compound_name].value
            modelled_absorption["Abs"] *= params["scaling"].value

            return modelled_absorption


        def fitting_eq(params, reference_spectra_dict, spectrum_to_fit):
            modelled_data = model_of_absorptions(xdat= spectrum_to_fit, p=params, refspec=reference_spectra_dict)
            uncertainty = 1.0
            diff_to_data = (spectrum_to_fit - modelled_data )/uncertainty
            return diff_to_data

        def callback_iter(params, iter, resid, *fcn_args, **fcn_kws):
            logger.debug("callback_iter")

            r = sum([rd["Abs"]**2 for _, rd in resid.iterrows()])
            logger.debug("iter: {}, residual: {}, params: {}".format(iter, r, params.valuesdict()))
            return

        # Perform curve fitting and plot results
        if not self._no_plot:
            logger.info("Performing curve fitting and plotting...")

        else:
            logger.info("Performing curve fitting...")


        for condition, condition_name in enumerate(self.names_of_conditions):
            logger.info(f"Fitting condition '{condition_name}'...")
            logger.debug(f"I am extracting timepoints for this condition. These are:")
            entries_for_current_condition = self._meta_data.query(f"Condition_Name == '{condition_name}'")
            timepoints_for_this_condition = entries_for_current_condition.Time_Point.unique()
            logger.debug(timepoints_for_this_condition)
            for timepoint in timepoints_for_this_condition:
                logger.debug(f"... fitting at timepoint {timepoint}...")

                _metadata_for_current_timepoint = entries_for_current_condition.query(f"Time_Point == {timepoint}")

                assert(len(_metadata_for_current_timepoint) == 1)

                isosbestic_point_wavelength = _metadata_for_current_timepoint.Isosbestic_Point.iloc[0]
                if type(isosbestic_point_wavelength) == str:
                    if isosbestic_point_wavelength == "":
                        isosbestic_point_wavelength = int(0)
                    else:
                        isosbestic_point_wavelength = int(isosbestic_point_wavelength)
                else:
                    isosbestic_point_wavelength = int(isosbestic_point_wavelength)


                which_reference_compounds_columns = []
                for col in _metadata_for_current_timepoint.columns:
                    if "Reference_Spectrum_" in col:
                        value = _metadata_for_current_timepoint[col].iloc[0]
                        if type(value) == str:
                            if value.strip() != "":
                                which_reference_compounds_columns.append(col)

                reference_spectra_dict = {}
                for col in which_reference_compounds_columns:
                    this_spectrum_filename = _metadata_for_current_timepoint[col].iloc[0]
                    if this_spectrum_filename:
                        ## is not NaN
                        tmp_df = pd.read_csv(this_spectrum_filename)
                        tmp_df = tmp_df.set_index("Wavelength")
                        if isosbestic_point_wavelength > 0:
                            tmp_df /= tmp_df.loc[isosbestic_point_wavelength]["Abs"]
                        name = col[len("Reference_Spectrum_"):]
                        reference_spectra_dict.update({ name : tmp_df })

                fit_from_wavelength = int(_metadata_for_current_timepoint.Wavelength_Start.iloc[0])
                fit_to_wavelength   = int(_metadata_for_current_timepoint.Wavelength_End.iloc[0])

                curve_to_fit = self.df_mean.query(
                    f"Condition_Name == '{condition_name}' & Time_Point == {timepoint} & Wavelength >= {fit_from_wavelength} & Wavelength "
                    f"<= {fit_to_wavelength}")
                curve_to_fit = curve_to_fit[["Wavelength", "Abs"]]
                curve_to_fit = curve_to_fit.set_index("Wavelength")
                if isosbestic_point_wavelength > 0:
                    curve_to_fit /= curve_to_fit.loc[isosbestic_point_wavelength]["Abs"]

                standard_deviations = self.df_std.query(
                    f"Condition_Name == '{condition_name}' & Time_Point == {timepoint} & Wavelength >= {fit_from_wavelength} & Wavelength "
                    f"<= {fit_to_wavelength}")[["Wavelength", "Abs"]]
                for idx in standard_deviations.index:
                    if standard_deviations["Abs"].loc[idx] == 0:
                        standard_deviations["Abs"].loc[idx] = np.finfo(float).eps
                    else:
                        if isosbestic_point_wavelength > 0:
                            standard_deviations["Abs"].loc[idx] /= curve_to_fit.loc[isosbestic_point_wavelength]["Abs"]

                logger.debug(curve_to_fit)

                import lmfit
                params = lmfit.Parameters()
                last_substance_name = ""
                for substance_name in reference_spectra_dict:
                    startvalue = 1.0/len(reference_spectra_dict.keys())
                    params.add(substance_name, min=0, max=1, value=startvalue)
                    last_substance_name = substance_name
                expression_construction = "1"
                for i in reference_spectra_dict:
                    if i == last_substance_name:
                        continue
                    expression_construction += " - {}".format(i)
                params[last_substance_name].set(expr=expression_construction)
                if isosbestic_point_wavelength < 0:
                    params.add("scaling", min=0.0, max=9.0, value=1.0, vary=True)
                else:
                    params.add("scaling", min=0.0, max=9.0, value=1.0, vary=False)

                minimizer        = lmfit.Minimizer(fitting_eq, params, fcn_args=(reference_spectra_dict, curve_to_fit), iter_cb = callback_iter)
                minimizer_result = minimizer.minimize(method="least_squares")

                logger.info(lmfit.fit_report(minimizer_result))

                ## stopped working at some point ...
                #model         = lmfit.Model(model_of_absorptions, nan_policy="propagate")
                #modelled_data = model.eval(p=minimizer_result.params, refspec=reference_spectra_dict, xdat=curve_to_fit)
                ## workaround:
                modelled_data = model_of_absorptions(p=minimizer_result.params, refspec=reference_spectra_dict, xdat=curve_to_fit)

                if not self._no_plot:
                    fig = plt.figure(figsize=(9.5 * 0.7, 4.46 * 0.7))
                    plt.plot( curve_to_fit.index,  curve_to_fit["Abs"]) #, fmt=".") #, standard_deviations)
                    plt.plot(modelled_data.index, modelled_data["Abs"]) #, 1.96 * np.sqrt(np.diag(pcov)[0] * combination))
                    #plt.ylim((-0.025, 0.7))
                    plt.xlabel("Wavelength [nm]")
                    plt.ylabel("Abs.")
                    plt.title(
                        f"Fit of condition '{condition_name}' and time point t{timepoint}\nwith parameters"
                        f"{minimizer_result.params.valuesdict()}", fontsize=10, fontweight="bold")
                    plt.legend(
                        [f"+Enz, condition {condition + 1}, t{timepoint}", "Fitted curve"],
                        loc="best")
                    #plt.tight_layout()
                    plt.savefig(
                        f"./Graphs/fit_cond_{condition + 1}_t_{timepoint}" + self.graph_file_ext,
                        dpi=300)
                    plt.close(fig)

                # Store conversion rates
                tmp_df = pd.DataFrame(columns=["Condition_Name", "Time_Point"]+list(params.keys())+["stderr_{}".format(key) for key in params.keys()], index=[0])

                tmp_df.iloc[0, :] = condition_name, timepoint, *[ minimizer_result.params[key].value for key in params.keys() ],\
                                                       *[ minimizer_result.params[key].stderr for key in params.keys() ]

                logger.debug(f"Current data frame to append: {tmp_df}")

                self.conv_rates = self.conv_rates.append(tmp_df, ignore_index=True)

        if not self._no_plot:
            logger.info("Done with curve fitting and plotting!")
        else:
            logger.info("Done with curve fitting!")

        plt.close("all")

        # Store conversion rates and covariances
        self.conv_rates.to_csv("./csv_files/08_conv_rates.csv")

    def plot_conversion_rates(self):
        """
        Plots the conversion rates with 95% CI at different time points.

        :return:
        """
        logger = logging.getLogger(__name__)

        logger.info("Reading the stored conversion rates ...")
        self.conv_rates = pd.read_csv("./csv_files/08_conv_rates.csv")
        logger.info("... Done.")

        logger.info("Starting plotting of conversion rates...")

        which_reference_compounds = [col[len("Reference_Spectrum_"):] for col in self._meta_data.columns if "Reference_Spectrum_" in col]

        # Plot conversion rates with 95% CI at different time points
        if not self._no_plot:
            for index, condition_name in enumerate(self.names_of_conditions):
                logger.debug(f"... for condition '{condition_name}'...")
                logger.debug(f"I am extracting timepoints for this condition. These are:")
                entries_for_current_condition = self._meta_data.query(f"Condition_Name == '{condition_name}'")
                timepoints_for_this_condition = entries_for_current_condition.Time_Point.unique()
                logger.debug(timepoints_for_this_condition)

                for compound in which_reference_compounds:
                    plt.figure(figsize=(9.5 * 0.7, 4.46 * 0.7))
                    for timepoint in timepoints_for_this_condition:
                        query = f"Condition_Name == '{condition_name}' & Time_Point == {timepoint}"
                        plt.errorbar(timepoint, 100 * (
                                1 - self.conv_rates.query(query)[compound].values[0]), 1.96 *
                            abs(self.conv_rates.query(query)["stderr_"+compound].values[0]), marker="o",
                                     color="#348abdff")
                    plt.ylim(0, 100)
                    plt.xlabel("Time [min]")
                    plt.ylabel("Conversion rate [%]")
                    plt.title("Conversion rate over time \n {}".format(condition_name), fontsize=10, fontweight="bold")
                    #plt.tight_layout()
                    plt.savefig(f"./Graphs/01_Conversion_rates_cond_{index + 1}_{compound}" + self.graph_file_ext, dpi=300)
                    plt.close("all")
        self.store_individual_concentration_values()

    def store_individual_concentration_values(self):
        """
        Stores the concentration values in individual csv files.

        :return:
        """
        logger = logging.getLogger(__name__)

        logger.info("Reading the stored conversion rates ...")
        self.conv_rates = pd.read_csv("./csv_files/08_conv_rates.csv")
        logger.info("... Done.")
        # Store concentrations of substrates

        which_reference_compounds = [col[len("Reference_Spectrum_"):] for col in self._meta_data.columns if "Reference_Spectrum_" in col]

        tmp_original_df = pd.DataFrame(columns=["Condition_Name", "Time_Point"]+list(which_reference_compounds)+["stderr_{}".format(key) for key in which_reference_compounds], index=[0])
        for index, condition_name in enumerate(self.names_of_conditions):
            self.concs = pd.DataFrame()

            entries_for_current_condition = self._meta_data.query(f"Condition_Name == '{condition_name}'")

            timepoints_for_this_condition = entries_for_current_condition.Time_Point.unique()

            for timepoint in timepoints_for_this_condition:
                query = f"Condition_Name == '{condition_name}' & Time_Point == {timepoint}"
                assert(len(self.conv_rates.query(query)) == 1)
                total_substance_concentration_for_this_condition = float(self._meta_data.query(query).Total_Concentration)

                tmp_df = tmp_original_df.copy()
                tmp_df["Condition_Name"] = condition_name
                tmp_df["Time_Point"]     = timepoint
                for compound in which_reference_compounds:
                    tmp_df[compound]             =  total_substance_concentration_for_this_condition * \
                                                    self.conv_rates.query(query)[compound].values[0]
                    tmp_df["stderr_"+compound]   =  total_substance_concentration_for_this_condition * \
                                                    self.conv_rates.query(query)["stderr_"+compound].values[0]

                self.concs = self.concs.append(tmp_df, ignore_index=True)

            # Store as csv
            self.concs.to_csv(f"./csv_files/07_concentrations_condition_{index + 1}.csv")

        logger.info("Done!")


def main():
    data = Data()
    data.parse_args()
    data.read_data()
    data.sub_background()
    if data._do_pathlength_correction:
        data.pathlength_correction()

    if data.prep:
        data.standard_preparation()
    else:
        data.plot_data()
        data.rearrange_data()
        data.group_data()
        data.plot_all_timepoints()
        if data._twopoint:
            data.conversion_rate_from_twopoint_calibration()
        else:
            data.conversion_rate()
        data.plot_conversion_rates()
        plt.close("all")


if __name__ == "__main__":
    main()
